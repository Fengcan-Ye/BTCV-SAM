import json 
import nibabel as nib
import os 
from segment_anything import sam_model_registry
from sam_btcv import SamBTCV
from dataset import to_uint8_rgb, remove_pure_background, to_tensor
from segment_anything.utils.transforms import ResizeLongestSide
from utils import *

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')


    with open('./data/dataset_0.json', 'r') as f:
        dataset_info = json.load(f)
    
    sam = sam_model_registry['default']('./sam_vit_h_4b8939.pth')
    model = SamBTCV(sam, requires_classification=True)
    model.load_state_dict(torch.load('./cls/best_model.pth'))
    model.to(device)

    resize_transform = ResizeLongestSide(model.image_encoder.img_size)

    preprocess = lambda images, labels: to_uint8_rgb(*remove_pure_background(images, labels))

    points_along_side = 32
    predicted_labels = []
    gt_labels = []

    for data in dataset_info['validation']:
        images = nib.load(os.path.join('./data', data['image'])).get_fdata()
        labels = nib.load(os.path.join('./data', data['label'])).get_fdata()

        images, labels = preprocess(images, labels) # [B, H, W, C]

        original_size = images.shape[1 : 3]

        xs = np.linspace(0, original_size[1], points_along_side, dtype=np.uint32).reshape((-1, 1))
        ys = np.linspace(0, original_size[0], points_along_side, dtype=np.uint32).reshape((-1, 1))

        # shape of point_grid should be [N, 1, 2]
        point_grid = np.hstack([xs, ys])[:, None, :]  
        point_coords = resize_transform.apply_coords(point_grid, original_size)
        point_labels = np.ones(point_coords.shape[: 2], dtype = point_coords.dtype)

        predicted_labels.append([])

        for image in images:
            # image: torch.Tensor [H, W, C]
            image = prepare_image(image, resize_transform, model.device)

            input = [{'image' : image, 'original_size' : original_size, 
                     'point_coords' : torch.tensor(point_coords, dtype=torch.float32 , device=model.device), 
                     'point_labels' : torch.tensor(point_labels, dtype=torch.float32, device=model.device)}]
        
            output = model(input, False)[0]
            predicted_labels[-1].append(predict_labels(output))
        
        predicted_labels[-1] = torch.stack(predicted_labels[-1]).cpu().numpy()
        gt_labels.append(labels)
    
    dice, mDice = calc_dice(predicted_labels, gt_labels)

    print(dice)
    print(mDice)


def calc_dice(predicted_labels, gt_labels):
    # list of [B, H, W, C]
    intersection = {}
    union = {}
    for organ_id in range(1, 14):
        for pred_label, gt_label in zip(predicted_labels, gt_labels):
            pred_mask, gt_mask = (pred_label == organ_id, gt_label == organ_id)

            intersection.setdefault(organ_id, 0)
            union.setdefault(organ_id, 0) 

            intersection[organ_id] += np.logical_and(pred_mask, gt_mask).sum()
            union[organ_id] += np.logical_or(pred_mask, gt_mask).sum()

    dice = {}
    mDice = []

    for k in intersection:
        dice[k] = intersection[k] / union[k]
        mDice.append(dice[k])
    
    return dice, np.mean(mDice)


@torch.no_grad()
def predict_labels(output, iou_thresh : float = 0.5):
    """
    Args:
    output: dictionary with keys 'masks', 'iou_predictions', 'low_res_logits', 'class_predictions'
                'masks': (torch.Tensor) Batched binary mask predictions,
                 with shape BxCxHxW, where B is the number of input prompts,
                 C is determined by multimask_output, and (H, W) is the
                 original size of the image.
                'iou_predictions': (torch.Tensor) The model's predictions
                 of mask quality, in shape BxC.
                'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
                'class_predictions': (torch.Tensor) BxCxn_classes, logits

    Returns:
    predicted labels: [H, W]
    """
    masks = output['masks'][:, 0, :, :]
    predicted_labels = torch.zeros(masks.shape[1:], dtype=torch.uint8).to(masks.device)
    iou_predictions = output['iou_predictions'].view(-1)
    class_predictions = output['class_predictions'][:, 0, :]
    organ_id_to_mask = {}


    permutation = torch.argsort(iou_predictions, descending=True)
    iou_predictions = iou_predictions[permutation]
    class_predictions = torch.argmax(class_predictions[permutation], dim=1)
    masks = masks[permutation]

    for iou, organ_pred, mask in zip(iou_predictions, class_predictions, masks):
        iou = iou.item()
        organ_pred = organ_pred.item()

        if organ_pred in organ_id_to_mask or organ_pred == 0:
            continue 

        # check if overlapped with other masks (IoU > threshold)
        overlapped = False
        for _, m in organ_id_to_mask.values():
            intersection = torch.sum(torch.logical_and(m, mask)).item()
            union = torch.sum(torch.logical_or(m, mask)).item()

            if intersection / union > iou_thresh:
                overlapped = True
                break 

        if overlapped:
            continue 

        mask = torch.logical_and(predicted_labels == 0, mask)
        organ_id_to_mask[organ_pred] = (iou, mask)
        predicted_labels[mask] = organ_pred
    
    return predicted_labels

if __name__ == '__main__':
    main()