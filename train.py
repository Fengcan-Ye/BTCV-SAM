import argparse 
from segment_anything import sam_model_registry
from dataset import *
from training import *
from sam_btcv import SamBTCV 
from torch.optim import AdamW
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='Fine-tuning SAM on BTCV')

parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--best_model_root', type=str, default='./')
parser.add_argument('--lr_step_size', type=int, default=5)
parser.add_argument('--lr_decay_frac', type=float, default=0.7)
parser.add_argument('--classification', action='store_true')

args = parser.parse_args()

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')    
    else:
        device = torch.device('cpu')

    # Register SAM
    sam = sam_model_registry['default']('./sam_vit_h_4b8939.pth')
    sam.to(device)

    # Prepare dataset
    preprocess = lambda images, labels: to_tensor(*to_uint8_rgb(*remove_pure_background(images, labels)))
    train_set = BTCV2DSliceDataset(root_dir='./data', 
                                   json_file='./data/dataset_0.json', 
                                   type='training', 
                                   preprocess=preprocess)
    validation_set = BTCV2DSliceDataset(root_dir='./data', 
                                        json_file='./data/dataset_0.json', 
                                        type='validation', 
                                        preprocess=preprocess)
    
    # Fine-tune SAM on BTCV Dataset
    model = SamBTCV(sam, requires_classification=args.classification)
    optimizer = AdamW(model.mask_decoder.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_decay_frac)
    train_model(train_set, validation_set, optimizer, exp_lr_scheduler, model, best_model_root=args.best_model_root, 
                batch_size=args.batch_size, n_epochs=args.n_epochs)

if __name__ == '__main__':
    main()