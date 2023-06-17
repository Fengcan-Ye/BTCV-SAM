import torch
from torch import nn
from torch.nn import functional as F
from segment_anything.modeling.mask_decoder import MaskDecoder, MLP

from typing import List, Tuple

class Decoder(nn.Module):
    def __init__(
        self,
        mask_decoder : MaskDecoder,
        cls_head_depth: int = 3, 
        cls_head_hidden_dim: int = 256,
        n_classes: int = 14,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
          cls_head_depth (int): the depth of the MLP used to predict
            mask class
          cls_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask class
          n_classes (int): number of classes
        """
        super(Decoder, self).__init__()

        self.transformer_dim = mask_decoder.transformer_dim
        self.transformer = mask_decoder.transformer 

        self.num_multimask_outputs = mask_decoder.num_multimask_outputs

        self.iou_token = mask_decoder.iou_token
        self.num_mask_tokens = mask_decoder.num_multimask_outputs
        self.mask_tokens = mask_decoder.mask_tokens

        self.output_upscaling = mask_decoder.output_upscaling
        self.output_hypernetworks_mlps = mask_decoder.output_hypernetworks_mlps

        self.iou_prediction_head = mask_decoder.iou_prediction_head

        self.n_classes = n_classes
        self.cls_token = nn.Embedding(1, self.transformer_dim)
        self.cls_prediction_head = MLP(
            self.transformer_dim, cls_head_hidden_dim, self.num_mask_tokens * n_classes, cls_head_depth
        )

    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        masks, iou_pred, cls_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        cls_pred = cls_pred[:, mask_slice, :]

        # Prepare output
        return masks, iou_pred, cls_pred
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.cls_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        cls_token_out = hs[:, 0, :]
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : (2 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        cls_pred = self.cls_prediction_head(cls_token_out).view(b, -1, self.n_classes)

        return masks, iou_pred, cls_pred



