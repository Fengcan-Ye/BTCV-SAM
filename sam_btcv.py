import torch.nn as nn
from segment_anything.modeling.sam import Sam

class SamBTCV(Sam):
    def __init__(self, sam : Sam):
        super(SamBTCV, self).__init__(sam.image_encoder, sam.prompt_encoder, sam.mask_decoder, 
                                      sam.pixel_mean, sam.pixel_std)
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False 
        
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, batched_input, multimask_output):
        raise NotImplementedError