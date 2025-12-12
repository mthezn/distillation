
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from repvit_sam.modeling.tiny_vit_sam import TinyViT
from repvit_sam.modeling.image_encoder import ImageEncoderViT
from repvit_sam.modeling.mask_decoder import MaskDecoder
from repvit_sam.modeling.prompt_encoder import PromptEncoder
from DecoderAutoSam.MaskDecoderAuto import MaskDecoderAuto
from DecoderAutoSam.UnetDecoder import UnetDecoder



class AutoSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"


    def __init__(
        self,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        prompt_encoder: PromptEncoder,

        mask_decoder:Union[ MaskDecoderAuto,UnetDecoder],
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        AutoSAM predicts object masks from an image without any input prompt.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.

            prompt_encoder (PromptEncoder): Encodes dimension of the image as prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            .
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder

        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward( #QUESTO FORWARD NON VA BENE PER ESSERE IMPIEGATO CON AUTOSAMUNET, IL MODELLO FUNZIONA SOLO SE LO CHIAMI SEPRATAMENTE NECODER E DECODER. bIOSGNA MODIFCARE I NMOMI O DIFFERENZIARE IN BASE AL DECODER
        self,
        batched_input: List[Dict[str, Any]],
       multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.


        Arguments:
          batched_input (list(dict)): A list of dictionaries containing
            input images and prompts.
          multimask_output (bool): Whether to output multiple masks per
            prompt. If False, only the best mask is returned.
        """
        """
        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        # Encode the images
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input],dim = 0)
        image_embeddings= self.image_encoder(input_images)
       # print(type(image_embeddings),getattr(image_embeddings, 'shape', None))

        outputs = []
        # Decode the masks
        for image_record,curr_embeddings in zip(batched_input, image_embeddings):
            low_res = self.mask_decoder(
                x=image_embeddings,


                #multimask_output=multimask_output,
            )

            masks = self.postprocess_masks(
                low_res,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                "masks": masks,
                #"iou_predictions": iou_pred,
                "low_res_logits": low_res,
                }
        )

        return outputs
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x