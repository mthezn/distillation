import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Encoder.CMT import CMT_Ti
from repvit_sam.modeling import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer, ImageEncoderViT, TinyViT, RepViT
from modeling.autoSam import AutoSam
from DecoderAutoSam.MaskDecoderAuto import MaskDecoderAuto
import torch
#from EdgeSAM.edge_sam.modeling import RepViT
from DecoderAutoSam.UnetDecoder import UnetDecoder
from timm.models import create_model
from functools import partial


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    """
    Function: _build_sam

    Purpose:
        Internal function to assemble a SAM model with a customizable ViT encoder
        (depth, embedding dimension, attention heads) and standard SAM decoding components.

    Inputs:
        encoder_embed_dim (int): Embedding dimension of the ViT encoder.
        encoder_depth (int): Number of transformer layers.
        encoder_num_heads (int): Number of attention heads.
        encoder_global_attn_indexes (List[int]): Indices for applying global attention.
        checkpoint (str, optional): Optional model checkpoint path.

    Returns:
        sam (Sam): Configured SAM model in eval() mode.
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam



def build_sam_CMT(checkpoint=None):
    """
    Function: build_sam_CMT

    Purpose:
        Constructs and returns a SAM (Segment Anything Model) instance with a
        lightweight CMT-Ti image encoder. Optionally loads pre-trained weights
        from a checkpoint file if provided.

    Inputs:
        checkpoint (str, optional):
            Path to a .pth file containing the model's state_dict (pre-trained weights).
            If None, the model is returned with randomly initialized weights.

    Returns:
        cmt_sam (Sam):
            A configured SAM model with the following components:
                - image_encoder: CMT-Ti for 1024×1024 input resolution
                - prompt_encoder: encodes spatial prompts and optional masks (256-dim)
                - mask_decoder: includes a bidirectional transformer and IoU prediction head
                - pixel normalization using ImageNet mean and std (RGB)

    Details:
        - The image encoder processes normalized RGB images of size (1024, 1024).
        - The prompt encoder handles point/box inputs and optional binary masks.
        - The mask decoder outputs 3 segmentation masks and IoU confidence scores.
        - The model is set to eval() mode by default (suitable for inference).

    """

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    cmt_sam = Sam(
            image_encoder=CMT_Ti(img_size=1024),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    cmt_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        cmt_sam.load_state_dict(state_dict)
    return cmt_sam


def build_sam_encoder_decoder(checkpoint=None):
    """
    Function: build_sam_encoder_decoder

    Purpose:
        Builds and returns a SAM-like model using an encoder-decoder architecture
        based on the lightweight CMT-Ti image encoder and a custom AutoSam wrapper.
        Designed for efficient segmentation with transformer-based mask decoding.

    Inputs:
        checkpoint (str, optional):
            Path to a .pth file containing the model’s state_dict.
            If None, model is initialized with random weights.

    Returns:
        enc_dec (AutoSam):
            A complete SAM encoder-decoder model:
                - Encoder: CMT-Ti
                - Prompt encoder: 256-dim embedding, spatial-aware
                - Decoder: MaskDecoderAuto with a TwoWayTransformer
            Model is returned in eval() mode, ready for inference.
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    enc_dec = AutoSam(
        image_encoder=CMT_Ti(img_size=1024),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderAuto(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    enc_dec.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        enc_dec.load_state_dict(state_dict)
    return enc_dec


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam_vit_t(checkpoint=None):
    """
    Function: build_sam_vit_t

    Purpose:
        Builds a lightweight SAM variant using TinyViT as the image encoder,
        targeting mobile or edge deployment with reduced model size.

    Inputs:
        checkpoint (str, optional)

    Returns:
        mobile_sam (Sam):
            A TinyViT-powered SAM model with standard prompt encoder and
            mask decoder. Suitable for real-time or low-resource settings.
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        mobile_sam.load_state_dict(state_dict)
    return mobile_sam



def build_sam_repvit(checkpoint=None):
    """
    Function: build_sam_repvit

    Purpose:
        Constructs a SAM model using RepViT as the image encoder. RepViT is a
        re-parameterized Vision Transformer optimized for inference speed.

    Inputs:
        checkpoint (str, optional)

    Returns:
        repvit_sam (Sam):
            SAM model with RepViT encoder and standard prompt + mask decoders.
            Optimized for efficient image segmentation.
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    repvit_sam = Sam(
            image_encoder=create_model('repvit'),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    repvit_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        repvit_sam.load_state_dict(state_dict)
    return repvit_sam

def build_edge_sam(checkpoint=None, upsample_mode="bicubic"):
    image_encoder = RepViT(
        arch="m1",
        img_size=(1024,1024),
        upsample_mode=upsample_mode,
        fuse=True
    )
    return _build_sam(image_encoder, checkpoint)
def build_sam_unet(checkpoint=None):
    """
    Function: build_sam_unet

    Purpose:
        Constructs a SAM-like model with a U-Net-style mask decoder instead of
        the standard transformer decoder. Uses CMT-Ti as the image encoder.

    Inputs:
        checkpoint (str, optional)

    Returns:
        enc_dec (AutoSam):
            An encoder-decoder SAM variant with U-Net decoder for binary segmentation.
            Suitable for tasks requiring simpler mask structures or medical images.
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    enc_dec = AutoSam(
        image_encoder=CMT_Ti(img_size=1024),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=UnetDecoder(
            encoder_channels=256,
            out_channels=1,



        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    enc_dec.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        enc_dec.load_state_dict(state_dict)
    return enc_dec

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
    "repvit": build_sam_repvit,
    "CMT": build_sam_CMT,
    "autoSam": build_sam_encoder_decoder,
    "autoSamUnet": build_sam_unet,
    "edgeSam": build_edge_sam,
}
