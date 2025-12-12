import torch
import torch.nn as nn
from timm.models import register_model
from .cmt_module import CMTStem, Patch_Aggregate, CMTBlock


class CMT(nn.Module):
    def __init__(self,
                 in_channels=3,
                 stem_channel=32,
                 cmt_channel=[46, 92, 184, 368],
                 patch_channel=[46, 92, 184, 368],
                 block_layer=[2, 2, 10, 2],
                 R=3.6,
                 img_size=1024,
                 output_dim=256,
                 # Nuovi parametri per controllare il downsampling
                 use_aggressive_downsample=False
                 ):
        super(CMT, self).__init__()

        # Downsampling più aggressivo: riduci più velocemente la dimensione spaziale
        if use_aggressive_downsample:
            # Dopo stem: 1024 -> 128 (divide per 8 invece di 2)
            # Stage 1: 128 -> 64
            # Stage 2: 64 -> 32
            # Stage 3: 32 -> 32 (mantiene)
            # Stage 4: 32 -> 32 (mantiene)
            size = [img_size // 8, img_size // 16, img_size // 32, img_size // 32]
        else:
            # Downsampling originale
            size = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]

        self.img_size = img_size
        self.use_aggressive_downsample = use_aggressive_downsample


        self.stem = CMTStem(in_channels, stem_channel)

        # Patch Aggregation Layer
        self.patch1 = Patch_Aggregate(stem_channel, patch_channel[0])
        self.patch2 = Patch_Aggregate(patch_channel[0], patch_channel[1])
        self.patch3 = Patch_Aggregate(patch_channel[1], patch_channel[2])

        # Stage 4 non ha patch aggregation perché manteniamo la risoluzione
        if use_aggressive_downsample:
            self.patch4 = nn.Identity()  # Non fa downsampling
        else:
            self.patch4 = Patch_Aggregate(patch_channel[2], patch_channel[3])

        # CMT Block Layer - Calcola stride corretto basato sulla dimensione reale
        # stride determina la dimensione della local window per l'attention
        stride1 = max(1, size[0] // 16) if use_aggressive_downsample else 8
        stride2 = max(1, size[1] // 16) if use_aggressive_downsample else 4
        stride3 = max(1, size[2] // 16) if use_aggressive_downsample else 2
        stride4 = 1

        stage1 = [CMTBlock(img_size=size[0], stride=stride1,
                           d_k=cmt_channel[0], d_v=cmt_channel[0],
                           num_heads=1, R=R, in_channels=patch_channel[0]) for _ in range(block_layer[0])]
        self.stage1 = nn.Sequential(*stage1)

        stage2 = [CMTBlock(img_size=size[1], stride=stride2,
                           d_k=cmt_channel[1] // 2, d_v=cmt_channel[1] // 2,
                           num_heads=2, R=R, in_channels=patch_channel[1]) for _ in range(block_layer[1])]
        self.stage2 = nn.Sequential(*stage2)

        stage3 = [CMTBlock(img_size=size[2], stride=stride3,
                           d_k=cmt_channel[2] // 4, d_v=cmt_channel[2] // 4,
                           num_heads=4, R=R, in_channels=patch_channel[2]) for _ in range(block_layer[2])]
        self.stage3 = nn.Sequential(*stage3)

        # Stage 4 lavora sulla stessa risoluzione di stage 3 se aggressive downsample
        in_ch_stage4 = patch_channel[2] if use_aggressive_downsample else patch_channel[3]
        stage4 = [CMTBlock(img_size=size[3], stride=stride4,
                           d_k=cmt_channel[3] // 8, d_v=cmt_channel[3] // 8,
                           num_heads=8, R=R, in_channels=in_ch_stage4) for _ in range(block_layer[3])]
        self.stage4 = nn.Sequential(*stage4)

        # Convolution per ottenere l'output finale
        conv_in_channels = patch_channel[2] if use_aggressive_downsample else cmt_channel[3]
        self.conv = nn.Conv2d(conv_in_channels, 256, kernel_size=3, padding=1)

        # Upsample per ottenere 64x64
        self.upsample = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.stem(x)

        x = self.patch1(x)
        x = self.stage1(x)

        x = self.patch2(x)
        x = self.stage2(x)

        x = self.patch3(x)
        x = self.stage3(x)

        x = self.patch4(x)
        x = self.stage4(x)

        # Conv per ridurre i canali a 256
        x = self.conv(x)

        # Upsampling a 64x64
        x = self.upsample(x)

        return x  # [B, 256, 64, 64]

@register_model
def CMT_Ti(pretrained=False, img_size=1024, output_dim=256, **kwargs):
    model = CMT(
        in_channels=3,
        stem_channel=16,
        cmt_channel=[46, 92, 184, 368],
        patch_channel=[46, 92, 184, 368],
        block_layer=[2, 2, 10, 2],
        R=3.6,
        img_size=img_size,
        output_dim=output_dim,
        use_aggressive_downsample=True  # Attiva downsampling aggressivo
    )
    return model


"""@register_model
def CMT_Ti(pretrained=False, img_size=1024, output_dim=256, **kwargs):
    model = CMT(
        in_channels=3,
        stem_channel=16,
        cmt_channel=[46, 92, 184, 368],
        patch_channel=[46, 92, 184, 368],
        block_layer=[2, 2, 10, 2],
        R=3.6,
        img_size=img_size,
        output_dim=output_dim,
    )
    return model"""
@register_model
def CMT_Nano(pretrained=False, img_size=1024, output_dim=256, **kwargs):
    model = CMT(
        in_channels=3,
        stem_channel=12,  # Ancora più piccolo
        cmt_channel=[24, 48, 96, 192],  # Dimezzato rispetto a Ti
        patch_channel=[24, 48, 96, 192],
        block_layer=[1, 1, 4, 1],  # Meno blocchi nello stage 3
        R=3.6,
        img_size=img_size,
        output_dim=output_dim,
    )
    return model
def CMT_XS(img_size=224, num_class=1000):
    model = CMT(
        in_channels=3,
        stem_channel=16,
        cmt_channel=[52, 104, 208, 416],
        patch_channel=[52, 104, 208, 416],
        block_layer=[3, 3, 12, 3],
        R=3.8,
        img_size=img_size,
        num_class=num_class
    )
    return model

def CMT_S(img_size=224, num_class=1000):
    model = CMT(
        in_channels=3,
        stem_channel=32,
        cmt_channel=[64, 128, 256, 512],
        patch_channel=[64, 128, 256, 512],
        block_layer=[3, 3, 16, 3],
        R=4,
        img_size=img_size,
        num_class=num_class
    )
    return model

def CMT_B(img_size=224, num_class=1000):
    model = CMT(
        in_channels=3,
        stem_channel=38,
        cmt_channel=[76, 152, 304, 608],
        patch_channel=[76, 152, 304, 608],
        block_layer=[4, 4, 20, 4],
        R=4,
        img_size=img_size,
        num_class=num_class
    )
    return model

def test():
    calc_param = lambda net: sum(p.numel() for p in net.parameters() if p.requires_grad)
    img = torch.rand(2, 3, 224, 224)
    cmt_ti = CMT_Ti()
    cmt_xs = CMT_XS()
    cmt_x = CMT_S()
    cmt_b = CMT_B()
    logit = cmt_b(img)
    print(logit.size())
    print(f"CMT_Ti param: {calc_param(cmt_ti) / 1e6 : .2f} M")
    print(f"CMT_XS param: {calc_param(cmt_xs) / 1e6 : .2f} M")
    print(f"CMT_X  param: {calc_param(cmt_x) / 1e6 : .2f} M")
    print(f"CMT_B  param: {calc_param(cmt_b) / 1e6 : .2f} M")


