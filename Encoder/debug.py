#from Encoder import CMT_Ti
from debug_script import analyze_model_flops
from modeling.build_sam import sam_model_registry
from numeroParam import debug_model_parameters
#model = CMT_Ti(img_size=1024, output_dim=256)
from ptflops import get_model_complexity_info
from debug_script import SAMWrapper

#model = sam_model_registry["autoSamUnet"](checkpoint=None)
model = sam_model_registry["vit_t"](checkpoint=None)
# 1. Conta totale
total = sum(p.numel() for p in model.parameters())
print(f"Totale: {total:,} ({total/1e6:.2f}M)")

# 2. Trova i layer più grandi
for name, param in model.named_parameters():
    #if param.numel() > 1_000_000:  # > 10M
        print(f" {name}: {param.shape} = {param.numel():,} ({param.numel()/1e6:.2f}M)")
wrap = SAMWrapper(model)
# 3. Usa il mio script

flops, params = get_model_complexity_info(

        wrap, (3, 1024, 1024), as_strings=True, print_per_layer_stat=True
    )
print(flops)
analyze_model_flops(model)