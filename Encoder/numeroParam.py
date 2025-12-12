import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import torch
from ptflops import get_model_complexity_info
from timm.models import checkpoint
from torchinfo import summary
from thop import profile

from modeling.build_sam import sam_model_registry


class SAMWrapper(torch.nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam_model = sam_model

    def forward(self, x):
        batched_input = [{
            "image": x[0],  # batch size 1
            "original_size": (1024, 1024),
            "multimask_output": False
        }]
        return self.sam_model(batched_input)[0]
def debug_model_parameters(model, top_n=20):
    """
    Trova dove sono i parametri nel modello.
    """
    print("=" * 80)
    print("🔍 DEBUG PARAMETRI MODELLO")
    print("=" * 80)

    # Conta totale
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 TOTALE:")
    print(f"  • Parametri totali:     {total_params:>15,} ({total_params / 1e6:.2f}M)")
    print(f"  • Parametri trainable:  {trainable_params:>15,} ({trainable_params / 1e6:.2f}M)")

    # Analizza ogni modulo
    print(f"\n📋 TOP {top_n} MODULI PER PARAMETRI:")
    print("-" * 80)
    print(f"{'Nome Modulo':<50} {'Parametri':>15} {'%':>8}")
    print("-" * 80)

    module_params = []
    for name, module in model.named_modules():
        # Conta solo i parametri diretti del modulo (non dei figli)
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            module_params.append({
                'name': name if name else 'root',
                'params': params,
                'percentage': (params / total_params) * 100
            })

    # Ordina per numero di parametri
    module_params.sort(key=lambda x: x['params'], reverse=True)

    for item in module_params[:top_n]:
        print(f"{item['name']:<50} {item['params']:>15,} {item['percentage']:>7.2f}%")

    # Analizza per tipo di layer
    print(f"\n📦 PARAMETRI PER TIPO DI LAYER:")
    print("-" * 80)

    layer_types = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Solo foglie
            layer_type = module.__class__.__name__
            params = sum(p.numel() for p in module.parameters())

            if layer_type not in layer_types:
                layer_types[layer_type] = {'count': 0, 'params': 0}
            layer_types[layer_type]['count'] += 1
            layer_types[layer_type]['params'] += params

    sorted_types = sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True)

    print(f"{'Tipo Layer':<30} {'Count':>8} {'Parametri':>15} {'%':>8}")
    print("-" * 80)
    for layer_type, info in sorted_types:
        percentage = (info['params'] / total_params) * 100
        print(f"{layer_type:<30} {info['count']:>8} {info['params']:>15,} {percentage:>7.2f}%")

    # Analizza per stage
    print(f"\n🎯 PARAMETRI PER STAGE:")
    print("-" * 80)

    stages = ['stem', 'patch1', 'stage1', 'patch2', 'stage2',
              'patch3', 'stage3', 'patch4', 'stage4', 'conv', 'upsample']

    for stage_name in stages:
        if hasattr(model, stage_name):
            stage = getattr(model, stage_name)
            stage_params = sum(p.numel() for p in stage.parameters())
            percentage = (stage_params / total_params) * 100
            print(f"{stage_name:<20} {stage_params:>15,} ({stage_params / 1e6:>6.2f}M) {percentage:>7.2f}%")

    print("\n" + "=" * 80)
    import torch
    from thop import profile
    # model: la tua istanza nn.Module

    input_image = torch.randn(1, 3, 1024, 1024).to(next(model.parameters()).device)

    batched_input = [{
        "image": input_image[0],  # rimuove dimensione batch extra se serve
        "original_size": (1024, 1024) #per edge sam
    }]

    wrapped_model = SAMWrapper(model).to("cpu")
    dummy_input = torch.randn(1, 3, 1024, 1024).to("cpu")
    # multimask_output = False per semplificare
    #flops = FlopCountAnalysis(model, batched_input)

    input_res = (3,1024,1024)
    macs, params = get_model_complexity_info(wrapped_model, input_res, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print(f"[ptflops] FLOPs: {macs}")
    print(f"[ptflops] Params: {params}")
    #flops, params = profile(wrapped_model, inputs=(dummy_input,))
    #print("FLOPs:", flops / 1e9, "GFLOPs")
    #print("Params:", params / 1e6, "M")
    #print("Total FLOPs:", flops.total())

    return total_params


def find_problematic_layers(model, threshold_mb=10):
    """
    Trova layer con troppi parametri (potenziali problemi).
    """
    print(f"\n⚠️  LAYER SOSPETTI (>{threshold_mb}MB):")
    print("-" * 80)

    suspicious = []
    for name, param in model.named_parameters():
        size_mb = param.numel() * 4 / (1024 ** 2)  # float32 = 4 bytes
        if size_mb > threshold_mb:
            suspicious.append({
                'name': name,
                'shape': list(param.shape),
                'params': param.numel(),
                'size_mb': size_mb
            })

    suspicious.sort(key=lambda x: x['size_mb'], reverse=True)

    if suspicious:
        print(f"{'Nome Parametro':<50} {'Shape':<25} {'Size (MB)':>12}")
        print("-" * 80)
        for item in suspicious[:10]:
            print(f"{item['name']:<50} {str(item['shape']):<25} {item['size_mb']:>11.2f}")
    else:
        print("✓ Nessun layer sospetto trovato")

    return suspicious


def compare_with_reference():
    """
    Confronta con i valori di riferimento per CMT-Ti.
    """
    print("\n📚 RIFERIMENTO CMT-Ti (Paper Originale):")
    print("-" * 80)

    reference = {
        'Total Params': '9.5M',
        'Input Size': '224x224',
        'FLOPs': '0.4G',
        'Channels': '[46, 92, 184, 368]',
        'Depths': '[2, 2, 10, 2]',
        'Expected Memory': '~38 MB (float32)'
    }

    for key, value in reference.items():
        print(f"  {key:<20}: {value}")

    print("\n💡 SE HAI 340M PARAMETRI, POSSIBILI CAUSE:")
    print("  1. ❌ Dimensioni embedding/hidden troppo grandi")
    print("  2. ❌ Attention con troppi head o dimensioni sbagliate")
    print("  3. ❌ FFN con expansion ratio troppo alto (R=3.6 corretto)")
    print("  4. ❌ Layer duplicati o non condivisi")
    print("  5. ❌ Bug nel calcolo di d_k, d_v negli attention blocks")
    print("  6. ❌ Immagine troppo grande (1024 vs 224) causa explosion nei position embeddings")


def check_attention_params(model):
    """
    Controlla i parametri degli attention layers.
    """
    print("\n🔎 ANALISI ATTENTION LAYERS:")
    print("-" * 80)

    attention_params = 0
    attention_count = 0

    for name, module in model.named_modules():
        # Cerca moduli che potrebbero essere attention
        if any(keyword in name.lower() for keyword in ['attn', 'attention', 'mhsa']):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                attention_params += params
                attention_count += 1
                print(f"  {name:<50} {params:>12,} ({params / 1e6:.2f}M)")

    if attention_count > 0:
        print(f"\n  TOTALE ATTENTION: {attention_params:,} ({attention_params / 1e6:.2f}M)")
        print(f"  Numero di attention layers: {attention_count}")


# ============= SCRIPT PRINCIPALE =============

if __name__ == "__main__":
    print("\n🔧 ISTRUZIONI:")
    print("=" * 80)
    print("1. Carica il tuo modello:")
    print("   from your_model import CMT_Ti")
    print("   model = CMT_Ti(img_size=1024, output_dim=256)")
    print()
    print("2. Esegui il debug:")
    print("   debug_model_parameters(model)")
    print("   find_problematic_layers(model)")
    print("   check_attention_params(model)")
    print()
    print("3. Manda l'output qui per analizzarlo!")
    print("=" * 80)

    print("\n\n💡 COSA CERCARE NELL'OUTPUT:")
    print("=" * 80)
    print("✓ Stage 3 dovrebbe avere più parametri (ha 10 blocks)")
    print("✓ Ogni CMTBlock NON dovrebbe superare 1-2M parametri")
    print("✓ Conv/Linear layers non dovrebbero essere troppo grandi")
    print("✓ Position embeddings per 1024x1024 possono essere grandi!")
    print()
    print("❌ PROBLEMI COMUNI:")
    print("  • Position embedding: 1024*1024 = 1M pixels → se embedded a 256D = 268M params!")
    print("  • Q,K,V projections con dimensioni sbagliate")
    print("  • FFN con hidden_dim = in_channels * R troppo grande")
    print("=" * 80)

    compare_with_reference()

    print("\n\n🚨 SOSPETTO PRINCIPALE:")
    print("=" * 80)
    print("Se usi img_size=1024, il problema potrebbe essere nei position embeddings!")
    print()
    print("Calcolo:")
    print("  • Patches 1024x1024 con patch_size=4 → 256x256 patches")
    print("  • Se ogni patch ha embedding 368D → 256*256*368 = 24M params")
    print("  • Ma se sbagli e usi pixel-level embedding → 1024*1024*368 = 386M params!")
    print()
    print("💡 SOLUZIONE: Controlla i position embeddings nel CMTBlock!")
    print("=" * 80)