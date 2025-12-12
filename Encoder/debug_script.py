import torch
import torch.nn as nn
from collections import OrderedDict
from thop import profile
from thop.vision.basic_hooks import count_convNd, count_linear
import numpy as np


class DetailedFLOPsCounter:
    """
    Conta FLOPs layer per layer con dettagli.
    """

    def __init__(self, model, input_size=(1, 3, 1024, 1024)):
        self.model = model
        self.input_size = input_size
        self.layer_flops = OrderedDict()
        self.layer_params = OrderedDict()
        self.layer_output_shapes = OrderedDict()

    def count_flops_per_layer(self):
        """
        Conta FLOPs per ogni layer usando hooks.
        """
        print("\n" + "=" * 100)
        print("⚡ ANALISI FLOPS DETTAGLIATA PER LAYER")
        print("=" * 100)

        device = next(self.model.parameters()).device
        input_tensor = torch.randn(self.input_size).to(device)

        # Registra hooks per ogni modulo
        hooks = []

        def hook_fn(module, input, output):
            """Hook per catturare FLOPs e shape."""
            module_name = None
            for name, mod in self.model.named_modules():
                if mod is module:
                    module_name = name
                    break

            if module_name is None:
                return

            # Calcola FLOPs per questo layer
            flops = 0
            params = sum(p.numel() for p in module.parameters(recurse=False))


            # Calcola FLOPs in base al tipo di layer
            if isinstance(module, nn.Conv2d):
                flops = self._count_conv2d(module, input[0], output)
            elif isinstance(module, nn.Linear):
                flops = self._count_linear(module, input[0], output)
            elif isinstance(module, nn.BatchNorm2d):
                flops = self._count_bn(module, input[0], output)
            elif isinstance(module, nn.LayerNorm):
                flops = self._count_ln(module, input[0], output)
            elif isinstance(module, nn.MultiheadAttention):
                flops = self._count_mha(module, input[0], output)

            # Salva statistiche
            if flops > 0 or params > 0:
                self.layer_flops[module_name] = flops
                self.layer_params[module_name] = params

                # Salva output shape
                if isinstance(output, torch.Tensor):
                    self.layer_output_shapes[module_name] = list(output.shape)
                elif isinstance(output, tuple) and len(output) > 0:
                    self.layer_output_shapes[module_name] = list(output[0].shape)

        # Registra hooks
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Solo foglie
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Rimuovi hooks
        for hook in hooks:
            hook.remove()

        return self.layer_flops, self.layer_params

    def _count_conv2d(self, module, input, output):
        """Conta FLOPs per Conv2d."""
        batch_size = output.shape[0]
        output_height, output_width = output.shape[2:]

        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        bias_ops = 1 if module.bias is not None else 0

        flops = batch_size * output_height * output_width * module.out_channels * (kernel_ops + bias_ops)
        return int(flops)

    def _count_linear(self, module, input, output):
        """Conta FLOPs per Linear."""
        if isinstance(input, tuple):
            input = input[0]

        batch_size = input.shape[0]
        flops = batch_size * module.in_features * module.out_features

        if module.bias is not None:
            flops += batch_size * module.out_features

        return int(flops)

    def _count_bn(self, module, input, output):
        """Conta FLOPs per BatchNorm."""
        if isinstance(input, tuple):
            input = input[0]

        # BN: mean, variance, normalize, scale, shift
        flops = input.numel() * 5
        return int(flops)

    def _count_ln(self, module, input, output):
        """Conta FLOPs per LayerNorm."""
        if isinstance(input, tuple):
            input = input[0]

        # LN: mean, variance, normalize, scale, shift
        flops = input.numel() * 5
        return int(flops)

    def _count_mha(self, module, input, output):
        """Stima FLOPs per MultiheadAttention."""
        if isinstance(input, tuple):
            input = input[0]


        batch_size, seq_len, embed_dim = input.shape
        num_heads = module.num_heads

        # Q, K, V projections
        qkv_flops = 3 * batch_size * seq_len * embed_dim * embed_dim

        # Attention scores: Q @ K^T
        attn_flops = batch_size * num_heads * seq_len * seq_len * (embed_dim // num_heads)

        # Attention @ V
        out_flops = batch_size * num_heads * seq_len * seq_len * (embed_dim // num_heads)

        # Output projection
        proj_flops = batch_size * seq_len * embed_dim * embed_dim

        return int(qkv_flops + attn_flops + out_flops + proj_flops)

    def print_detailed_report(self):
        """
        Stampa report dettagliato.
        """
        print("\n📊 REPORT DETTAGLIATO PER LAYER:")
        print("-" * 100)
        print(f"{'Layer Name':<50} {'FLOPs':>15} {'Params':>12} {'Output Shape':<20}")
        print("-" * 100)

        total_flops = sum(self.layer_flops.values())
        total_params = sum(self.layer_params.values())

        # Ordina per FLOPs
        sorted_layers = sorted(self.layer_flops.items(), key=lambda x: x[1], reverse=True)

        for name, flops in sorted_layers:
            params = self.layer_params.get(name, 0)
            output_shape = self.layer_output_shapes.get(name, [])

            flops_str = self._format_number(flops)
            params_str = self._format_number(params)
            shape_str = str(output_shape) if output_shape else "N/A"

            print(f"{name:<50} {flops_str:>15} {params_str:>12} {shape_str:<20}")

        print("-" * 100)
        print(f"{'TOTALE':<50} {self._format_number(total_flops):>15} {self._format_number(total_params):>12}")
        print(f"{'':50} {f'({total_flops / 1e9:.2f} GFLOPs)':>15} {f'({total_params / 1e6:.2f}M)':>12}")
        print("-" * 100)

    def analyze_by_stage(self):
        """
        Analizza FLOPs per stage (stem, stage1, stage2, etc.).
        """
        print("\n🎯 FLOPS PER STAGE:")
        print("-" * 80)

        stage_flops = {}
        stage_params = {}

        # Identifica stages
        for name, flops in self.layer_flops.items():
            # Estrai stage name (es: "stage1.0.conv" -> "stage1")
            stage = name.split('.')[0] if '.' in name else name

            if stage not in stage_flops:
                stage_flops[stage] = 0
                stage_params[stage] = 0

            stage_flops[stage] += flops
            stage_params[stage] += self.layer_params.get(name, 0)

        # Ordina per FLOPs
        sorted_stages = sorted(stage_flops.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Stage':<20} {'FLOPs':>20} {'Params':>15} {'% FLOPs':>10} {'% Params':>10}")
        print("-" * 80)

        total_flops = sum(stage_flops.values())
        total_params = sum(stage_params.values())

        for stage, flops in sorted_stages:
            params = stage_params[stage]
            flops_pct = (flops / total_flops) * 100 if total_flops > 0 else 0
            params_pct = (params / total_params) * 100 if total_params > 0 else 0

            print(f"{stage:<20} {self._format_number(flops):>20} {self._format_number(params):>15} "
                  f"{flops_pct:>9.1f}% {params_pct:>9.1f}%")

        print("-" * 80)
        print(f"{'TOTALE':<20} {self._format_number(total_flops):>20} {self._format_number(total_params):>15} "
              f"{'100.0%':>10} {'100.0%':>10}")
        print("-" * 80)

    def analyze_by_operation_type(self):
        """
        Analizza FLOPs per tipo di operazione.
        """
        print("\n🔧 FLOPS PER TIPO DI OPERAZIONE:")
        print("-" * 80)

        op_flops = {}
        op_counts = {}

        for name, module in self.model.named_modules():
            if name in self.layer_flops:
                op_type = module.__class__.__name__

                if op_type not in op_flops:
                    op_flops[op_type] = 0
                    op_counts[op_type] = 0

                op_flops[op_type] += self.layer_flops[name]
                op_counts[op_type] += 1

        # Ordina per FLOPs
        sorted_ops = sorted(op_flops.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Operation Type':<25} {'Count':>8} {'Total FLOPs':>20} {'% Total':>10} {'Avg FLOPs/Op':>20}")
        print("-" * 80)

        total = sum(op_flops.values())

        for op_type, flops in sorted_ops:
            count = op_counts[op_type]
            pct = (flops / total) * 100 if total > 0 else 0
            avg = flops / count if count > 0 else 0

            print(f"{op_type:<25} {count:>8} {self._format_number(flops):>20} "
                  f"{pct:>9.1f}% {self._format_number(avg):>20}")

        print("-" * 80)
        print(f"{'TOTALE':<25} {sum(op_counts.values()):>8} {self._format_number(total):>20} {'100.0%':>10}")
        print("-" * 80)

    def generate_visualization(self):
        """
        Genera grafici di visualizzazione.
        """
        try:
            import matplotlib.pyplot as plt

            # Plot 1: FLOPs per stage
            stage_flops = {}
            for name, flops in self.layer_flops.items():
                stage = name.split('.')[0] if '.' in name else name
                stage_flops[stage] = stage_flops.get(stage, 0) + flops

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Bar plot FLOPs
            stages = list(stage_flops.keys())
            flops = [stage_flops[s] / 1e9 for s in stages]  # Convert to GFLOPs

            ax1.bar(stages, flops, color='#3498db', edgecolor='black')
            ax1.set_xlabel('Stage', fontsize=12, fontweight='bold')
            ax1.set_ylabel('FLOPs (G)', fontsize=12, fontweight='bold')
            ax1.set_title('FLOPs per Stage', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)

            # Pie chart per tipo operazione
            op_flops = {}
            for name, module in self.model.named_modules():
                if name in self.layer_flops:
                    op_type = module.__class__.__name__
                    op_flops[op_type] = op_flops.get(op_type, 0) + self.layer_flops[name]

            # Top 5 + Others
            sorted_ops = sorted(op_flops.items(), key=lambda x: x[1], reverse=True)
            top_ops = sorted_ops[:5]
            others = sum(f for _, f in sorted_ops[5:])

            labels = [op for op, _ in top_ops]
            values = [f for _, f in top_ops]

            if others > 0:
                labels.append('Others')
                values.append(others)

            ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('FLOPs Distribution by Operation Type', fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.savefig('flops_analysis.png', dpi=300, bbox_inches='tight')
            print("\n📊 Grafico salvato: flops_analysis.png")
            plt.close()

        except ImportError:
            print("\n⚠️  matplotlib non disponibile, skip visualizzazione")

    def _format_number(self, num):
        """Formatta numeri grandi."""
        if num >= 1e9:
            return f"{num / 1e9:.2f}G"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        else:
            return str(num)

    def run_complete_analysis(self):
        """
        Esegue analisi completa.
        """
        print("\n" + "=" * 100)
        print("🚀 ANALISI FLOPS COMPLETA")
        print("=" * 100)

        # 1. Conta FLOPs per layer
        self.count_flops_per_layer()

        # 2. Report dettagliato
        self.print_detailed_report()

        # 3. Analisi per stage
        self.analyze_by_stage()

        # 4. Analisi per tipo operazione
        self.analyze_by_operation_type()

        # 5. Visualizzazione
        self.generate_visualization()

        # 6. Summary
        total_flops = sum(self.layer_flops.values())
        total_params = sum(self.layer_params.values())

        print("\n" + "=" * 100)
        print("📈 SUMMARY")
        print("=" * 100)
        print(f"Total FLOPs:      {self._format_number(total_flops):>15} ({total_flops / 1e9:.3f} GFLOPs)")
        print(f"Total Parameters: {self._format_number(total_params):>15} ({total_params / 1e6:.3f} M)")
        print(f"Input Size:       {self.input_size}")
        print(f"FLOPs/Param:      {total_flops / total_params:.2f}")
        print("=" * 100)

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

# ============= SCRIPT DI UTILIZZO =============

def analyze_model_flops(model, input_size=(1, 3, 1024, 1024)):
    """
    Funzione helper per analizzare un modello.
    """

    counter = DetailedFLOPsCounter(SAMWrapper(model), input_size)
    counter.run_complete_analysis()
    return counter


if __name__ == "__main__":
    print("""
🔧 COME USARE:

from your_model import CMT_Ti
from flops_counter import analyze_model_flops

# Carica modello
model = CMT_Ti(img_size=1024, output_dim=256)
model.eval()

# Analizza FLOPs
counter = analyze_model_flops(model, input_size=(1, 3, 1024, 1024))

# Output:
# - Report dettagliato per layer
# - Analisi per stage
# - Analisi per tipo operazione
# - Grafici di visualizzazione

    """)

    print("\n" + "=" * 100)
    print("💡 COSA ASPETTARSI:")
    print("=" * 100)
    print("CMT-Ti (224x224):    ~0.4 GFLOPs")
    print("CMT-Ti (1024x1024):  ~8-10 GFLOPs (stima)")
    print()
    print("Stage con più FLOPs: stage3 (ha 10 blocks)")
    print("Operazioni costose: Conv2d, Attention")
    print("=" * 100)