import torch
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from collections import defaultdict
import json

import torch
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from collections import defaultdict
import json


class YOLOStyleReporter:
    """
    Genera report in stile YOLO per classificazione multi-classe.
    Gestisce correttamente False Positives e False Negatives.
    """

    def __init__(self, class_names, save_dir='resultsMultiClass'):
        """
        Args:
            class_names: Dict {class_id: 'class_name'}
            save_dir: Directory per salvare i report
        """
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Statistiche per classe
        self.stats = {
            'predictions': [],
            'ground_truths': [],
            'confidences': [],
            'processing_times': [],
            'false_positives': [],  # NUOVO: Lista di FP
            'false_negatives': [],  # NUOVO: Lista di FN
            'per_class': defaultdict(lambda: {
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'predictions': [],
                'ground_truths': [],
                'confidences': []
            })
        }

    def update(self, predictions, ground_truths, confidences=None, processing_time=None):
        """
        Aggiorna statistiche con nuove predizioni.
        Gestisce anche FP (pred ma no GT) e FN (GT ma no pred).

        Args:
            predictions: Lista di class_id predetti (può includere -1 per FP)
            ground_truths: Lista di class_id GT (può includere -1 per FN)
            confidences: Confidence scores (opzionale)
            processing_time: Tempo di processing
        """
        # GESTIONE FALSE POSITIVES
        # Se pred != -1 ma gt == -1 → False Positive
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            if pred != -1 and gt == -1:
                self.stats['false_positives'].append({
                    'predicted_class': pred,
                    'confidence': confidences[i] if confidences else None
                })

        # GESTIONE FALSE NEGATIVES
        # Se gt != -1 ma non c'è corrispondente pred → False Negative
        # (questo caso viene gestito quando chiami update con liste non allineate)

        # Filtra -1 prima di aggiungere alle statistiche generali
        valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths)
                       if p != -1 and g != -1]

        if len(valid_pairs) > 0:
            valid_preds, valid_gts = zip(*valid_pairs)

            self.stats['predictions'].extend(valid_preds)
            self.stats['ground_truths'].extend(valid_gts)

            if confidences is not None:
                valid_confs = [c for i, c in enumerate(confidences)
                               if predictions[i] != -1 and ground_truths[i] != -1]
                self.stats['confidences'].extend(valid_confs)

            # Aggiorna statistiche per classe
            for pred, gt in zip(valid_preds, valid_gts):
                for class_id in self.class_names.keys():
                    if pred == class_id and gt == class_id:
                        self.stats['per_class'][class_id]['tp'] += 1
                    elif pred == class_id and gt != class_id:
                        self.stats['per_class'][class_id]['fp'] += 1
                    elif pred != class_id and gt == class_id:
                        self.stats['per_class'][class_id]['fn'] += 1
                    else:
                        self.stats['per_class'][class_id]['tn'] += 1

                self.stats['per_class'][pred]['predictions'].append(pred)
                self.stats['per_class'][gt]['ground_truths'].append(gt)

        # Conta FP per classe
        for fp in [p for p, g in zip(predictions, ground_truths) if p != -1 and g == -1]:
            self.stats['per_class'][fp]['fp'] += 1

        # Conta FN per classe
        for fn in [g for p, g in zip(predictions, ground_truths) if p == -1 and g != -1]:
            self.stats['per_class'][fn]['fn'] += 1
            self.stats['false_negatives'].append({'missed_class': fn})

        if processing_time is not None:
            self.stats['processing_times'].append(processing_time)

    def compute_metrics(self):
        """
        Calcola metriche complete includendo FP e FN.
        """
        if len(self.stats['predictions']) == 0:
            # Nessuna predizione valida
            metrics = {
                'overall': {
                    'accuracy': 0.0,
                    'total_samples': 0,
                    'false_positives': len(self.stats['false_positives']),
                    'false_negatives': len(self.stats['false_negatives']),
                    'avg_time_ms': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
                },
                'per_class': {}
            }

            # Metriche per classe anche senza predizioni valide
            for class_id, class_name in self.class_names.items():
                tp = self.stats['per_class'][class_id]['tp']
                fp = self.stats['per_class'][class_id]['fp']
                fn = self.stats['per_class'][class_id]['fn']
                tn = self.stats['per_class'][class_id]['tn']

                metrics['per_class'][class_id] = {
                    'name': class_name,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'support': fn,  # Solo FN se nessuna pred
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'specificity': 0.0
                }

            return metrics

        preds = np.array(self.stats['predictions'])
        gts = np.array(self.stats['ground_truths'])

        # Metriche globali
        overall_accuracy = np.mean(preds == gts)

        # Metriche per classe
        precision, recall, f1, support = precision_recall_fscore_support(
            gts, preds, labels=list(self.class_names.keys()), zero_division=0
        )

        # Costruisci dizionario metriche
        metrics = {
            'overall': {
                'accuracy': overall_accuracy,
                'total_samples': len(preds),
                'false_positives': len(self.stats['false_positives']),
                'false_negatives': len(self.stats['false_negatives']),
                'avg_time_ms': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
            },
            'per_class': {}
        }

        for idx, class_id in enumerate(self.class_names.keys()):
            tp = self.stats['per_class'][class_id]['tp']
            fp = self.stats['per_class'][class_id]['fp']
            fn = self.stats['per_class'][class_id]['fn']
            tn = self.stats['per_class'][class_id]['tn']

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            metrics['per_class'][class_id] = {
                'name': self.class_names[class_id],
                'accuracy': accuracy,
                'precision': precision[idx],
                'recall': recall[idx],
                'f1': f1[idx],
                'support': support[idx],
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }

        return metrics

    def generate_fp_fn_analysis(self):
        """
        NUOVO: Genera report dettagliato su False Positives e False Negatives.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. False Positives per classe
        ax1 = axes[0, 0]
        fp_by_class = defaultdict(int)
        for fp in self.stats['false_positives']:
            fp_by_class[fp['predicted_class']] += 1

        if fp_by_class:
            classes = [self.class_names[c] for c in sorted(fp_by_class.keys())]
            counts = [fp_by_class[c] for c in sorted(fp_by_class.keys())]

            ax1.bar(classes, counts, color='#e74c3c', edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Class', fontweight='bold')
            ax1.set_ylabel('Count', fontweight='bold')
            ax1.set_title(f'False Positives per Class (Total: {len(self.stats["false_positives"])})',
                          fontweight='bold', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)

            for i, (cls, cnt) in enumerate(zip(classes, counts)):
                ax1.text(i, cnt + 0.5, str(cnt), ha='center', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No False Positives', ha='center', va='center',
                     fontsize=14, transform=ax1.transAxes)
            ax1.axis('off')

        # 2. False Negatives per classe
        ax2 = axes[0, 1]
        fn_by_class = defaultdict(int)
        for fn in self.stats['false_negatives']:
            fn_by_class[fn['missed_class']] += 1

        if fn_by_class:
            classes = [self.class_names[c] for c in sorted(fn_by_class.keys())]
            counts = [fn_by_class[c] for c in sorted(fn_by_class.keys())]

            ax2.bar(classes, counts, color='#f39c12', edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Class', fontweight='bold')
            ax2.set_ylabel('Count', fontweight='bold')
            ax2.set_title(f'False Negatives per Class (Total: {len(self.stats["false_negatives"])})',
                          fontweight='bold', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)

            for i, (cls, cnt) in enumerate(zip(classes, counts)):
                ax2.text(i, cnt + 0.5, str(cnt), ha='center', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No False Negatives', ha='center', va='center',
                     fontsize=14, transform=ax2.transAxes)
            ax2.axis('off')

        # 3. TP vs FP vs FN per classe
        ax3 = axes[1, 0]
        metrics = self.compute_metrics()

        classes = []
        tps = []
        fps = []
        fns = []

        for class_id in sorted(self.class_names.keys()):
            cm = metrics['per_class'][class_id]
            classes.append(cm['name'])
            tps.append(cm['tp'])
            fps.append(cm['fp'])
            fns.append(cm['fn'])

        x = np.arange(len(classes))
        width = 0.25

        ax3.bar(x - width, tps, width, label='True Positives', color='#2ecc71', edgecolor='black')
        ax3.bar(x, fps, width, label='False Positives', color='#e74c3c', edgecolor='black')
        ax3.bar(x + width, fns, width, label='False Negatives', color='#f39c12', edgecolor='black')

        ax3.set_xlabel('Class', fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('TP vs FP vs FN per Class', fontweight='bold', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. Error Rate per classe
        ax4 = axes[1, 1]
        error_rates = []

        for class_id in sorted(self.class_names.keys()):
            cm = metrics['per_class'][class_id]
            total = cm['tp'] + cm['fp'] + cm['fn']
            error_rate = (cm['fp'] + cm['fn']) / total if total > 0 else 0
            error_rates.append(error_rate * 100)

        colors = ['#2ecc71' if er < 10 else '#f39c12' if er < 30 else '#e74c3c'
                  for er in error_rates]

        ax4.barh(classes, error_rates, color=colors, edgecolor='black')
        ax4.set_xlabel('Error Rate (%)', fontweight='bold')
        ax4.set_title('Error Rate per Class (FP + FN)', fontweight='bold', fontsize=12)
        ax4.set_xlim([0, 100])
        ax4.grid(axis='x', alpha=0.3)

        for i, (cls, er) in enumerate(zip(classes, error_rates)):
            ax4.text(er + 1, i, f'{er:.1f}%', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'fp_fn_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Salva anche CSV dettagliato
        fp_fn_df = pd.DataFrame({
            'Class': classes,
            'True_Positives': tps,
            'False_Positives': fps,
            'False_Negatives': fns,
            'Error_Rate_%': error_rates
        })
        fp_fn_df.to_csv(os.path.join(self.save_dir, 'fp_fn_details.csv'), index=False)

        print(f"  ✓ FP/FN Analysis salvata")
        print(f"    Total FP: {len(self.stats['false_positives'])}")
        print(f"    Total FN: {len(self.stats['false_negatives'])}")

    def generate_results_table(self):
        """
        Genera tabella risultati AGGIORNATA con FP/FN.
        """
        metrics = self.compute_metrics()

        # Crea DataFrame
        data = []
        for class_id in sorted(metrics['per_class'].keys()):
            cm = metrics['per_class'][class_id]
            data.append({
                'Class': cm['name'],
                'Support': cm['support'],
                'TP': cm['tp'],
                'FP': cm['fp'],
                'FN': cm['fn'],
                'Accuracy': f"{cm['accuracy']:.3f}",
                'Precision': f"{cm['precision']:.3f}",
                'Recall': f"{cm['recall']:.3f}",
                'F1-Score': f"{cm['f1']:.3f}"
            })

        df = pd.DataFrame(data)

        # Aggiungi riga totale
        total_tp = sum([metrics['per_class'][c]['tp'] for c in metrics['per_class']])
        total_fp = sum([metrics['per_class'][c]['fp'] for c in metrics['per_class']])
        total_fn = sum([metrics['per_class'][c]['fn'] for c in metrics['per_class']])

        total_row = {
            'Class': 'all',
            'Support': len(self.stats['predictions']),
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Accuracy': f"{np.mean([metrics['per_class'][c]['accuracy'] for c in metrics['per_class']]):.3f}",
            'Precision': f"{np.mean([metrics['per_class'][c]['precision'] for c in metrics['per_class']]):.3f}",
            'Recall': f"{np.mean([metrics['per_class'][c]['recall'] for c in metrics['per_class']]):.3f}",
            'F1-Score': f"{np.mean([metrics['per_class'][c]['f1'] for c in metrics['per_class']]):.3f}"
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        # Salva CSV
        df.to_csv(os.path.join(self.save_dir, 'results.csv'), index=False)

        # Crea immagine tabella
        fig, ax = plt.subplots(figsize=(18, len(df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(df.columns)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Stile header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Evidenzia ultima riga (totale)
        for i in range(len(df.columns)):
            table[(len(df), i)].set_facecolor('#e8f8f5')
            table[(len(df), i)].set_text_props(weight='bold')

        plt.savefig(os.path.join(self.save_dir, 'results_table.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return df

    def generate_confusion_matrix(self):
        """Genera confusion matrix."""
        if len(self.stats['predictions']) == 0:
            print("  ⚠️  Nessuna predizione valida per confusion matrix")
            return None

        preds = self.stats['predictions']
        gts = self.stats['ground_truths']

        labels = sorted(self.class_names.keys())
        label_names = [self.class_names[i] for i in labels]

        cm = confusion_matrix(gts, preds, labels=labels)

        # Plot confusion matrix normalizzata
        fig, ax = plt.subplots(figsize=(12, 10))

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            cbar_kws={'label': 'Normalized Count'},
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return cm

    def generate_pr_curves(self):
        """Genera grafici metriche per classe."""
        metrics = self.compute_metrics()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        classes = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for class_id, class_metrics in metrics['per_class'].items():
            classes.append(class_metrics['name'])
            accuracies.append(class_metrics['accuracy'])
            precisions.append(class_metrics['precision'])
            recalls.append(class_metrics['recall'])
            f1s.append(class_metrics['f1'])

        x = np.arange(len(classes))
        width = 0.2

        ax1.bar(x - 1.5 * width, accuracies, width, label='Accuracy', color='#9b59b6')
        ax1.bar(x - 0.5 * width, precisions, width, label='Precision', color='#2ecc71')
        ax1.bar(x + 0.5 * width, recalls, width, label='Recall', color='#3498db')
        ax1.bar(x + 1.5 * width, f1s, width, label='F1-Score', color='#e74c3c')

        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Metrics per Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])

        supports = [metrics['per_class'][cid]['support'] for cid in sorted(metrics['per_class'].keys())]

        ax2.bar(classes, supports, color='#f39c12')
        ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
        ax2.set_title('Support per Class', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_per_class.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_plot(self):
        """Genera summary plot."""
        metrics = self.compute_metrics()

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Overall accuracy
        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(
            0.5, 0.5,
            f"Overall Accuracy: {metrics['overall']['accuracy'] * 100:.2f}%\n"
            f"FP: {metrics['overall']['false_positives']} | FN: {metrics['overall']['false_negatives']}",
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
        )
        ax1.axis('off')

        # Metriche medie
        ax2 = fig.add_subplot(gs[1, 0])
        avg_metrics = {
            'Accuracy': np.mean([m['accuracy'] for m in metrics['per_class'].values()]),
            'Precision': np.mean([m['precision'] for m in metrics['per_class'].values()]),
            'Recall': np.mean([m['recall'] for m in metrics['per_class'].values()]),
            'F1': np.mean([m['f1'] for m in metrics['per_class'].values()])
        }
        colors_bar = ['#9b59b6', '#2ecc71', '#3498db', '#e74c3c']
        ax2.bar(avg_metrics.keys(), avg_metrics.values(), color=colors_bar)
        ax2.set_title('Average Metrics', fontweight='bold', fontsize=12)
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=15)
        for i, (k, v) in enumerate(avg_metrics.items()):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

        # Distribuzione classi
        ax3 = fig.add_subplot(gs[1, 1])
        class_counts = [metrics['per_class'][cid]['support'] for cid in sorted(metrics['per_class'].keys())]
        class_labels = [self.class_names[cid] for cid in sorted(metrics['per_class'].keys())]
        ax3.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Class Distribution', fontweight='bold', fontsize=12)

        # Processing time
        ax4 = fig.add_subplot(gs[1, 2])
        if self.stats['processing_times']:
            times = self.stats['processing_times']
            ax4.hist(times, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(times), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(times):.1f}ms')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Processing Time', fontweight='bold', fontsize=12)
            ax4.legend()
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No timing data', ha='center', va='center', fontsize=12)
            ax4.axis('off')

        # Accuracy per classe
        ax5 = fig.add_subplot(gs[2, :])
        sorted_classes = sorted(
            metrics['per_class'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        class_names_sorted = [self.class_names[cid] for cid, _ in sorted_classes]
        accuracy_scores = [m['accuracy'] for _, m in sorted_classes]

        colors_perf = ['#2ecc71' if acc > 0.9 else '#f39c12' if acc > 0.7 else '#e74c3c'
                       for acc in accuracy_scores]

        ax5.barh(class_names_sorted, accuracy_scores, color=colors_perf, edgecolor='black')
        ax5.set_xlabel('Accuracy', fontweight='bold')
        ax5.set_title('Accuracy per Class (Sorted)', fontweight='bold', fontsize=12)
        ax5.set_xlim([0, 1])
        ax5.grid(axis='x', alpha=0.3)

        for i, (name, score) in enumerate(zip(class_names_sorted, accuracy_scores)):
            ax5.text(score + 0.02, i, f'{score:.3f}', va='center', fontweight='bold')

        plt.suptitle('Classification Report Summary', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(os.path.join(self.save_dir, 'summary_report.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_json_report(self):
        """Salva report JSON."""
        metrics = self.compute_metrics()

        report = {
            'overall_metrics': metrics['overall'],
            'per_class_metrics': {
                self.class_names[cid]: cm
                for cid, cm in metrics['per_class'].items()
            },
            'false_positives_detail': self.stats['false_positives'],
            'false_negatives_detail': self.stats['false_negatives']
        }

        with open(os.path.join(self.save_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    def generate_all_reports(self):
        """Genera tutti i report INCLUSI FP/FN."""
        print("\n" + "=" * 70)
        print("📊 GENERAZIONE REPORT")
        print("=" * 70)

        print("\n1️⃣ Confusion Matrix...")
        self.generate_confusion_matrix()

        print("2️⃣ Metrics per Class...")
        self.generate_pr_curves()

        print("3️⃣ Results Table...")
        df = self.generate_results_table()

        print("4️⃣ Summary Plot...")
        self.generate_summary_plot()

        print("5️⃣ FP/FN Analysis...")  # NUOVO!
        self.generate_fp_fn_analysis()

        print("6️⃣ JSON Report...")
        self.save_json_report()

        print("\n✅ Report completati!")
        print(f"📁 Salvati in: {self.save_dir}")

        print("\n" + "=" * 70)
        print("RESULTS TABLE")


# ============= INTEGRAZIONE NEL TUO CODICE =============

def add_reporting_to_your_code():
    """
    Esempio di come integrare nel tuo codice.
    """

    print("""
# ===== AGGIUNGI ALL'INIZIO DEL TUO SCRIPT =====

from yolo_reports import YOLOStyleReporter

# Inizializza reporter
reporter = YOLOStyleReporter(
    class_names=class_names,  # Il tuo dizionario class_names
    save_dir='results_classification1'
)

# ===== NEL LOOP DI INFERENZA (dopo aver processato un'immagine) =====

# Dopo aver raccolto image_preds e image_labels per un'immagine:
if len(image_preds) > 0:
    reporter.update(
        predictions=image_preds,
        ground_truths=image_labels,
        confidences=None,  # Aggiungi se hai confidence scores
        processing_time=processing_time
    )

# ===== ALLA FINE, GENERA TUTTI I REPORT =====

# Genera tutti i report
reporter.generate_all_reports()

# Salva anche il tuo DataFrame originale
timeDf.to_csv(os.path.join(out_dir, "detection_results.csv"), index=False)
    """)


if __name__ == "__main__":
    add_reporting_to_your_code()

    print("\n" + "=" * 70)
    print("📚 REPORT GENERATI:")
    print("=" * 70)
    print("  ✓ confusion_matrix.png          - Matrice confusione normalizzata")
    print("  ✓ confusion_matrix_absolute.png - Matrice confusione valori assoluti")
    print("  ✓ metrics_per_class.png         - Precision/Recall/F1 per classe")
    print("  ✓ results_table.png              - Tabella risultati completa")
    print("  ✓ summary_report.png             - Report riassuntivo")
    print("  ✓ results.csv                    - Tabella metriche CSV")
    print("  ✓ report.json                    - Report completo JSON")
    print("=" * 70)