import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def generate_inference_times_exact(n_samples=203,
                                   mean=100.0,
                                   std=15.0,
                                   min_value=None,
                                   max_value=None,
                                   outliers_count=0,
                                   outliers_values=None,
                                   seed=42,
                                   tolerance=1e-6):
    """
    Genera esattamente n_samples con media e std PRECISE.

    Parameters:
    -----------
    n_samples : int
        Numero totale di campioni (default: 203)
    mean : float
        Media ESATTA desiderata (ms)
    std : float
        Deviazione standard ESATTA desiderata (ms)
    min_value : float, optional
        Valore minimo assoluto
    max_value : float, optional
        Valore massimo assoluto
    outliers_count : int
        Numero di outlier da inserire
    outliers_values : list, optional
        Valori esatti degli outlier (se None, vengono generati random)
    seed : int
        Seed per riproducibilità
    tolerance : float
        Tolleranza per convergenza (default: 1e-6)

    Returns:
    --------
    np.array : Array con media e std esatte
    """
    np.random.seed(seed)

    # Genera outlier se specificati
    if outliers_count > 0:
        if outliers_values is not None:
            outliers = np.array(outliers_values[:outliers_count])
        else:
            # Outlier lontani dalla media
            outliers = np.random.uniform(mean + 3 * std, mean + 5 * std, outliers_count)

        n_normal = n_samples - outliers_count
    else:
        outliers = np.array([])
        n_normal = n_samples

    # PASSO 1: Genera dati normali standard (media=0, std=1)
    data_std = np.random.randn(n_normal)

    # PASSO 2: Standardizza esattamente (media=0, std=1)
    data_std = (data_std - np.mean(data_std)) / np.std(data_std)

    # PASSO 3: Scala e shifta per ottenere media e std desiderati
    data = data_std * std + mean

    # PASSO 4: Applica limiti se specificati
    if min_value is not None or max_value is not None:
        # Clippa i valori
        data_clipped = np.clip(data, min_value, max_value)

        # CORREZIONE: Riscala per mantenere std dopo clipping
        # Calcola quanto è cambiata la std
        current_std = np.std(data_clipped)
        current_mean = np.mean(data_clipped)

        if current_std > 0:
            # Riscala per recuperare la std
            data = (data_clipped - current_mean) / current_std * std + mean
            # Ri-clippa
            data = np.clip(data, min_value, max_value)

    # PASSO 5: Aggiungi outlier
    if outliers_count > 0:
        data = np.concatenate([data, outliers])
        np.random.shuffle(data)

    # PASSO 6: CORREZIONE FINALE per garantire media e std ESATTE
    # Questo passo assicura precisione numerica
    current_mean = np.mean(data)
    current_std = np.std(data, ddof=0)  # popolazione, non campione

    # Aggiusta ogni valore proporzionalmente
    data = (data - current_mean) / current_std * std + mean

    # Verifica finale
    final_mean = np.mean(data)
    final_std = np.std(data, ddof=0)

    assert abs(final_mean - mean) < tolerance, f"Mean mismatch: {final_mean} vs {mean}"
    assert abs(final_std - std) < tolerance, f"Std mismatch: {final_std} vs {std}"

    return data


def generate_inference_times_with_constraints(n_samples=203,
                                              mean=100.0,
                                              std=15.0,
                                              min_value=50.0,
                                              max_value=150.0,
                                              outliers_count=0,
                                              outliers_range=(160, 200),
                                              seed=42,
                                              max_iterations=10000):
    """
    Versione iterativa che rispetta TUTTI i vincoli:
    - Media esatta
    - Std esatta
    - Min/Max rispettati
    - Outlier inseriti

    Usa ottimizzazione iterativa per trovare la configurazione migliore.
    """
    np.random.seed(seed)

    best_data = None
    best_error = float('inf')

    # Calcola il numero di campioni normali
    n_normal = n_samples - outliers_count

    for iteration in range(max_iterations):
        # Genera campioni base
        data = np.random.randn(n_normal)

        # Standardizza
        data = (data - np.mean(data)) / np.std(data)

        # Scala per ottenere la std desiderata
        data = data * std + mean

        # Applica limiti
        if min_value is not None:
            data = np.maximum(data, min_value)
        if max_value is not None:
            data = np.minimum(data, max_value)

        # Aggiungi outlier
        if outliers_count > 0:
            outliers = np.random.uniform(outliers_range[0], outliers_range[1], outliers_count)
            data_full = np.concatenate([data, outliers])
        else:
            data_full = data

        # Calcola statistiche
        current_mean = np.mean(data_full)
        current_std = np.std(data_full, ddof=0)

        # Calcola errore
        error = abs(current_mean - mean) + abs(current_std - std)

        # Controlla se i limiti sono rispettati
        if min_value is not None and np.min(data_full) < min_value:
            continue
        if max_value is not None and np.max(data_full) > max_value:
            continue

        # Salva se migliore
        if error < best_error:
            best_error = error
            best_data = data_full.copy()

            # Esci se abbastanza buono
            if error < 0.01:
                print(f"  ✓ Converged at iteration {iteration}")
                break

    if best_data is None:
        raise ValueError("Could not generate data with given constraints. Try relaxing min/max or reducing outliers.")

    # Shuffle
    np.random.shuffle(best_data)

    return best_data


def generate_exact_moments(n_samples=203, mean=100.0, std=15.0, seed=42):
    """
    Metodo MATEMATICO per generare esattamente media e std,
    senza limiti o outlier.

    Questo è il metodo più preciso possibile.
    """
    np.random.seed(seed)

    # Genera n-1 valori casuali
    data = np.random.randn(n_samples - 1)

    # Calcola l'ultimo valore in modo che la media sia esatta
    last_value = n_samples * mean - np.sum(data)
    data = np.append(data, last_value)

    # Ora aggiusta per ottenere la std esatta
    current_mean = np.mean(data)
    current_std = np.std(data, ddof=0)

    # Scala per std esatta
    if current_std > 0:
        data = (data - current_mean) / current_std * std + mean

    # Verifica
    assert abs(np.mean(data) - mean) < 1e-10
    assert abs(np.std(data, ddof=0) - std) < 1e-10

    return data


def visualize_generated_data(data, title="Generated Data", target_mean=None, target_std=None):
    """Visualizza con verifica di media e std"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    actual_mean = np.mean(data)
    actual_std = np.std(data, ddof=0)

    # Histogram
    axes[0].hist(data, bins=40, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    axes[0].axvline(actual_mean, color='red', linestyle='--', linewidth=2.5,
                    label=f'Actual Mean: {actual_mean:.4f}')
    if target_mean is not None:
        axes[0].axvline(target_mean, color='orange', linestyle=':', linewidth=2.5,
                        label=f'Target Mean: {target_mean:.4f}')

    # Overlay normal distribution
    x = np.linspace(data.min(), data.max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, actual_mean, actual_std),
                 'g-', linewidth=2, label=f'Normal PDF (μ={actual_mean:.2f}, σ={actual_std:.2f})')

    axes[0].set_xlabel('Value (ms)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Boxplot
    bp = axes[1].boxplot(data, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
    axes[1].scatter([1] * len(data), data, alpha=0.4, color='steelblue', s=30, edgecolors='black', linewidth=0.5)
    axes[1].axhline(actual_mean, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].set_ylabel('Value (ms)', fontsize=11, fontweight='bold')
    axes[1].set_title('Boxplot', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

    # Stampa statistiche dettagliate
    print(f"\n{'=' * 70}")
    print(f"📊 STATISTICS: {title}")
    print(f"{'=' * 70}")
    print(f"  N samples:        {len(data)}")
    print(f"  Mean:             {actual_mean:.10f} ms")
    if target_mean is not None:
        error_mean = abs(actual_mean - target_mean)
        print(f"  Target Mean:      {target_mean:.10f} ms")
        print(f"  Mean Error:       {error_mean:.2e} {'✓' if error_mean < 1e-6 else '✗'}")

    print(f"  Std (population): {actual_std:.10f} ms")
    if target_std is not None:
        error_std = abs(actual_std - target_std)
        print(f"  Target Std:       {target_std:.10f} ms")
        print(f"  Std Error:        {error_std:.2e} {'✓' if error_std < 1e-6 else '✗'}")

    print(f"  Std (sample):     {np.std(data, ddof=1):.10f} ms")
    print(f"  Min:              {np.min(data):.4f} ms")
    print(f"  Max:              {np.max(data):.4f} ms")
    print(f"  Median:           {np.median(data):.4f} ms")
    print(f"  Q1 (25%):         {np.percentile(data, 25):.4f} ms")
    print(f"  Q3 (75%):         {np.percentile(data, 75):.4f} ms")
    print(f"  IQR:              {np.percentile(data, 75) - np.percentile(data, 25):.4f} ms")
    print(f"  Skewness:         {stats.skew(data):.4f}")
    print(f"  Kurtosis:         {stats.kurtosis(data):.4f}")
    print(f"{'=' * 70}\n")


# ============================================
# ESEMPI DI UTILIZZO
# ============================================

if __name__ == "__main__":
    print("\n" + "🎯 METODO 1: Generazione ESATTA (senza limiti)")
    print("=" * 70)
    data1 = generate_exact_moments(
        n_samples=203,
        mean=142.42,
        std=1.2,
        seed=42
    )
    visualize_generated_data(data1, "Exact Method (No Constraints)",
                             target_mean=142.42, target_std=1.2)


    df = pd.DataFrame(data1)
    df.to_csv('exact_synthetic_times.csv', index=False)
    print("\n💾 Data saved to: exact_synthetic_times.csv")