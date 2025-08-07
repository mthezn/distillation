import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, f_oneway, kruskal, mannwhitneyu
import warnings
import argparse

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")


def check_normality(data_dict):
    rows = []
    for model, values in data_dict.items():
        stat, p = shapiro(values)
        rows.append({"Model": model, "W-statistic": stat, "p-value": p, "Normal": p > 0.05})
    return pd.DataFrame(rows)

def perform_anova_or_kruskal(data_dict, all_normal):
    data_cleaned = [vals for vals in data_dict.values()]
    if all_normal:
        stat, p = f_oneway(*data_cleaned)
        return "ANOVA", stat, p
    else:
        stat, p = kruskal(*data_cleaned)
        return "Kruskal-Wallis", stat, p

def pairwise_tests(data_dict):
    """Mann-Whitney U test pairwise, with Bonferroni correction"""
    models = list(data_dict.keys())
    results = []
    num_comparisons = len(models) * (len(models) - 1) // 2
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            v1, v2 = data_dict[m1], data_dict[m2]
            stat, p = mannwhitneyu(v1, v2, alternative='two-sided')
            corrected_p = p * num_comparisons  # Bonferroni
            corrected_p = min(corrected_p, 1.0)
            results.append({
                "Model 1": m1,
                "Model 2": m2,
                "U-statistic": stat,
                "p-value": p,
                "corrected p": corrected_p,
                "Significance": (
                    '***' if corrected_p < 0.001 else
                    '**' if corrected_p < 0.01 else
                    '*' if corrected_p < 0.05 else
                    'ns'
                )
            })
    return pd.DataFrame(results)

def save_plot(data_dict, metric_name, significance_df, plot_star):
    df = pd.DataFrame([(k, val) for k, vs in data_dict.items() for val in vs], columns=["Model", metric_name])
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Model", y=metric_name, data=df, palette="Set2", showfliers=False)
    plt.title(f"{metric_name} Comparison Across Models")
    plt.xticks(rotation=45)

    model_positions = {model: i for i, model in enumerate(df["Model"].unique())}
    y_max = df[metric_name].max()

    if plot_star:
        for idx, row in significance_df.iterrows():
            if row["Significance"] == "ns":
                continue  # skip non-significant
            x1 = model_positions[row["Model 1"]]
            x2 = model_positions[row["Model 2"]]
            y = y_max + 0.1 + idx * 0.1
            plt.plot([x1, x1, x2, x2], [y, y+0.03, y+0.03, y], lw=1.5, c='k')
            plt.text((x1 + x2)/2, y + 0.035, row["Significance"], ha='center', va='bottom', color='k')

    plt.tight_layout()
    plt.savefig(f"plots/{metric_name}_comparison.png")
    plt.close()

def main(args):

    file_path = args.input_file
    
    xls = pd.ExcelFile(file_path)
    metrics = xls.sheet_names

    os.makedirs("plots", exist_ok=True)
    results_dict = {}
    for metric in metrics:
        df = xls.parse(metric).dropna(axis=1, how='all')
        df = df.loc[:, df.columns != 'Unnamed: 0']
        data_dict = {col: df[col].dropna().astype(float).tolist() for col in df.columns}

        # Normality check
        norm_df = check_normality(data_dict)
        all_normal = norm_df["Normal"].all()

        # ANOVA/Kruskal
        test_type, stat, p_val = perform_anova_or_kruskal(data_dict, all_normal)
        test_df = pd.DataFrame([{
            "Test Type": test_type,
            "Statistic": stat,
            "p-value": p_val
        }])

        # Pairwise significance (Mann-Whitney + Bonferroni)
        pairwise_df = pairwise_tests(data_dict)

        # Plot with significance
        save_plot(data_dict, metric_name=metric, significance_df=pairwise_df, plot_star=False)

        # Store results
        results_dict[metric] = {
            "Normality": norm_df,
            "Main Test": test_df,
            "Pairwise": pairwise_df
        }


    with pd.ExcelWriter("statistical_results.xlsx") as writer:
        for metric, sections in results_dict.items():
            sections["Normality"].to_excel(writer, sheet_name=f"{metric}_normality", index=False)
            sections["Main Test"].to_excel(writer, sheet_name=f"{metric}_main_test", index=False)
            sections["Pairwise"].to_excel(writer, sheet_name=f"{metric}_pairwise", index=False)

    print("✅ Statistical results saved to 'statistical_results.xlsx'")
    print("📊 Annotated plots saved to 'plots/' folder")

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Statistical Analysis of Model Metrics")
    parser.add_argument("--input_file", type=str, default="data_analysis/Book1.xlsx")

    args = parser.parse_args()
    print(f"Starting statistical analysis on file {args.input_file} ...")
    
    main(args)

