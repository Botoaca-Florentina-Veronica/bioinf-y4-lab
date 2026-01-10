"""
Exercise 10 — PCA Single-Omics vs Joint

TODO:
- încărcați SNP și Expression
- normalizați fiecare strat (z-score)
- rulați PCA pe:
    1) strat SNP
    2) strat Expression
    3) strat Joint (concat)
- generați 3 figuri PNG
- comparați vizual distribuția probelor
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Înlocuiește cu handle-ul tău
HANDLE = "Botoaca-Florentina-Veronica" 

SNP_CSV = Path("/workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/snp_matrix_test_Botoaca-Florentina-Veronica.csv")
EXP_CSV = Path("/workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/expression_matrix_test_Botoaca-Florentina-Veronica.csv")

OUT_DIR = Path(f"labs/10_integrative/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Încarcă și preprocesează datele SNP și Expression"""
    
    # 1. Încărcare date
    snp_data = pd.read_csv(SNP_CSV, index_col=0)
    expr_data = pd.read_csv(EXP_CSV, index_col=0)
    
    print(f"Dimensiuni inițiale - SNP: {snp_data.shape}, Expression: {expr_data.shape}")
    
    # 2. Identifică probele comune (coloanele)
    common_samples = snp_data.columns.intersection(expr_data.columns)
    print(f"Probe comune: {len(common_samples)}")
    
    # 3. Selectează doar probele comune
    snp_data = snp_data[common_samples]
    expr_data = expr_data[common_samples]
    
    # 4. Transpune pentru a avea probe pe rânduri și features pe coloane
    snp_data = snp_data.T  # Probe x SNP-uri
    expr_data = expr_data.T  # Probe x Gene
    
    return snp_data, expr_data, common_samples

def normalize_zscore(data):
    """Normalizează datele folosind z-score"""
    scaler = StandardScaler()
    normalized = scaler.fit_transform(data)
    return pd.DataFrame(normalized, index=data.index, columns=data.columns)

def run_pca_and_plot(data, title, filename, color_labels=None):
    """Rulează PCA și generează plot"""
    
    # Rulează PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    # Creează DataFrame pentru rezultate PCA
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1]
    }, index=data.index)
    
    # Creează figura
    plt.figure(figsize=(10, 8))
    
    # Dacă avem etichete de culoare (subtype), folosește-le
    if color_labels is not None:
        unique_labels = color_labels.unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = color_labels == label
            plt.scatter(pca_df.loc[mask, 'PC1'], 
                       pca_df.loc[mask, 'PC2'], 
                       c=[colors[i]], label=label, alpha=0.7, s=50)
        plt.legend(title='Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, s=50)
    
    plt.title(f'PCA: {title}\nVariance PC1: {pca.explained_variance_ratio_[0]:.2%}, '
              f'PC2: {pca.explained_variance_ratio_[1]:.2%}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True, alpha=0.3)
    
    # Salvează figura
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca_df, pca

def create_joint_dataframe(snp_norm, expr_norm):
    """Creează matricea integrată și o salvează"""
    
    # Concatenează straturile
    # Adaugă prefixe pentru a diferenția SNP-urile de gene
    snp_norm_prefixed = snp_norm.copy()
    snp_norm_prefixed.columns = ['SNP_' + str(col) for col in snp_norm_prefixed.columns]
    
    expr_norm_prefixed = expr_norm.copy()
    expr_norm_prefixed.columns = ['GENE_' + str(col) for col in expr_norm_prefixed.columns]
    
    # Concatenează pe coloane (features)
    joint_data = pd.concat([snp_norm_prefixed, expr_norm_prefixed], axis=1)
    
    # Salvează matricea integrată
    joint_csv_path = OUT_DIR / f"multiomics_concat_{HANDLE}.csv"
    joint_data.to_csv(joint_csv_path)
    print(f"Matricea integrată salvată: {joint_csv_path}")
    print(f"Dimensiuni matrice integrată: {joint_data.shape}")
    
    return joint_data

def main():
    print("=" * 60)
    print("EXERCISE 10.1: PCA Single-Omics vs Joint")
    print("=" * 60)
    
    # 1. Încărcare și preprocesare date
    snp_data, expr_data, common_samples = load_and_preprocess_data()
    
    # 2. Normalizare z-score pentru fiecare strat
    print("\nNormalizare date...")
    snp_norm = normalize_zscore(snp_data)
    expr_norm = normalize_zscore(expr_data)
    
    # 3. Creează etichete pentru colorare (exemplu: primele 2 litere din numele probei)
    # În practică, ai avea un fișier cu subtipurile
    # Aici folosim o simulare pentru demonstrație
    np.random.seed(42)
    subtypes = np.random.choice(['A', 'B', 'C'], size=len(common_samples))
    subtype_series = pd.Series(subtypes, index=common_samples)
    
    # 4. PCA pe stratul SNP
    print("\nRulează PCA pe SNP...")
    pca_snp_df, pca_snp = run_pca_and_plot(
        snp_norm, 
        'SNP Data Only', 
        f'pca_snp_{HANDLE}.png',
        color_labels=subtype_series
    )
    
    # 5. PCA pe stratul Expression
    print("Rulează PCA pe Expression...")
    pca_expr_df, pca_expr = run_pca_and_plot(
        expr_norm, 
        'Expression Data Only', 
        f'pca_expr_{HANDLE}.png',
        color_labels=subtype_series
    )
    
    # 6. Creează și salvează matricea integrată
    print("\nCreare matrice integrată...")
    joint_data = create_joint_dataframe(snp_norm, expr_norm)
    
    # 7. PCA pe matricea integrată
    print("Rulează PCA pe matricea integrată...")
    pca_joint_df, pca_joint = run_pca_and_plot(
        joint_data, 
        'Integrated Multi-Omics Data', 
        f'pca_joint_{HANDLE}.png',
        color_labels=subtype_series
    )
    
    # 8. Comparație statistică
    print("\n" + "=" * 60)
    print("COMPARAȚIE VARIENȚĂ EXPLICATĂ:")
    print("-" * 60)
    print(f"SNP PCA - Variance explained: {pca_snp.explained_variance_ratio_.sum():.2%}")
    print(f"  PC1: {pca_snp.explained_variance_ratio_[0]:.2%}, "
          f"PC2: {pca_snp.explained_variance_ratio_[1]:.2%}")
    
    print(f"\nExpression PCA - Variance explained: {pca_expr.explained_variance_ratio_.sum():.2%}")
    print(f"  PC1: {pca_expr.explained_variance_ratio_[0]:.2%}, "
          f"PC2: {pca_expr.explained_variance_ratio_[1]:.2%}")
    
    print(f"\nJoint PCA - Variance explained: {pca_joint.explained_variance_ratio_.sum():.2%}")
    print(f"  PC1: {pca_joint.explained_variance_ratio_[0]:.2%}, "
          f"PC2: {pca_joint.explained_variance_ratio_[1]:.2%}")
    
    print("\n" + "=" * 60)
    print("Proces finalizat cu succes!")
    print(f"Fișiere generate în: {OUT_DIR}")

if __name__ == "__main__":
    main()