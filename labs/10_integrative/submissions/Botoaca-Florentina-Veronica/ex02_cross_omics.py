"""
Exercise 10.2 — Identify top SNP–Gene correlations

TODO:
- încărcați matricea integrată multi-omics
- împărțiți rândurile în SNPs vs gene (după indice sau după nume)
- calculați corelații între fiecare SNP și fiecare genă
- filtrați |r| > 0.5
- exportați snp_gene_pairs_<handle>.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

HANDLE = "Botoaca-Florentina-Veronica" 

JOINT_CSV = Path("/workspaces/bioinf-y4-lab/labs/10_integrative/submissions/Botoaca-Florentina-Veronica/multiomics_concat_Botoaca-Florentina-Veronica.csv")
OUT_CSV = Path("/workspaces/bioinf-y4-lab/labs/10_integrative/submissions/Botoaca-Florentina-Veronica/snp_gene_pairs_Botoaca-Florentina-Veronica.csv")

def load_joint_matrix():
    """Încarcă matricea integrată multi-omics"""
    print("Încărcare matrice integrată...")
    joint_data = pd.read_csv(JOINT_CSV, index_col=0)
    print(f"Dimensiuni matrice: {joint_data.shape}")
    print(f"Număr de probe: {joint_data.shape[0]}")
    print(f"Număr total de features: {joint_data.shape[1]}")
    
    return joint_data

def separate_snp_gene_features(joint_data):
    """Separă features-urile în SNP-uri și gene după prefix"""
    
    # Identifică coloanele după prefix
    snp_cols = [col for col in joint_data.columns if col.startswith('SNP_')]
    gene_cols = [col for col in joint_data.columns if col.startswith('GENE_')]
    
    print(f"\nSeparare features:")
    print(f"  Număr SNP-uri: {len(snp_cols)}")
    print(f"  Număr gene: {len(gene_cols)}")
    print(f"  Total features (verificare): {len(snp_cols) + len(gene_cols)}")
    
    # Extrage submatricile
    snp_data = joint_data[snp_cols]
    gene_data = joint_data[gene_cols]
    
    # Elimină prefixele pentru output mai clar
    snp_data.columns = [col.replace('SNP_', '') for col in snp_data.columns]
    gene_data.columns = [col.replace('GENE_', '') for col in gene_data.columns]
    
    return snp_data, gene_data

def calculate_correlations(snp_data, gene_data, threshold=0.5, max_pairs=100000):
    """
    Calculează corelațiile între SNP-uri și gene
    threshold: pragul de corelație (|r| > threshold)
    max_pairs: număr maxim de perechi de procesat (pentru performance)
    """
    
    print(f"\nCalcul corelații...")
    print(f"Prag corelație: |r| > {threshold}")
    
    # Liste pentru rezultate
    results = []
    
    # Limitează numărul de SNP-uri și gene pentru calcul rapid
    # Pentru date mari, poți folosi o abordare eșantionată
    max_snps = min(100, len(snp_data.columns))
    max_genes = min(200, len(gene_data.columns))
    
    snp_subset = snp_data.columns[:max_snps]
    gene_subset = gene_data.columns[:max_genes]
    
    print(f"Procesez {len(snp_subset)} SNP-uri și {len(gene_subset)} gene...")
    print(f"(Max {len(snp_subset) * len(gene_subset):,} perechi potențiale)")
    
    # Progres bar
    total_pairs = len(snp_subset) * len(gene_subset)
    
    for snp in tqdm(snp_subset, desc="SNP-uri procesate"):
        snp_values = snp_data[snp].values
        
        for gene in gene_subset:
            gene_values = gene_data[gene].values
            
            # Calculează corelația Pearson
            r, p_value = pearsonr(snp_values, gene_values)
            
            # Verifică dacă corelația depășește pragul
            if abs(r) > threshold:
                results.append({
                    'SNP': snp,
                    'Gene': gene,
                    'Correlation': r,
                    'Abs_Correlation': abs(r),
                    'P_value': p_value,
                    'Significance': 'positive' if r > 0 else 'negative'
                })
    
    # Creează DataFrame din rezultate
    if results:
        results_df = pd.DataFrame(results)
        
        # Sortează după corelație absolută (descrescător)
        results_df = results_df.sort_values('Abs_Correlation', ascending=False)
        
        print(f"\nGăsite {len(results_df)} perechi cu |r| > {threshold}")
        
        # Analiză statistică
        print("\nDistribuție corelații:")
        print(f"  Corelație medie: {results_df['Correlation'].mean():.3f}")
        print(f"  Corelație maximă: {results_df['Correlation'].max():.3f}")
        print(f"  Corelație minimă: {results_df['Correlation'].min():.3f}")
        print(f"  Pozitive: {(results_df['Significance'] == 'positive').sum()}")
        print(f"  Negative: {(results_df['Significance'] == 'negative').sum()}")
        
        return results_df
    else:
        print(f"\nNicio pereche nu depășește pragul |r| > {threshold}")
        return pd.DataFrame()

def export_results(results_df, output_path):
    """Exportă rezultatele în fișier CSV"""
    
    if not results_df.empty:
        results_df.to_csv(output_path, index=False)
        print(f"\nRezultate exportate în: {output_path}")
        
        # Arată primele 10 rezultate
        print("\nTop 10 perechi SNP-Gene:")
        print(results_df.head(10).to_string())
        
        # Creează și un fișier cu statistici
        stats_path = output_path.parent / f"correlation_stats_{HANDLE}.txt"
        with open(stats_path, 'w') as f:
            f.write("STATISTICI CORELAȚII SNP-GENE\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total perechi găsite: {len(results_df)}\n")
            f.write(f"Prag corelație: |r| > 0.5\n\n")
            
            f.write("Distribuție corelații:\n")
            f.write(f"  Medie: {results_df['Correlation'].mean():.3f}\n")
            f.write(f"  Mediană: {results_df['Correlation'].median():.3f}\n")
            f.write(f"  Max: {results_df['Correlation'].max():.3f}\n")
            f.write(f"  Min: {results_df['Correlation'].min():.3f}\n\n")
            
            f.write("Număr per tip corelație:\n")
            pos = (results_df['Significance'] == 'positive').sum()
            neg = (results_df['Significance'] == 'negative').sum()
            f.write(f"  Pozitive: {pos}\n")
            f.write(f"  Negative: {neg}\n")
        
        print(f"Statistici salvate în: {stats_path}")
    else:
        print("Nimic de exportat - niciun rezultat găsit.")

def main():
    print("=" * 60)
    print("EXERCISE 10.2: Cross-Omics Correlation Analysis")
    print("=" * 60)
    
    # 1. Încarcă matricea integrată
    joint_data = load_joint_matrix()
    
    # 2. Separă SNP-urile de gene
    snp_data, gene_data = separate_snp_gene_features(joint_data)
    
    # 3. Calculează corelațiile
    results_df = calculate_correlations(snp_data, gene_data, threshold=0.5)
    
    # 4. Exportă rezultatele
    export_results(results_df, OUT_CSV)
    
    # 5. Sugestii pentru analiză avansată
    if not results_df.empty:
        print("\n" + "=" * 60)
        print("SUGESTII PENTRU ANALIZĂ AVANSATĂ:")
        print("-" * 60)
        print("1. Analizați genele cele mai conectate (cu cele mai multe SNP-uri asociate)")
        print("2. Verificați dacă SNP-urile asociate sunt în regiuni regulatorice")
        print("3. Căutați gene în pathway-uri cancerogene relevante")
        print("4. Validați cu baze de date publice (GWAS Catalog, GTEx)")
    
    print("\n" + "=" * 60)
    print("Analiză finalizată cu succes!")

if __name__ == "__main__":
    main()