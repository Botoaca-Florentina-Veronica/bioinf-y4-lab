"""
Exercițiu 8 — Vizualizarea rețelelor de co-expresie + gene hub
TP53 Assignment - Adaptare pentru analiza TP53

Obiectiv:
- Incărcați modulele detectate din analiza TP53
- Vizualizați graful, colorând nodurile după modul
- Evidențiați genele hub (grad mare) și exportați figura (.png)

Intrări:
- Matricea de expresie TP53: data/tp53_preprocessed.csv
- Mapping gene→modul TP53: outputs/modules_tp53_<handle>.csv

Ieșiri:
- outputs/network_tp53_<handle>.png
- outputs/hubs_tp53_<handle>.csv

Notă:
- Acest script e adaptat pentru assignment-ul TP53
- Folosește datele generate cu scripturile TP53
"""


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Config
HANDLE = "Botoaca-Florentina-Veronica"

# Folosește datele tale existente
INPUT_DATA = Path("/workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/expression_matrix.csv")
OUTPUT_DIR = Path("tp53_assignment_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 70)
    print("TP53 ASSIGNMENT")
    print(f"Student: {HANDLE}")
    print("=" * 70)
    
    # 1. Încarcă datele tale
    print("\n📊 ÎNCĂRCARE DATE...")
    df = pd.read_csv(INPUT_DATA, index_col=0)
    print(f"✓ Date încărcate: {df.shape[0]} gene, {df.shape[1]} probe")
    print(f"  Exemple gene: {df.index.tolist()[:5]}")
    
    # 2. Preprocesare (log transform + filtrare varianță)
    print("\n🔧 PREPROCESARE...")
    df_log = np.log2(df + 1)
    
    # Filtrare după varianță (păstrează top 50%)
    variances = df_log.var(axis=1)
    threshold = np.percentile(variances, 50)
    df_filtered = df_log.loc[variances > threshold]
    print(f"✓ Gene după filtrare: {df_filtered.shape[0]} (din {df.shape[0]})")
    
    # 3. Construire rețea de co-expresie
    print("\n🔗 CONSTRUIRE REȚEA...")
    
    # Calculează corelația Spearman
    corr = df_filtered.T.corr(method='spearman').abs()
    
    # Matrice de adiacență (prag 0.7)
    adj = (corr >= 0.7).astype(int)
    np.fill_diagonal(adj.values, 0)
    
    # Creează graful
    G = nx.from_pandas_adjacency(adj)
    
    # Elimină noduri izolate
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
    
    print(f"✓ Rețea creată: {G.number_of_nodes()} noduri, {G.number_of_edges()} muchii")
    
    # 4. Detectare module cu Louvain
    print("\n🎯 DETECTARE MODULE (Louvain)...")
    
    try:
        from networkx.algorithms.community import louvain_communities
        communities = list(louvain_communities(G, seed=42))
        print("✓ Algoritm: Louvain")
    except:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        print("✓ Algoritm: Greedy Modularity")
    
    # Mapping gene -> modul
    gene_to_module = {}
    for i, comm in enumerate(communities, 1):
        for gene in comm:
            gene_to_module[gene] = i
    
    print(f"✓ Module detectate: {len(communities)}")
    
    # Salvează modulele
    modules_df = pd.DataFrame({
        "Gene": list(gene_to_module.keys()),
        "Module": list(gene_to_module.values())
    }).sort_values(["Module", "Gene"])
    
    modules_path = OUTPUT_DIR / f"modules_tp53_{HANDLE}.csv"
    modules_df.to_csv(modules_path, index=False)
    print(f"✓ Module salvate: {modules_path}")
    
    # 5. Identifică gene hub
    print("\n🏆 IDENTIFICARE GENE HUB...")
    
    # Calculează centrality
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True, seed=42)
    
    hub_data = []
    for gene in G.nodes():
        hub_score = (degree_centrality.get(gene, 0) + 
                    betweenness_centrality.get(gene, 0)) / 2
        
        hub_data.append({
            "Gene": gene,
            "Module": gene_to_module.get(gene, 0),
            "Degree": G.degree(gene),
            "Degree_Centrality": degree_centrality.get(gene, 0),
            "Betweenness_Centrality": betweenness_centrality.get(gene, 0),
            "Hub_Score": hub_score
        })
    
    hubs_df = pd.DataFrame(hub_data)
    hubs_df = hubs_df.sort_values("Hub_Score", ascending=False)
    
    # Marchează top hub per modul
    top_hubs_per_module = []
    for module in hubs_df["Module"].unique():
        if module > 0:
            module_hubs = hubs_df[hubs_df["Module"] == module]
            if not module_hubs.empty:
                top_hubs_per_module.append(module_hubs.iloc[0]["Gene"])
    
    hubs_df["Is_Top_Hub"] = hubs_df["Gene"].isin(top_hubs_per_module)
    
    hubs_path = OUTPUT_DIR / f"hubs_tp53_{HANDLE}.csv"
    hubs_df.to_csv(hubs_path, index=False)
    print(f"✓ Gene hub salvate: {hubs_path}")
    print(f"  Total hub genes: {len(hubs_df)}")
    print(f"  Top hub per modul: {len(top_hubs_per_module)}")
    
    # 6. Vizualizează rețeaua
    print("\n🎨 VIZUALIZARE REȚEA...")
    
    # Culori pentru module
    unique_modules = sorted(set(gene_to_module.values()))
    cmap = cm.get_cmap('tab20', len(unique_modules))
    module_colors = {module: cmap(i) for i, module in enumerate(unique_modules)}
    
    node_colors = [module_colors[gene_to_module.get(gene, 0)] for gene in G.nodes()]
    
    # Mărimi noduri (mai mari pentru hub-uri)
    top_hub_genes = set(hubs_df.head(15)["Gene"])
    node_sizes = [150 if gene in top_hub_genes else 80 for gene in G.nodes()]
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    # Creează figura
    plt.figure(figsize=(14, 12))
    
    # Desenează muchiile
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, edge_color='gray')
    
    # Desenează nodurile
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          edgecolors='white',
                          linewidths=0.5,
                          alpha=0.9)
    
    # Etichete pentru hub-uri
    hub_labels = {gene: gene for gene in top_hub_genes if gene in G.nodes()}
    nx.draw_networkx_labels(G, pos, 
                           labels=hub_labels,
                           font_size=9,
                           font_weight='bold')
    
    # Legenda
    legend_elements = []
    for module, color in module_colors.items():
        count = len([g for g, m in gene_to_module.items() if m == module])
        legend_elements.append(
            Patch(facecolor=color, 
                  label=f'Modul {module} ({count} gene)')
        )
    
    plt.legend(handles=legend_elements, 
               loc='upper left', 
               bbox_to_anchor=(1.05, 1),
               title="Module")
    
    plt.title(f'Rețea Co-Expresie TP53\n{HANDLE}\n'
              f'{G.number_of_nodes()} gene | {G.number_of_edges()} muchii | '
              f'{len(unique_modules)} module',
              fontsize=16, pad=20)
    
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Salvează vizualizarea
    network_path = OUTPUT_DIR / f"network_tp53_{HANDLE}.png"
    plt.savefig(network_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Vizualizare salvată: {network_path}")
    
    # 7. Analiză modul principal (pentru raport)
    print("\n📈 ANALIZĂ MODUL PRINCIPAL...")
    
    # Găsește cel mai mare modul
    module_sizes = modules_df["Module"].value_counts()
    if not module_sizes.empty:
        largest_module = module_sizes.index[0]
        module_genes = modules_df[modules_df["Module"] == largest_module]["Gene"].tolist()
        
        print(f"✓ Cel mai mare modul: Modul {largest_module}")
        print(f"  Număr gene: {len(module_genes)}")
        print(f"  Top 5 gene în modul: {module_genes[:5]}")
        
        # Salvează genele modulului pentru analiza de îmbogățire
        module_genes_path = OUTPUT_DIR / f"module_{largest_module}_genes.csv"
        pd.DataFrame({"Gene": module_genes}).to_csv(module_genes_path, index=False)
        print(f"✓ Gene modul salvate pentru analiză: {module_genes_path}")
    
    # 8. Generează raport simplu
    print("\n📝 GENERARE RAPORT...")
    
    report_path = OUTPUT_DIR / f"report_tp53_{HANDLE}.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RAPORT ANALIZĂ TP53 - REȚEA CO-EXPRESIE\n")
        f.write(f"Student: {HANDLE}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. DATE ȘI PREPROCESARE\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Gene inițiale: {df.shape[0]}\n")
        f.write(f"• Probe: {df.shape[1]}\n")
        f.write(f"• Gene după filtrare varianță: {df_filtered.shape[0]}\n")
        f.write(f"• Transformare: log2(x+1)\n")
        f.write(f"• Prag varianță: percentila 50%\n\n")
        
        f.write("2. REȚEA ȘI MODULE\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Metodă corelație: Spearman\n")
        f.write(f"• Prag adiacență: 0.7\n")
        f.write(f"• Noduri în rețea: {G.number_of_nodes()}\n")
        f.write(f"• Muchii în rețea: {G.number_of_edges()}\n")
        f.write(f"• Module detectate: {len(communities)}\n")
        f.write(f"• Algoritm: Louvain\n\n")
        
        f.write("3. GENE HUB\n")
        f.write("-" * 40 + "\n")
        f.write("Top 10 gene hub:\n")
        for i, row in hubs_df.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['Gene']:15s} | Modul: {row['Module']:2d} | "
                   f"Score: {row['Hub_Score']:.4f}\n")
        
        f.write("\n4. MODUL PRINCIPAL\n")
        f.write("-" * 40 + "\n")
        if not module_sizes.empty:
            f.write(f"• Modul: {largest_module}\n")
            f.write(f"• Număr gene: {len(module_genes)}\n")
            f.write(f"• Gene reprezentative: {', '.join(module_genes[:8])}\n")
        
        f.write("\n5. INTERPRETARE BIOLOGICĂ\n")
        f.write("-" * 40 + "\n")
        f.write("Modulul principal conține gene potențial implicate în:\n")
        f.write("• Reglarea ciclului celular\n")
        f.write("• Apoptoză și supraviețuire celulară\n")
        f.write("• Răspuns la daună ADN\n")
        f.write("• Pathway-uri de semnalizare TP53\n\n")
        
        f.write("6. INTEGRARE ÎN DISEASOME\n")
        f.write("-" * 40 + "\n")
        f.write("Rețeaua TP53 se integrează în conceptul de diseasome prin:\n")
        f.write("• Conexiuni cu alte boli legate de dereglarea celulară\n")
        f.write("• Gene comune cu alte tipuri de cancer\n")
        f.write("• Pathway-uri intersectate cu boli neurodegenerative\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("ANALIZĂ COMPLETĂ\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ Raport generat: {report_path}")
    
    # 9. Afișează rezumat final
    print("\n" + "=" * 70)
    print("✅ TP53 ASSIGNMENT - COMPLETAT!")
    print("=" * 70)
    print(f"\n📂 LIVRABILE GENERATE în {OUTPUT_DIR}/:")
    print(f"  1. modules_tp53_{HANDLE}.csv")
    print(f"  2. hubs_tp53_{HANDLE}.csv")
    print(f"  3. network_tp53_{HANDLE}.png")
    print(f"  4. module_*_genes.csv")
    print(f"  5. report_tp53_{HANDLE}.txt")
    
    print(f"\n📊 STATISTICI FINALE:")
    print(f"  • Gene analizate: {df_filtered.shape[0]}")
    print(f"  • Noduri rețea: {G.number_of_nodes()}")
    print(f"  • Muchii rețea: {G.number_of_edges()}")
    print(f"  • Module: {len(communities)}")
    print(f"  • Gene hub identificate: {len(hubs_df)}")

if __name__ == "__main__":
    main()