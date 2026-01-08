"""
Exercise 9.2 — Disease Proximity and Drug Ranking
Versiune actualizată pentru fișierele tale existente
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, List, Tuple
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Config - FOLOSEȘTE HANDLE-UL TĂU
# --------------------------
HANDLE = "Botoaca-Florentina-Veronica"

BASE_DIR = Path("labs/09_repurposing/submissions/Botoaca-Florentina-Veronica")

GRAPH_FILE = Path("/workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/network_drug_gene_Botoaca-Florentina-Veronica.graphml")

DRUG_SUMMARY_CSV = Path("/workspaces/bioinf-y4-lab/labs/09_repurposing/submissions/Botoaca-Florentina-Veronica/drug_summary_Botoaca-Florentina-Veronica.csv")

# Director pentru gene bolii - creează dacă nu există
DISEASE_GENES_DIR = Path("data/work") / HANDLE / "lab09"
DISEASE_GENES_DIR.mkdir(parents=True, exist_ok=True)
DISEASE_GENES_TXT = DISEASE_GENES_DIR / f"disease_genes_{HANDLE}.txt"

# Gene bolii exemplu (alege sau modifică)
DISEASE_GENES_EXAMPLE = [
    "ACE", "AGTR1", "AGT", "NOS3", "EDN1", 
    "ADRB1", "ADRB2", "CYP11B2", "SLC12A3",
    "KCNJ1", "ADD1", "GNB3", "NPPA", "NPPB",
    "CYP3A4", "CYP2D6", "CYP2C9", "PTGS1", "PTGS2"
]

# Output directory (același cu input-ul)
OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_PRIORITY = OUT_DIR / f"drug_priority_{HANDLE}.csv"
OUT_NETWORK_IMAGE = OUT_DIR / f"network_drug_gene_{HANDLE}.png"
OUT_REPORT = OUT_DIR / f"report_{HANDLE}.txt"


def create_disease_genes_file() -> Path:
    """
    Creează fișierul cu gene bolii dacă nu există
    """
    if DISEASE_GENES_TXT.exists():
        print(f"[INFO] Fișier gene bolii există: {DISEASE_GENES_TXT}")
        return DISEASE_GENES_TXT
    
    print(f"[INFO] Creez fișier gene bolii: {DISEASE_GENES_TXT}")
    
    # Creează fișierul
    with open(DISEASE_GENES_TXT, 'w') as f:
        for gene in DISEASE_GENES_EXAMPLE:
            f.write(f"{gene}\n")
    
    print(f"[INFO] Creat fișier cu {len(DISEASE_GENES_EXAMPLE)} gene bolii")
    print(f"[INFO] Exemple: {', '.join(DISEASE_GENES_EXAMPLE[:5])}...")
    
    return DISEASE_GENES_TXT


def load_graph() -> nx.Graph:
    """
    Încarcă graful din fișierul GraphML
    """
    print(f"[INFO] Încarc graful din {GRAPH_FILE}")
    
    if not GRAPH_FILE.exists():
        print(f"[ERROR] Fișierul {GRAPH_FILE} nu există!")
        print("[INFO] Fișiere disponibile în director:")
        for f in BASE_DIR.iterdir():
            print(f"  - {f.name}")
        raise FileNotFoundError(f"Fișierul {GRAPH_FILE} nu există")
    
    try:
        # Încarcă din GraphML
        B = nx.read_graphml(GRAPH_FILE)
        print(f"[SUCCESS] Graful încărcat: {B.number_of_nodes()} noduri, {B.number_of_edges()} muchii")
        
        # Afișează statistici
        drugs = [n for n, d in B.nodes(data=True) if d.get('type') == 'drug' or d.get('bipartite') == '0']
        genes = [n for n, d in B.nodes(data=True) if d.get('type') == 'gene' or d.get('bipartite') == '1']
        
        print(f"[INFO] - Medicamente: {len(drugs)}")
        print(f"[INFO] - Gene: {len(genes)}")
        
        return B
        
    except Exception as e:
        print(f"[ERROR] Eroare la încărcarea grafului: {e}")
        print("[INFO] Încerc reconstrucția din summary...")
        return reconstruct_graph_from_summary()


def reconstruct_graph_from_summary() -> nx.Graph:
    """
    Reconstruiește graful din fișierul summary (plan B)
    """
    print("[INFO] Reconstruiesc graful din drug_summary...")
    
    if not DRUG_SUMMARY_CSV.exists():
        print(f"[ERROR] {DRUG_SUMMARY_CSV} nu există!")
        raise FileNotFoundError("Niciun fișier de input disponibil")
    
    # Încarcă summary
    df_summary = pd.read_csv(DRUG_SUMMARY_CSV)
    print(f"[INFO] Încărcat summary: {len(df_summary)} medicamente")
    
    # Creează graful (simplificat)
    B = nx.Graph()
    
    for _, row in df_summary.iterrows():
        drug = row['drug']
        B.add_node(drug, type='drug', bipartite=0)
        
        # Extrage genele din target_genes (dacă există)
        if 'target_genes' in row and pd.notna(row['target_genes']):
            genes_str = str(row['target_genes'])
            # Încearcă să parsezi genele
            if ',' in genes_str:
                genes = [g.strip() for g in genes_str.split(',')]
                for gene in genes:
                    if gene and gene != '...':
                        B.add_node(gene, type='gene', bipartite=1)
                        B.add_edge(drug, gene)
    
    print(f"[INFO] Graf reconstruit: {B.number_of_nodes()} noduri, {B.number_of_edges()} muchii")
    return B


def load_disease_genes(path: Path) -> Set[str]:
    """
    Încarcă genele bolii
    """
    try:
        with open(path, 'r') as f:
            genes = {line.strip() for line in f if line.strip()}
        
        print(f"[INFO] Încărcat {len(genes)} gene bolii")
        if genes:
            print(f"[INFO] Primele 5 gene: {', '.join(list(genes)[:5])}")
        
        return genes
    except Exception as e:
        print(f"[WARNING] Eroare la încărcare gene bolii: {e}")
        print(f"[INFO] Folosesc gene exemplu")
        return set(DISEASE_GENES_EXAMPLE)


def get_drug_nodes(B: nx.Graph) -> List[str]:
    """Returnează nodurile de tip drug"""
    drugs = []
    for node, data in B.nodes(data=True):
        if (data.get('type') == 'drug' or 
            data.get('bipartite') == 0 or 
            data.get('bipartite') == '0' or
            data.get('node_type') == 'drug'):
            drugs.append(str(node))
    return drugs


def get_gene_nodes(B: nx.Graph) -> List[str]:
    """Returnează nodurile de tip gene"""
    genes = []
    for node, data in B.nodes(data=True):
        if (data.get('type') == 'gene' or 
            data.get('bipartite') == 1 or 
            data.get('bipartite') == '1' or
            data.get('node_type') == 'gene'):
            genes.append(str(node))
    return genes


def compute_proximity(B: nx.Graph, drug: str, disease_genes: Set[str]) -> dict:
    """
    Calculează proximitatea unui medicament față de genele bolii
    """
    if drug not in B:
        return {'closest': float('inf'), 'avg': float('inf'), 'direct': 0}
    
    distances = []
    direct_targets = []
    
    for gene in disease_genes:
        gene_str = str(gene)
        if gene_str in B:
            try:
                dist = nx.shortest_path_length(B, source=drug, target=gene_str)
                distances.append(dist)
                if dist == 1:
                    direct_targets.append(gene_str)
            except nx.NetworkXNoPath:
                distances.append(float('inf'))
    
    # Calculează metrici
    valid_distances = [d for d in distances if d != float('inf')]
    
    if not valid_distances:
        return {'closest': float('inf'), 'avg': float('inf'), 'direct': 0}
    
    result = {
        'closest': min(valid_distances),
        'avg': np.mean(valid_distances),
        'direct': len(direct_targets),
        'direct_genes': direct_targets[:3]  # Primele 3 gene directe
    }
    
    return result


def calculate_drug_ranking(B: nx.Graph, disease_genes: Set[str], max_drugs: int = 1000) -> pd.DataFrame:
    """
    Calculează ranking-ul medicamentelor
    """
    # Obține medicamentele
    all_drugs = get_drug_nodes(B)
    
    print(f"[INFO] Analizez {len(all_drugs)} medicamente...")
    
    # Verifică care gene bolii sunt în rețea
    genes_in_graph = set(get_gene_nodes(B))
    disease_in_graph = disease_genes & genes_in_graph
    print(f"[INFO] Gene bolii în rețea: {len(disease_in_graph)}/{len(disease_genes)}")
    
    if not disease_in_graph:
        print("[WARNING] Niciun gen din lista bolii nu este în rețea!")
        print("[INFO] Gene disponibile în rețea (primele 10):")
        print(f"       {', '.join(list(genes_in_graph)[:10])}")
        return pd.DataFrame()
    
    results = []
    
    for i, drug in enumerate(all_drugs[:max_drugs]):  # Limitează pentru performanță
        if i % 100 == 0 and i > 0:
            print(f"  Procesat {i}/{min(len(all_drugs), max_drugs)} medicamente...")
        
        proximity = compute_proximity(B, drug, disease_in_graph)
        
        if proximity['closest'] < float('inf'):
            results.append({
                'drug': drug,
                'closest_distance': proximity['closest'],
                'avg_distance': proximity['avg'],
                'direct_targets': proximity['direct'],
                'degree': B.degree(drug),
                'has_direct': proximity['direct'] > 0
            })
    
    if not results:
        print("[WARNING] Niciun medicament nu are conexiune cu boala!")
        return pd.DataFrame()
    
    # Creează DataFrame
    df = pd.DataFrame(results)
    
    # Sortează: mai întâi medicamente cu ținte directe, apoi după distanță
    df['priority_score'] = df.apply(
        lambda x: (0 if x['has_direct'] else 1, x['closest_distance'], x['avg_distance']),
        axis=1
    )
    
    df = df.sort_values(['has_direct', 'closest_distance', 'avg_distance'], 
                       ascending=[False, True, True])
    
    df = df.reset_index(drop=True)
    df['rank'] = df.index + 1
    
    print(f"[SUCCESS] Găsit {len(df)} medicamente cu conexiuni la boală")
    print(f"[INFO] Medicamente cu ținte directe: {df['has_direct'].sum()}")
    
    return df


def create_visualization(B: nx.Graph, disease_genes: Set[str], top_drugs: list = None):
    """
    Creează vizualizarea rețelei
    """
    print("[INFO] Creez vizualizarea rețelei...")
    
    # Obține subset pentru vizualizare
    drugs = get_drug_nodes(B)
    genes = get_gene_nodes(B)
    
    # Selectează un subset rezonabil
    max_nodes = 300
    if len(drugs) + len(genes) > max_nodes:
        print(f"[INFO] Rețeaua este mare ({len(drugs)+len(genes)} noduri). Selectez subset...")
        
        # Folosește top_drugs sau selectează aleator
        if top_drugs:
            selected_drugs = top_drugs[:30]
        else:
            selected_drugs = drugs[:30]
        
        # Găsește genele conectate la aceste medicamente
        connected_genes = set()
        for drug in selected_drugs:
            if drug in B:
                neighbors = set(B.neighbors(drug))
                connected_genes.update(neighbors)
        
        # Include genele bolii
        disease_in_graph = disease_genes & set(genes)
        connected_genes.update(disease_in_graph)
        
        # Creează subgraf
        nodes_to_keep = set(selected_drugs) | connected_genes
        B_viz = B.subgraph(nodes_to_keep)
        
        drugs_viz = get_drug_nodes(B_viz)
        genes_viz = get_gene_nodes(B_viz)
        
        print(f"[INFO] Subgraf: {len(drugs_viz)} medicamente, {len(genes_viz)} gene")
    else:
        B_viz = B
        drugs_viz = drugs
        genes_viz = genes
    
    # Setează culori și mărimi
    plt.figure(figsize=(14, 10))
    
    # Layout
    pos = nx.spring_layout(B_viz, k=0.5, iterations=50, seed=42)
    
    # Separa nodurile
    disease_genes_viz = [g for g in genes_viz if g in disease_genes]
    other_genes_viz = [g for g in genes_viz if g not in disease_genes]
    
    # Desenează
    # 1. Gene bolii - ROȘU
    nx.draw_networkx_nodes(B_viz, pos, nodelist=disease_genes_viz,
                          node_color='red', node_size=150, alpha=0.8)
    
    # 2. Alte gene - VERDE
    nx.draw_networkx_nodes(B_viz, pos, nodelist=other_genes_viz,
                          node_color='green', node_size=50, alpha=0.5)
    
    # 3. Medicamente - ALBASTRU
    nx.draw_networkx_nodes(B_viz, pos, nodelist=drugs_viz,
                          node_color='blue', node_size=100, alpha=0.6)
    
    # 4. Muchii
    nx.draw_networkx_edges(B_viz, pos, alpha=0.2, width=0.5, edge_color='gray')
    
    # 5. Etichete (doar pentru noduri importante)
    labels = {}
    for node in disease_genes_viz[:10]:  # Primele 10 gene bolii
        labels[node] = node
    
    if top_drugs:
        for drug in top_drugs[:10]:  # Primele 10 medicamente top
            if drug in B_viz:
                labels[drug] = drug
    
    nx.draw_networkx_labels(B_viz, pos, labels, font_size=8, font_weight='bold')
    
    # Legendă
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label=f'Gene bolii ({len(disease_genes_viz)})'),
        Patch(facecolor='green', alpha=0.5, label=f'Alte gene ({len(other_genes_viz)})'),
        Patch(facecolor='blue', alpha=0.6, label=f'Medicamente ({len(drugs_viz)})')
    ]
    
    plt.legend(handles=legend_elements, loc='upper left')
    plt.title(f"Rețea Drug-Gene pentru Drug Repurposing\n({B_viz.number_of_nodes()} noduri, {B_viz.number_of_edges()} muchii)")
    plt.axis('off')
    
    # Salvează
    plt.tight_layout()
    plt.savefig(OUT_NETWORK_IMAGE, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Imagine salvată: {OUT_NETWORK_IMAGE}")
    
    # Afișează
    plt.show()


def generate_detailed_report(df_ranking: pd.DataFrame, B: nx.Graph, disease_genes: Set[str]):
    """
    Generează raport detaliat
    """
    print("\n" + "="*60)
    print("RAPORT REZULTATE DRUG REPURPOSING")
    print("="*60)
    
    if df_ranking.empty:
        print("Niciun rezultat găsit.")
        return
    
    # Statistici
    total_drugs = len(get_drug_nodes(B))
    analyzed = len(df_ranking)
    direct_target = df_ranking['has_direct'].sum()
    
    print(f"\n📊 STATISTICI:")
    print(f"   • Medicamente totale: {total_drugs}")
    print(f"   • Medicamente analizate: {analyzed}")
    print(f"   • Cu ținte directe: {direct_target}")
    print(f"   • Gene bolii în rețea: {len([g for g in disease_genes if g in B])}/{len(disease_genes)}")
    
    # Top medicamente
    print(f"\n🏆 TOP 10 MEDICAMENTE RECOMANDATE:")
    print("-"*70)
    print(f"{'Rank':<5} {'Medicament':<25} {'Dist.':<8} {'Ținte directe':<15} {'Grad':<8}")
    print("-"*70)
    
    for i, row in df_ranking.head(10).iterrows():
        name = str(row['drug'])[:24]
        dist = f"{row['closest_distance']:.1f}"
        direct = f"{row['direct_targets']}" if row['direct_targets'] > 0 else "-"
        degree = row['degree']
        
        print(f"{row['rank']:<5} {name:<25} {dist:<8} {direct:<15} {degree:<8}")
    
    # Salvează raport în fișier
    with open(OUT_REPORT, 'w') as f:
        f.write("RAPORT DRUG REPURPOSING - ANALIZĂ REȚEA\n")
        f.write("="*50 + "\n\n")
        f.write(f"Handle: {HANDLE}\n")
        f.write(f"Data: {pd.Timestamp.now()}\n\n")
        
        f.write("STATISTICI GENERALE:\n")
        f.write(f"- Medicamente totale în rețea: {total_drugs}\n")
        f.write(f"- Medicamente analizate: {analyzed}\n")
        f.write(f"- Medicamente cu ținte directe: {direct_target}\n")
        f.write(f"- Gene bolii analizate: {len(disease_genes)}\n")
        f.write(f"- Gene bolii în rețea: {len([g for g in disease_genes if g in B])}\n\n")
        
        f.write("TOP 20 MEDICAMENTE PENTRU REPURPOSING:\n")
        f.write("-"*50 + "\n")
        for i, row in df_ranking.head(20).iterrows():
            f.write(f"{row['rank']}. {row['drug']} | "
                   f"Distanță: {row['closest_distance']:.2f} | "
                   f"Ținte directe: {row['direct_targets']} | "
                   f"Grad: {row['degree']}\n")
        
        f.write("\nGENE BOLII ANALIZATE:\n")
        for gene in sorted(disease_genes):
            in_network = "✓" if gene in B else "✗"
            f.write(f"{in_network} {gene}\n")
    
    print(f"\n📄 Raport detaliat salvat: {OUT_REPORT}")


def main():
    """
    Funcția principală
    """
    print("="*60)
    print("EXERCIȚIU 9.2 - DRUG REPURPOSING ANALYSI")
    print("="*60)
    
    try:
        # 1. Creează fișier gene bolii (dacă nu există)
        disease_file = create_disease_genes_file()
        
        # 2. Încarcă genele bolii
        disease_genes = load_disease_genes(disease_file)
        
        # 3. Încarcă graful
        B = load_graph()
        
        if B.number_of_nodes() == 0:
            print("[ERROR] Graful este gol!")
            return
        
        # 4. Calculează ranking
        print("\n" + "="*60)
        print("CALCUL PROXIMITATE MEDICAMENTE")
        print("="*60)
        
        df_ranking = calculate_drug_ranking(B, disease_genes, max_drugs=2000)
        
        if df_ranking.empty:
            print("[WARNING] Nu s-au găsit rezultate!")
            print("[INFO] Sugestii:")
            print("  1. Modifică lista de gene bolii în fișierul:")
            print(f"     {disease_file}")
            print("  2. Verifică că genele există în rețea")
            return
        
        # 5. Salvează ranking
        df_ranking.to_csv(OUT_DRUG_PRIORITY, index=False)
        print(f"[SUCCESS] Ranking salvat: {OUT_DRUG_PRIORITY}")
        
        # 6. Generează raport
        generate_detailed_report(df_ranking, B, disease_genes)
        
        # 7. Creează vizualizare (folosește top 10 medicamente)
        top_10_drugs = df_ranking.head(10)['drug'].tolist()
        create_visualization(B, disease_genes, top_10_drugs)
        
        # 8. Final
        print("\n" + "="*60)
        print("✅ ANALIZĂ COMPLETĂ!")
        print("="*60)
        print(f"\n🎯 REZULTATE:")
        print(f"   • Fișier ranking: {OUT_DRUG_PRIORITY.name}")
        print(f"   • Imagine rețea: {OUT_NETWORK_IMAGE.name}")
        print(f"   • Raport detaliat: {OUT_REPORT.name}")
        
        # Sugestii pentru următorul pas
        print(f"\n💡 SUGESTII PENTRU INTERPRETARE:")
        print(f"   1. Medicamentele cu 'has_direct' = True sunt cele mai promițătoare")
        print(f"   2. Verifică medicamentele cu distanță mică (< 3)")
        print(f"   3. Consultă baze de date farmaceutice pentru validare")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 Verifică că ai fișierele necesare în:")
        print(f"   {BASE_DIR}")


if __name__ == "__main__":
    main()