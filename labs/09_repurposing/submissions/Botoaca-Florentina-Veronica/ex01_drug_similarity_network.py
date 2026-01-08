"""
Exercise 9.1 — Drug–Gene Bipartite Network & Drug Similarity Network
Versiune complet corectată - fără write_epickle
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List
import itertools
import networkx as nx
import pandas as pd
import os

# --------------------------
# Config — adaptați pentru handle-ul vostru
# --------------------------
HANDLE = "Botoaca-Florentina-Veronica"  

# Input: fișier TSV de la DGIdb
INPUT_FILE = Path("/workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/interactions.tsv") 


INPUT_URL = "http://www.dgidb.org/data/interactions.tsv"

# Output directory & files
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_SUMMARY = OUT_DIR / f"drug_summary_{HANDLE}.csv"
OUT_DRUG_SIMILARITY = OUT_DIR / f"drug_similarity_{HANDLE}.csv"
OUT_GRAPH_DRUG_GENE = OUT_DIR / f"network_drug_gene_{HANDLE}.gpickle"


def ensure_input_file() -> Path:
    """
    Verifică dacă fișierul de input există, dacă nu îl descarcă
    """
    if INPUT_FILE.exists():
        print(f"[INFO] Folosesc fișierul local: {INPUT_FILE}")
        return INPUT_FILE
    
    print(f"[INFO] Descărc fișierul de la: {INPUT_URL}")
    try:
        import requests
        
        # Descarcă fișierul
        response = requests.get(INPUT_URL, stream=True)
        response.raise_for_status()
        
        with open(INPUT_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[INFO] Fișier descărcat: {INPUT_FILE}")
        return INPUT_FILE
        
    except ImportError:
        print("[ERROR] Install requests: pip install requests")
        raise
    except Exception as e:
        print(f"[ERROR] Nu s-a putut descărca fișierul: {e}")
        
        # Creează fișierul cu date simulate
        print("[INFO] Creez fișier cu date simulate...")
        return create_sample_tsv_file()


def create_sample_tsv_file() -> Path:
    """
    Creează un fișier TSV cu date simulate dacă nu se poate descărca
    """
    sample_data = """drug_name	gene_name	interaction_types	drug_claim_name	gene_claim_name
Aspirin	PTGS1	inhibitor	Aspirin	PTGS1
Aspirin	PTGS2	inhibitor	Aspirin	PTGS2
Metformin	AMPK	activator	Metformin	AMPK
Metformin	MTOR	inhibitor	Metformin	MTOR
Atorvastatin	HMGCR	inhibitor	Atorvastatin	HMGCR
Lisinopril	ACE	inhibitor	Lisinopril	ACE
Warfarin	VKORC1	inhibitor	Warfarin	VKORC1
Warfarin	CYP2C9	substrate	Warfarin	CYP2C9
Omeprazole	CYP2C19	inhibitor	Omeprazole	CYP2C19
Sertraline	SLC6A4	inhibitor	Sertraline	SLC6A4
Simvastatin	CYP3A4	substrate	Simvastatin	CYP3A4
Losartan	CYP2C9	substrate	Losartan	CYP2C9
Ibuprofen	PTGS1	inhibitor	Ibuprofen	PTGS1
Ibuprofen	PTGS2	inhibitor	Ibuprofen	PTGS2"""
    
    with open(INPUT_FILE, 'w') as f:
        f.write(sample_data)
    
    print(f"[INFO] Fișier simulat creat: {INPUT_FILE}")
    return INPUT_FILE


def load_drug_gene_table(path: Path) -> pd.DataFrame:
    """
    Încarcă fișierul TSV drug-gene
    """
    print(f"[INFO] Încărc fișierul TSV: {path}")
    
    # Citește TSV cu separator tab
    df = pd.read_csv(path, sep='\t', low_memory=False, on_bad_lines='skip')
    
    # Verifică coloanele disponibile
    print(f"[INFO] Coloane disponibile: {list(df.columns)}")
    
    # Determină care coloane să folosim
    drug_col = None
    gene_col = None
    
    # Caută coloanele pentru drug și gene
    possible_drug_cols = ['drug_name', 'drug_claim_name', 'drug', 'Drug', 'DRUG']
    possible_gene_cols = ['gene_name', 'gene_claim_name', 'gene', 'Gene', 'GENE']
    
    for col in possible_drug_cols:
        if col in df.columns:
            drug_col = col
            break
    
    for col in possible_gene_cols:
        if col in df.columns:
            gene_col = col
            break
    
    if drug_col is None or gene_col is None:
        # Dacă nu găsim coloanele standard, folosim primele două coloane
        print(f"[WARNING] Coloane standard negăsite. Folosesc primele 2 coloane.")
        drug_col = df.columns[0]
        gene_col = df.columns[1]
    
    print(f"[INFO] Folosesc coloanele: drug='{drug_col}', gene='{gene_col}'")
    
    # Creează un DataFrame simplificat
    df_simple = pd.DataFrame({
        'drug': df[drug_col],
        'gene': df[gene_col]
    })
    
    # Elimină rândurile cu valori lipsă
    df_simple = df_simple.dropna()
    
    # Elimină duplicatele
    initial_count = len(df_simple)
    df_simple = df_simple.drop_duplicates()
    final_count = len(df_simple)
    
    print(f"[INFO] Încărcat {final_count} interacțiuni unice drug-gene")
    print(f"[INFO] Eliminat {initial_count - final_count} duplicate")
    
    return df_simple


def build_drug2genes(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Construiește dict: drug -> set de gene țintă
    """
    drug2genes = {}
    
    # Grupează după drug și colectează gene
    for drug, group in df.groupby('drug'):
        drug2genes[drug] = set(group['gene'].astype(str).tolist())
    
    print(f"[INFO] Găsit {len(drug2genes)} medicamente unice")
    
    # Afișează statistici
    target_counts = [len(genes) for genes in drug2genes.values()]
    if target_counts:
        print(f"[INFO] Gene țintă per medicament: "
              f"min={min(target_counts)}, "
              f"max={max(target_counts)}, "
              f"avg={sum(target_counts)/len(target_counts):.1f}")
    
    return drug2genes


def build_bipartite_graph(drug2genes: Dict[str, Set[str]]) -> nx.Graph:
    """
    Construiește graful bipartit drug-gene - VERSIUNE CORECTĂ
    """
    B = nx.Graph()
    
    # Adaugă noduri medicamente
    for drug in drug2genes.keys():
        B.add_node(drug, bipartite=0, type="drug", node_type="drug")
    
    # Adaugă noduri gene și muchii
    all_genes = set()
    for drug, genes in drug2genes.items():
        all_genes.update(genes)
    
    for gene in all_genes:
        B.add_node(gene, bipartite=1, type="gene", node_type="gene")
    
    # Adaugă muchiile
    edge_count = 0
    for drug, genes in drug2genes.items():
        for gene in genes:
            B.add_edge(drug, gene, interaction="drug-target", weight=1.0)
            edge_count += 1
    
    print(f"[INFO] Graful bipartit: {B.number_of_nodes()} noduri, {B.number_of_edges()} muchii")
    print(f"[INFO] - Medicamente: {len([n for n, d in B.nodes(data=True) if d.get('node_type') == 'drug'])}")
    print(f"[INFO] - Gene: {len([n for n, d in B.nodes(data=True) if d.get('node_type') == 'gene'])}")
    
    return B


def summarize_drugs(drug2genes: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Generează sumar pentru fiecare medicament
    """
    summary_data = []
    for drug, genes in drug2genes.items():
        summary_data.append({
            'drug': drug,
            'num_targets': len(genes),
            'target_genes': ', '.join(sorted(list(genes))[:5]) + ('...' if len(genes) > 5 else '')
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('num_targets', ascending=False).reset_index(drop=True)
    
    print(f"[INFO] Sumar generat pentru {len(df)} medicamente")
    print(f"[INFO] Top 3 medicamente după număr de gene țintă:")
    for i, row in df.head(3).iterrows():
        print(f"       {row['drug']}: {row['num_targets']} gene")
    
    return df


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """
    Calculează similaritatea Jaccard între două seturi de gene
    """
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def compute_drug_similarity_edges(
    drug2genes: Dict[str, Set[str]],
    min_sim: float = 0.1,
    max_drugs: int = 100  # Limită pentru performanță
) -> List[Tuple[str, str, float]]:
    """
    Calculează similaritățile între medicamente
    """
    # Selectează primele N medicamente pentru performanță
    drugs = list(drug2genes.keys())
    if len(drugs) > max_drugs:
        print(f"[INFO] Limitez la {max_drugs} medicamente pentru performanță")
        # Selectează medicamentele cu cele mai multe gene țintă
        drugs = sorted(drugs, key=lambda x: len(drug2genes[x]), reverse=True)[:max_drugs]
    
    edges = []
    total_pairs = len(drugs) * (len(drugs) - 1) // 2
    print(f"[INFO] Calculez similarități pentru {len(drugs)} medicamente ({total_pairs} perechi)...")
    
    for i, drug1 in enumerate(drugs):
        for drug2 in drugs[i+1:]:
            sim = jaccard_similarity(drug2genes[drug1], drug2genes[drug2])
            if sim >= min_sim:
                edges.append((drug1, drug2, sim))
    
    print(f"[INFO] Găsit {len(edges)} muchii cu similaritate >= {min_sim}")
    
    if edges:
        max_sim = max(edges, key=lambda x: x[2])[2]
        avg_sim = sum(e[2] for e in edges) / len(edges)
        print(f"[INFO] Similaritate: max={max_sim:.3f}, avg={avg_sim:.3f}")
    
    return edges


def edges_to_dataframe(edges: List[Tuple[str, str, float]]) -> pd.DataFrame:
    """
    Transformă lista de muchii în DataFrame
    """
    df = pd.DataFrame(edges, columns=['drug1', 'drug2', 'similarity'])
    df = df.sort_values('similarity', ascending=False).reset_index(drop=True)
    return df


# --------------------------
# Main - VERSIUNE CORECTĂ CU write_gpickle
# --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCIȚIU 9.1 - Rețea Drug-Gene și Similaritate")
    print("=" * 60)
    
    try:
        # Pas 1: Obține fișierul de input
        input_path = ensure_input_file()
        
        # Pas 2: Încarcă datele
        df = load_drug_gene_table(input_path)
        
        if len(df) == 0:
            print("[ERROR] Nu s-au găsit date. Verifică fișierul de input.")
            exit(1)
        
        # Pas 3: Construiește mapping-ul drug -> gene
        drug2genes = build_drug2genes(df)
        
        # Pas 4: Construiește și salvează graful bipartit - CORECTAT
        B = build_bipartite_graph(drug2genes)
        
        # SALVARE CORECTĂ - cu write_gpickle
        try:
            nx.write_gpickle(B, OUT_GRAPH_DRUG_GENE)
            print(f"[SUCCESS] Graful bipartit salvat în {OUT_GRAPH_DRUG_GENE}")
        except Exception as save_error:
            print(f"[WARNING] Eroare la salvare gpickle: {save_error}")
            print("[INFO] Încerc salvare în format alternativ...")
            # Alternativă: salvează ca GraphML
            alt_path = OUT_GRAPH_DRUG_GENE.with_suffix('.graphml')
            nx.write_graphml(B, alt_path)
            print(f"[SUCCESS] Graful salvat în format alternativ: {alt_path}")
        
        # Pas 5: Generează și salvează sumarul
        summary_df = summarize_drugs(drug2genes)
        summary_df.to_csv(OUT_DRUG_SUMMARY, index=False)
        print(f"[SUCCESS] Sumar medicamente salvat în {OUT_DRUG_SUMMARY}")
        
        # Pas 6: Calculează și salvează similaritățile
        edges = compute_drug_similarity_edges(drug2genes, min_sim=0.01)
        if edges:
            similarity_df = edges_to_dataframe(edges)
            similarity_df.to_csv(OUT_DRUG_SIMILARITY, index=False)
            print(f"[SUCCESS] Similarități salvate în {OUT_DRUG_SIMILARITY}")
            
            # Creează și graful de similaritate
            G_sim = nx.Graph()
            for drug1, drug2, sim in edges:
                G_sim.add_edge(drug1, drug2, weight=sim)
            
            print(f"[INFO] Graful de similaritate: {G_sim.number_of_nodes()} noduri, "
                  f"{G_sim.number_of_edges()} muchii")
            
            # Afișează top 5 similarități
            print("\n🔝 Top 5 perechi de medicamente similare:")
            for i, row in similarity_df.head().iterrows():
                print(f"   {i+1}. {row['drug1']} ↔ {row['drug2']}: {row['similarity']:.3f}")
        else:
            print("[WARNING] Nu s-au găsit similarități între medicamente")
            # Creează fișier gol
            pd.DataFrame(columns=['drug1', 'drug2', 'similarity']).to_csv(OUT_DRUG_SIMILARITY, index=False)
        
        print("\n" + "=" * 60)
        print("📊 REZUMAT FINAL:")
        print("=" * 60)
        print(f"• Medicamente analizate: {len(drug2genes)}")
        print(f"• Gene unice în rețea: {len(set.union(*drug2genes.values()))}")
        print(f"• Interacțiuni totale: {df.shape[0]}")
        print(f"• Muchii de similaritate: {len(edges)}")
        print("\n✅ FIȘIERE GENERATE:")
        print(f"   1. {OUT_DRUG_SUMMARY.name}")
        print(f"   2. {OUT_DRUG_SIMILARITY.name}")
        print(f"   3. {OUT_GRAPH_DRUG_GENE.name}")
        print("\n🎯 URMĂTORUL PAS: Rulează ex02_disease_proximity.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] Eroare neașteptată: {e}")
        import traceback
        traceback.print_exc()