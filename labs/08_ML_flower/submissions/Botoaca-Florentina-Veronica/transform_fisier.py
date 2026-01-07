# transform_fisier.py
import pandas as pd
import numpy as np
from pathlib import Path

def transforma_fisier_etichete():
    """Transformă fișierul de expresie genică în format pentru ML"""
    
    # Calea către fișierul tău original
    fisier_original = "/workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/expression_matrix.csv"
    
    # Încarcă fișierul original
    print(f"📂 Încarc {fisier_original}...")
    
    try:
        # Încearcă să citești ca CSV normal
        df = pd.read_csv(fisier_original)
    except:
        # Dacă nu merge, încearcă cu index ca primă coloană
        df = pd.read_csv(fisier_original, index_col=0)
    
    print(f"📊 Structura originală: {df.shape}")
    print(f"   Rânduri: {df.shape[0]} (probabil gene)")
    print(f"   Coloane: {df.shape[1]} (probabil eșantioane)")
    print(f"\n🔍 Primele 3 rânduri:")
    print(df.head(3))
    print(f"\n🔍 Primele 3 coloane:")
    print(df.iloc[:, :3].head())
    
    # Verifică dacă e transpus sau nu
    # Dacă prima coloană are nume de gene, probabil e matrice transpusă
    prima_coloana = df.columns[0]
    if prima_coloana.startswith('Sample_') or prima_coloana.startswith('sample'):
        print("\n✅ Fișierul pare să aibă eșantioane pe rânduri (bun format)")
        df_transform = df.copy()
    else:
        print("\n🔄 Fișierul pare să aibă gene pe rânduri. Transpun...")
        # Transpunem: gene devin coloane, eșantioane devin rânduri
        df_transform = df.T
        print(f"   După transpunere: {df_transform.shape}")
    
    # Acum avem eșantioane pe rânduri și gene pe coloane
    # Trebuie să adăugăm etichete
    
    # OPȚIUNI pentru etichete:
    print("\n🎯 Alege cum să generezi etichetele:")
    print("   1. Etichete dummy pentru testare (Normal/Tumor)")
    print("   2. Etichete bazate pe clusterizare expresiei")
    print("   3. Etichete din alt fișier (dacă ai)")
    
    optiune = input("\nIntrodu opțiunea (1, 2 sau 3): ")
    
    if optiune == "1":
        # Etichete dummy - 2 clase
        n_esantioane = df_transform.shape[0]
        # Generează etichete echilibrate
        labels = ['Normal'] * (n_esantioane // 2) + ['Tumor'] * (n_esantioane - n_esantioane // 2)
        np.random.seed(42)
        np.random.shuffle(labels)
        df_transform['Diagnosis'] = labels
        
    elif optiune == "2":
        # Clusterizare pentru a genera etichete "realiste"
        from sklearn.cluster import KMeans
        
        n_esantioane = df_transform.shape[0]
        n_clusters = min(3, n_esantioane // 10)  # Maxim 3 clustere
        
        print(f"\n🔬 Clusterizare KMeans cu {n_clusters} clustere...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Standardizez pentru clusterizare
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_transform)
        
        clustere = kmeans.fit_predict(X_scaled)
        
        # Transformă clustere în etichete
        tipuri_cancer = ['Normal', 'Tumor_A', 'Tumor_B', 'Metastatic']
        labels = [tipuri_cancer[c % len(tipuri_cancer)] for c in clustere]
        df_transform['Cancer_Subtype'] = labels
        
    elif optiune == "3":
        # Încarcă etichete dintr-un alt fișier
        fisier_etichete = input("Introdu calea către fișierul cu etichete: ")
        try:
            etichete_df = pd.read_csv(fisier_etichete)
            # Asigură-te că etichetele au aceeași ordine ca eșantioanele
            df_transform['Label'] = etichete_df.iloc[:, 0].values
        except:
            print("❌ Nu pot încărca fișierul cu etichete. Folosesc etichete dummy.")
            n_esantioane = df_transform.shape[0]
            labels = ['Class_' + str(i % 3) for i in range(n_esantioane)]
            df_transform['Class'] = labels
    else:
        print("⚠️  Opțiune invalidă. Folosesc etichete dummy.")
        n_esantioane = df_transform.shape[0]
        labels = ['Normal'] * (n_esantioane // 2) + ['Tumor'] * (n_esantioane - n_esantioane // 2)
        np.random.seed(42)
        np.random.shuffle(labels)
        df_transform['Diagnosis'] = labels
    
    # Verifică etichetele
    coloana_eticheta = df_transform.columns[-1]
    print(f"\n🏷️  Coloana de etichete adăugată: '{coloana_eticheta}'")
    print(f"📈 Distribuție etichete:")
    distributie = df_transform[coloana_eticheta].value_counts()
    for eticheta, count in distributie.items():
        print(f"   {eticheta}: {count} eșantioane ({count/len(df_transform)*100:.1f}%)")
    
    # Salvează fișierul transformat
    fisier_nou = "expression_with_labels.csv"
    df_transform.to_csv(fisier_nou, index=False)
    
    print(f"\n✅ Fișier transformat salvat ca: {fisier_nou}")
    print(f"   Eșantioane: {df_transform.shape[0]}")
    print(f"   Features (gene + eticheta): {df_transform.shape[1]}")
    print(f"   Structură perfectă pentru exercițiul 2!")
    
    # Salvează și o versiune de test cu mai puține eșantioane
    fisier_test = "expression_small_test.csv"
    df_transform.sample(min(100, len(df_transform)), random_state=42).to_csv(fisier_test, index=False)
    print(f"📦 Versiune test (100 eșantioane) salvată ca: {fisier_test}")
    
    return fisier_nou

if __name__ == "__main__":
    fisier_transform = transforma_fisier_etichete()
    
    print("\n" + "="*60)
    print("🎉 FIȘIER TRANSFORMAT CU SUCCES!")
    print("="*60)
    print(f"\nAcum poți modifica codul ex02_logreg_vs_rf.py:")
    print(f"""
# Înlocuiește linia DATA_CSV cu:
DATA_CSV = Path("{fisier_transform}")

# Sau pentru versiunea mică de test:
# DATA_CSV = Path("expression_small_test.csv")
    """)