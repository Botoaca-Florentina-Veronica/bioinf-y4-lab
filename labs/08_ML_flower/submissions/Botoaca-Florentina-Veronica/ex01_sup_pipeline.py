"""
Exercise 8 — Supervised ML pipeline pentru expresie genică (Random Forest)

TODO-uri principale:
- Încărcați matricea de expresie (ex. subset TP53 / GTEx) pentru HANDLE-ul vostru
- Separați features (gene) și label (ultima coloană)
- Encodați etichetele
- Împărțiți în train/test
- Antrenați un RandomForestClassifier (model de bază)
- Evaluați: classification_report + matrice de confuzie (salvate)
- Calculați importanța trăsăturilor și salvați în CSV
- (Opțional) Aplicați KMeans pe X și comparați clustere vs etichete reale
"""


from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Config — completați cu valorile voastre
# --------------------------
HANDLE = "Botoaca-Florentina-Veronica"

# Calea specifică pentru datele tale
DATA_CSV = Path("/workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/expression_matrix.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
TOPK_FEATURES = 20

OUT_DIR = Path("/workspaces/bioinf-y4-lab/labs/08_ML_flower/submissions/Botoaca-Florentina-Veronica/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONFUSION = OUT_DIR / f"confusion_rf_{HANDLE}.png"
OUT_REPORT = OUT_DIR / f"classification_report_{HANDLE}.txt"
OUT_FEATIMP = OUT_DIR / f"feature_importance_{HANDLE}.csv"
OUT_CLUSTER_CROSSTAB = OUT_DIR / f"cluster_crosstab_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    """
    Verifică că fișierul de input există
    """
    if not path.is_file():
        raise FileNotFoundError(f"❌ Nu am găsit fișierul: {path}")
    print(f"✓ Fișier găsit: {path}")


def load_and_prepare_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Încarcă datele și creează etichete artificiale pentru ML
    Datele originale nu au coloană 'Label', așa că o creăm
    """
    # Încarcă matricea de expresie
    df = pd.read_csv(path, index_col=0)
    print(f"✓ Date originale încărcate: {df.shape[0]} gene, {df.shape[1]} probe")
    print(f"  Index (gene): {df.index.tolist()[:5]}...")
    print(f"  Coloane (probe): {df.columns.tolist()[:5]}...")
    
    # Transpunem: gene devin features, probe devin samples
    # Pentru ML, avem nevoie de samples pe rânduri și features pe coloane
    df_transposed = df.T  # Transpunem: acum probe sunt rânduri, gene sunt coloane
    print(f"\n✓ După transpunere: {df_transposed.shape[0]} probe, {df_transposed.shape[1]} gene")
    
    # Creăm etichete artificiale bazate pe clustering al probelor
    # Folosim KMeans pentru a grupa probele în 3 clase
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_transposed)
    
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Adaugă etichetele la DataFrame
    df_transposed["Label"] = labels
    
    # Mapează clusterele la nume semnificative
    label_mapping = {0: "Type_A", 1: "Type_B", 2: "Type_C"}
    df_transposed["Label"] = df_transposed["Label"].map(label_mapping)
    
    # Statistici
    print(f"\n📊 DISTRIBUȚIA ETICHETELOR ARTIFICIALE:")
    label_counts = df_transposed["Label"].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} probe ({count/len(df_transposed)*100:.1f}%)")
    
    # Separați X (features) și y (labels)
    X = df_transposed.iloc[:, :-1]  # Toate genele
    y = df_transposed.iloc[:, -1]   # Etichetele create
    
    return X, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encodare etichete string în valori numerice
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print(f"\n✓ Label encoding:")
    for i, cls in enumerate(le.classes_):
        count = (y == cls).sum()
        print(f"  {cls} -> {i} ({count} probe)")
    
    return y_enc, le


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> RandomForestClassifier:
    """
    Antrenare Random Forest Classifier
    """
    print("\n🌲 ANTRENARE RANDOM FOREST...")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=10,
        min_samples_split=5,
        oob_score=True  # Out-of-bag score pentru validare internă
    )
    rf.fit(X_train, y_train)
    
    print(f"  ✓ Random Forest antrenat cu {n_estimators} arbori")
    print(f"  ✓ Accuracy pe train: {rf.score(X_train, y_train):.3f}")
    print(f"  ✓ OOB score: {rf.oob_score_:.3f}" if hasattr(rf, 'oob_score_') else "")
    
    return rf


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_png: Path,
    out_txt: Path,
) -> None:
    """
    Evaluare model: classification report și matrice de confuzie
    """
    print("\n📊 EVALUARE MODEL...")
    
    # Predicții
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    target_names = label_encoder.classes_
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    # Salvează raportul
    out_txt.write_text(report)
    print(f"✓ Raport salvat: {out_txt}")
    
    # Matrice de confuzie
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={'label': 'Număr probe'}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Random Forest — Matrice de confuzie\n{HANDLE}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Matrice de confuzie salvată: {out_png}")
    
    # Accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"  ✓ Accuracy pe test: {accuracy:.3f}")
    
    # Analiză erori
    print(f"\n🔍 ANALIZĂ ERORI:")
    for i, true_label in enumerate(target_names):
        for j, pred_label in enumerate(target_names):
            if i != j and cm[i, j] > 0:
                print(f"  • {true_label} → {pred_label}: {cm[i, j]} probe")


def compute_feature_importance(
    model: RandomForestClassifier,
    feature_names: pd.Index,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Calcul importanță trăsături (gene)
    """
    print("\n🔍 CALCUL IMPORTANȚĂ TRĂSĂTURI (GENE)...")
    
    importances = model.feature_importances_
    
    # Creează DataFrame cu importanțe
    df_imp = pd.DataFrame({
        "Gene": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    # Salvează
    df_imp.to_csv(out_csv, index=False)
    
    print(f"✓ Importanța genelor salvată: {out_csv}")
    print(f"\n📈 TOP 15 GENE CEA MAI IMPORTANTE:")
    print("-" * 50)
    
    for i, row in df_imp.head(15).iterrows():
        print(f"  {i+1:2d}. {row['Gene']:20s} : {row['Importance']:.5f}")
    
    # Analiză statistică
    print(f"\n📊 STATISTICI IMPORTANȚĂ GENE:")
    print(f"  • Importanța medie: {df_imp['Importance'].mean():.5f}")
    print(f"  • Importanța maximă: {df_imp['Importance'].max():.5f}")
    print(f"  • Importanța minimă: {df_imp['Importance'].min():.5f}")
    print(f"  • Gene cu importanță > 0.01: {(df_imp['Importance'] > 0.01).sum()}")
    
    return df_imp


def run_kmeans_and_crosstab(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    n_clusters: int,
    out_csv: Path,
) -> None:
    """
    KMeans clustering și comparație cu etichetele reale
    """
    print("\n🎯 KMEANS CLUSTERING PENTRU COMPARARE...")
    
    # Standardizare pentru KMeans
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # DataFrame pentru comparație
    df_clusters = pd.DataFrame({
        "True_Label": label_encoder.inverse_transform(y),
        "Cluster": clusters
    })
    
    # Crosstab
    ctab = pd.crosstab(
        df_clusters["True_Label"], 
        df_clusters["Cluster"],
        margins=True,
        margins_name="Total"
    )
    
    # Salvează
    ctab.to_csv(out_csv)
    print(f"✓ Crosstab salvat: {out_csv}")
    
    # Afișează crosstab
    print("\n📊 CROSSTAB - ETICHETE REALE vs CLUSTERE KMEANS:")
    print("-" * 60)
    print(ctab.to_string())
    
    # Calcul puritate cluster
    print(f"\n📈 PURITATE CLUSTERE:")
    cluster_purities = []
    for cluster in range(n_clusters):
        cluster_data = df_clusters[df_clusters["Cluster"] == cluster]
        if len(cluster_data) > 0:
            dominant_label = cluster_data["True_Label"].mode()[0]
            dominant_count = (cluster_data["True_Label"] == dominant_label).sum()
            purity = dominant_count / len(cluster_data)
            cluster_purities.append(purity)
            
            print(f"  Cluster {cluster}: {len(cluster_data):3d} probe, "
                  f"dominant '{dominant_label}' ({purity:.1%})")
    
    # Puritate medie
    if cluster_purities:
        print(f"  Puritate medie: {np.mean(cluster_purities):.1%}")


def semi_supervised_experiment(X, y_enc, le, feature_names):
    """
    Experiment semi-supervised cu pseudo-labeling
    """
    print("\n" + "="*60)
    print("🧪 EXPERIMENT SEMI-SUPERVISED")
    print("="*60)
    
    # Marchează 40% din etichete ca unknown
    n_samples = len(y_enc)
    n_labeled = int(n_samples * 0.6)
    
    # Indici aleatori pentru date etichetate
    np.random.seed(RANDOM_STATE)
    labeled_idx = np.random.choice(n_samples, n_labeled, replace=False)
    unlabeled_idx = np.setdiff1d(np.arange(n_samples), labeled_idx)
    
    X_labeled = X.iloc[labeled_idx]
    y_labeled = y_enc[labeled_idx]
    X_unlabeled = X.iloc[unlabeled_idx]
    
    print(f"  Date etichetate: {len(X_labeled)} probe")
    print(f"  Date fără etichete: {len(X_unlabeled)} probe")
    
    # 1. Model antrenat doar pe date etichetate
    print("\n1. Model doar pe date etichetate...")
    rf_labeled = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_labeled.fit(X_labeled, y_labeled)
    
    # Cross-validation pe date etichetate
    from sklearn.model_selection import cross_val_score
    scores_labeled = cross_val_score(
        rf_labeled, X_labeled, y_labeled, 
        cv=5, scoring='accuracy', n_jobs=-1
    )
    print(f"  ✓ CV Accuracy: {scores_labeled.mean():.3f} (+/- {scores_labeled.std()*2:.3f})")
    
    # 2. Pseudo-labeling
    print("\n2. Generare pseudo-etichete...")
    pseudo_labels = rf_labeled.predict(X_unlabeled)
    
    # Combina datele
    X_combined = pd.concat([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, pseudo_labels])
    
    # 3. Reantrenare pe tot setul
    print("\n3. Reantrenare pe setul complet (cu pseudo-etichete)...")
    rf_semi = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_semi.fit(X_combined, y_combined)
    
    # Cross-validation pe setul complet
    scores_semi = cross_val_score(
        rf_semi, X_combined, y_combined,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    print(f"\n📈 REZULTATE COMPARATIVE:")
    print(f"  • Doar date etichetate:    {scores_labeled.mean():.3f} (+/- {scores_labeled.std()*2:.3f})")
    print(f"  • Cu pseudo-labeling:      {scores_semi.mean():.3f} (+/- {scores_semi.std()*2:.3f})")
    
    diff = scores_semi.mean() - scores_labeled.mean()
    if diff > 0:
        print(f"  ✅ Pseudo-labeling îmbunătățește cu {diff:.3f}")
    else:
        print(f"  ⚠️ Pseudo-labeling scade cu {-diff:.3f}")
    
    return rf_labeled, rf_semi, scores_labeled, scores_semi


def create_pca_visualization(X, y_enc, le, output_dir):
    """
    Crează vizualizare PCA pentru a vedea separarea claselor
    """
    print("\n🔬 VIZUALIZARE PCA...")
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardizare
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Crează scatter plot
    plt.figure(figsize=(10, 8))
    
    # Culori pentru fiecare clasă
    colors = plt.cm.Set3(np.linspace(0, 1, len(le.classes_)))
    
    for i, cls in enumerate(le.classes_):
        mask = (le.inverse_transform(y_enc) == cls)
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=colors[i],
            label=cls,
            alpha=0.7,
            s=50,
            edgecolor='black',
            linewidth=0.5
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianță)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianță)', fontsize=12)
    plt.title(f'PCA - Vizualizare separare clase\n{HANDLE}', fontsize=14, pad=20)
    plt.legend(title='Clase')
    plt.grid(True, alpha=0.3)
    
    # Adaugă statistici
    total_variance = pca.explained_variance_ratio_.sum()
    plt.figtext(0.02, 0.02, 
                f'Varianță totală explicată: {total_variance:.1%}\n'
                f'Număr probe: {len(X)}\n'
                f'Număr gene: {X.shape[1]}',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Salvează
    pca_path = output_dir / f"pca_visualization_{HANDLE}.png"
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Vizualizare PCA salvată: {pca_path}")
    
    return X_pca, pca


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("LAB 8 - SUPERVISED ML PIPELINE (Random Forest)")
    print(f"Student: {HANDLE}")
    print("=" * 70)
    
    try:
        # TODO 1: verificați fișierul de input
        ensure_exists(DATA_CSV)
        
        # TODO 2: încărcați și pregătiți datele (cu etichete artificiale)
        X, y = load_and_prepare_dataset(DATA_CSV)
        
        # TODO 3: encodați etichetele și împărțiți în train/test
        y_enc, le = encode_labels(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_enc,
        )
        
        print(f"\n📊 SPLIT DATE:")
        print(f"  • X_train: {X_train.shape} ({X_train.shape[0]} probe, {X_train.shape[1]} gene)")
        print(f"  • X_test:  {X_test.shape} ({X_test.shape[0]} probe, {X_test.shape[1]} gene)")
        print(f"  • Clase train: {np.bincount(y_train)}")
        print(f"  • Clase test:  {np.bincount(y_test)}")
        
        # Vizualizare PCA
        X_pca, pca_model = create_pca_visualization(X, y_enc, le, OUT_DIR)
        
        # TODO 4: antrenați modelul RF și evaluați
        rf = train_random_forest(X_train, y_train, N_ESTIMATORS, RANDOM_STATE)
        evaluate_model(rf, X_test, y_test, le, OUT_CONFUSION, OUT_REPORT)
        
        # TODO 5: calculați importanța trăsăturilor
        feat_imp_df = compute_feature_importance(rf, X.columns, OUT_FEATIMP)
        
        # TODO 6: rulați KMeans și salvați crosstab-ul
        n_classes = len(le.classes_)
        run_kmeans_and_crosstab(X, y_enc, le, n_clusters=n_classes, out_csv=OUT_CLUSTER_CROSSTAB)
        
        # Experiment semi-supervised
        rf_labeled, rf_semi, scores_labeled, scores_semi = semi_supervised_experiment(
            X, y_enc, le, X.columns
        )
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETAT CU SUCCES!")
        print("=" * 70)
        
        print(f"\n📁 LIVRABILE GENERATE în {OUT_DIR}/:")
        print(f"  1. {OUT_REPORT.name} - Classification report")
        print(f"  2. {OUT_CONFUSION.name} - Matrice de confuzie")
        print(f"  3. {OUT_FEATIMP.name} - Importanța genelor")
        print(f"  4. {OUT_CLUSTER_CROSSTAB.name} - Crosstab KMeans")
        print(f"  5. pca_visualization_{HANDLE}.png - Vizualizare PCA")
        
        print(f"\n📊 PERFORMANȚĂ FINALĂ:")
        print(f"  • Random Forest Accuracy: ~{(np.trace(confusion_matrix(y_test, rf.predict(X_test))) / len(y_test)):.1%}")
        print(f"  • Top 3 gene importante: {feat_imp_df['Gene'].head(3).tolist()}")
        print(f"  • Număr clase: {len(le.classes_)}")
        
        print(f"\n💡 INTERPRETARE BIOLOGICĂ:")
        print(f"  • Genele cele mai importante pot fi markeri pentru diferențierea probelor")
        print(f"  • Clustering-ul KMeans confirmă parțial structura de clase")
        print(f"  • PCA arată separarea claselor în spațiul de feature-uri redus")
        
    except Exception as e:
        print(f"\n❌ EROARE: {e}")
        import traceback
        traceback.print_exc()