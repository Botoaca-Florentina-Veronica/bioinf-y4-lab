======================================================================
LAB 8 - SUPERVISED ML PIPELINE (Random Forest)
Student: Botoaca-Florentina-Veronica
======================================================================
✓ Fișier găsit: /workspaces/bioinf-y4-lab/labs/01_intro&databases/data/work/Botoaca-Florentina-Veronica/lab01/expression_matrix.csv
✓ Date originale încărcate: 200 gene, 50 probe
  Index (gene): ['RPL11', 'RPL34', 'RPL35A', 'RPS23', 'RPS3A']...
  Coloane (probe): ['Sample_000', 'Sample_001', 'Sample_002', 'Sample_003', 'Sample_004']...

✓ După transpunere: 50 probe, 200 gene

📊 DISTRIBUȚIA ETICHETELOR ARTIFICIALE:
  Type_B: 21 probe (42.0%)
  Type_C: 16 probe (32.0%)
  Type_A: 13 probe (26.0%)

✓ Label encoding:
  Type_A -> 0 (13 probe)
  Type_B -> 1 (21 probe)
  Type_C -> 2 (16 probe)

📊 SPLIT DATE:
  • X_train: (40, 200) (40 probe, 200 gene)
  • X_test:  (10, 200) (10 probe, 200 gene)
  • Clase train: [10 17 13]
  • Clase test:  [3 4 3]

🔬 VIZUALIZARE PCA...
✓ Vizualizare PCA salvată: /workspaces/bioinf-y4-lab/labs/08_ML_flower/submissions/Botoaca-Florentina-Veronica/outputs/pca_visualization_Botoaca-Florentina-Veronica.png

🌲 ANTRENARE RANDOM FOREST...
  ✓ Random Forest antrenat cu 200 arbori
  ✓ Accuracy pe train: 1.000
  ✓ OOB score: 0.900

📊 EVALUARE MODEL...

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

      Type_A      1.000     1.000     1.000         3
      Type_B      1.000     1.000     1.000         4
      Type_C      1.000     1.000     1.000         3

    accuracy                          1.000        10
   macro avg      1.000     1.000     1.000        10
weighted avg      1.000     1.000     1.000        10

✓ Raport salvat: /workspaces/bioinf-y4-lab/labs/08_ML_flower/submissions/Botoaca-Florentina-Veronica/outputs/classification_report_Botoaca-Florentina-Veronica.txt
✓ Matrice de confuzie salvată: /workspaces/bioinf-y4-lab/labs/08_ML_flower/submissions/Botoaca-Florentina-Veronica/outputs/confusion_rf_Botoaca-Florentina-Veronica.png
  ✓ Accuracy pe test: 1.000

🔍 ANALIZĂ ERORI:

🔍 CALCUL IMPORTANȚĂ TRĂSĂTURI (GENE)...
✓ Importanța genelor salvată: /workspaces/bioinf-y4-lab/labs/08_ML_flower/submissions/Botoaca-Florentina-Veronica/outputs/feature_importance_Botoaca-Florentina-Veronica.csv

📈 TOP 15 GENE CEA MAI IMPORTANTE:
--------------------------------------------------
  130. ACTB_1               : 0.03495
  73. RPL11_1              : 0.02534
  122. EGFR_1               : 0.02529
  99. PCK1_1               : 0.02358
  107. ESR1_1               : 0.02105
  88. C4BPA_1              : 0.01940
  96. IL6_1                : 0.01917
  22. HP                   : 0.01866
   4. RPS23                : 0.01866
   1. RPL11                : 0.01804
  152. RPL13A_2             : 0.01594
   7. RPLP0                : 0.01514
  18. CYP8B1               : 0.01508
  28. PON1                 : 0.01504
  13. AHSG                 : 0.01431

📊 STATISTICI IMPORTANȚĂ GENE:
  • Importanța medie: 0.00500
  • Importanța maximă: 0.03495
  • Importanța minimă: 0.00000
  • Gene cu importanță > 0.01: 31

🎯 KMEANS CLUSTERING PENTRU COMPARARE...
✓ Crosstab salvat: /workspaces/bioinf-y4-lab/labs/08_ML_flower/submissions/Botoaca-Florentina-Veronica/outputs/cluster_crosstab_Botoaca-Florentina-Veronica.csv

📊 CROSSTAB - ETICHETE REALE vs CLUSTERE KMEANS:
------------------------------------------------------------
Cluster      0   1   2  Total
True_Label                   
Type_A      13   0   0     13
Type_B       0  21   0     21
Type_C       0   0  16     16
Total       13  21  16     50

📈 PURITATE CLUSTERE:
  Cluster 0:  13 probe, dominant 'Type_A' (100.0%)
  Cluster 1:  21 probe, dominant 'Type_B' (100.0%)
  Cluster 2:  16 probe, dominant 'Type_C' (100.0%)
  Puritate medie: 100.0%

============================================================
🧪 EXPERIMENT SEMI-SUPERVISED
============================================================
  Date etichetate: 30 probe
  Date fără etichete: 20 probe

1. Model doar pe date etichetate...
  ✓ CV Accuracy: 0.833 (+/- 0.211)

2. Generare pseudo-etichete...

3. Reantrenare pe setul complet (cu pseudo-etichete)...

📈 REZULTATE COMPARATIVE:
  • Doar date etichetate:    0.833 (+/- 0.211)
  • Cu pseudo-labeling:      0.880 (+/- 0.080)
  ✅ Pseudo-labeling îmbunătățește cu 0.047

======================================================================
✅ PIPELINE COMPLETAT CU SUCCES!
======================================================================

📁 LIVRABILE GENERATE în /workspaces/bioinf-y4-lab/labs/08_ML_flower/submissions/Botoaca-Florentina-Veronica/outputs/:
  1. classification_report_Botoaca-Florentina-Veronica.txt - Classification report
  2. confusion_rf_Botoaca-Florentina-Veronica.png - Matrice de confuzie
  3. feature_importance_Botoaca-Florentina-Veronica.csv - Importanța genelor
  4. cluster_crosstab_Botoaca-Florentina-Veronica.csv - Crosstab KMeans
  5. pca_visualization_Botoaca-Florentina-Veronica.png - Vizualizare PCA

📊 PERFORMANȚĂ FINALĂ:
  • Random Forest Accuracy: ~100.0%
  • Top 3 gene importante: ['ACTB_1', 'RPL11_1', 'EGFR_1']
  • Număr clase: 3

💡 INTERPRETARE BIOLOGICĂ:
  • Genele cele mai importante pot fi markeri pentru diferențierea probelor
  • Clustering-ul KMeans confirmă parțial structura de clase
  • PCA arată separarea claselor în spațiul de feature-uri redus