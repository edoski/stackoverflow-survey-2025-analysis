- Task: Classificazione (AIAccBin) + Regressione lineare semplice (WorkExp → CompTotal)

# Slide 2 — Dataset (Panoramica)
- Origine: sondaggio multi‑paese sviluppatori (Kaggle, CSV in root).
- Target classificazione: `AIAccBin` (1 = Highly/Somewhat trust; 0 = Highly/Somewhat distrust).
- Regressione: `WorkExp` (anni) → `CompTotal` (EUR, subset `Currency`=EUR*).
- Feature chiave: numeriche (WorkExp, YearsCode, AgeMapped, EdLevelOrd) e categoriali (Employment, IndustryTop, RemoteWork, OrgSize, ICorPM, DevTypeTop, AI usage).

# Slide 3 — Feature (Input Numeriche)
- `WorkExp` (anni di esperienza lavorativa complessiva)
- `YearsCode` (anni di programmazione)
- (Derivate usate come numeriche nella pipeline: vedi slide successive)

# Slide 4 — Feature (Input Categoriali)
- `Employment` (tipologia impiego),`IndustryTop` (Top‑N + Other)
- `RemoteWork`, `OrgSize`, `ICorPM`, `DevTypeTop` (Top‑N + Other)
- AI usage: `AISelect`, `AIAgents`, `AIModelsChoice`

# Slide 5 — Feature (Derivate)
- `AgeMapped` (mappatura ordinale età)
- `EdLevelOrd` (mappatura ordinale titolo di studio)
- `IndustryTop`, `DevTypeTop` (riduzione cardinalità per one‑hot stabile)

# Slide 6 — Feature (Target)
- Classificazione: `AIAccBin` (1=trust, 0=distrust)
- Regressione: `CompTotal (EUR)` con predittore `WorkExp`

# Slide 7 — preprocessing.py (EUR)
- `preprocess_eur(df)`:
  - Filtro `Currency` → "EUR*"; colonne: `WorkExp`, `CompTotal`, `Country`, `Employment`, `EdLevel`, `Industry`.
  - Numeriche coerced + plausibilità (0<WorkExp≤60, CompTotal>0).
  - IQR trim su `CompTotal`; dedup.

# Slide 8 — preprocessing.py (Classificazione)
- `preprocess_aiaccbin(df)`:
  - Filtra AIAcc (4 etichette), crea `AIAccBin` (1 trust, 0 distrust).
  - Bounds su `WorkExp`, `YearsCode`; mapping ordinali `AgeMapped`, `EdLevelOrd`.
  - Cardinalità: `IndustryTop`, `DevTypeTop` (Top‑N + Other); dedup.

# Slide 9 — EDA (EUR) — Boxplot + Istogrammi
![box](plots_eda/eur/box_CompTotal.png)
![hist1](plots_eda/eur/hist_WorkExp.png)
![hist2](plots_eda/eur/hist_CompTotal.png)
- Salari destrorsi; WorkExp skew; giustifica l’uso di misure robuste e IQR trim.

# Slide 10 — EDA (EUR) — Scatter + Correlazione
![scatter](plots_eda/eur/scatter_WorkExp_CompTotal.png)
![corr](plots_eda/eur/corr_numeric.png)
- Trend positivo debole con alta dispersione; correlazione moderata.

# Slide 11 — EDA (Class.) — Bilanciamento + Istogrammi
![balance](plots_eda/aiaccbin/distribuzione_AIAccBin.png)
![h_w](plots_eda/aiaccbin/hist_WorkExp.png)
![h_y](plots_eda/aiaccbin/hist_YearsCode.png)
- Sbilanciamento moderato; WorkExp/YearsCode destrorsi.

# Slide 12 — EDA (Class.) — Box + Employment vs Target
![box_target](plots_eda/aiaccbin/box_WorkExp_by_AIAccBin.png)
![stacked](plots_eda/aiaccbin/stacked_Employment_AIAccBin.png)
- Employment porta segnale; WorkExp da solo separa poco.

# Slide 13 — EDA (Class.) — Correlazioni
![corrC](plots_eda/aiaccbin/corr_numeric.png)
- YearsCode correlato a WorkExp e AgeMapped; EdLevelOrd più debole.

# Slide 14 — classification.py (Flusso)
- Split 70% Train, 15% Validation, 15% Test (stratificato).
- Preprocess (Pipeline): StandardScaler su numeriche + One‑Hot su categoriali (ColumnTransformer).
- Baseline su Validation (accuracy): Logistic, SVM(linear/poly/rbf).
- Hyperparameter Tuning su Train (GridSearchCV):
  - Linear: C ∈ {0.1, 1.0, 10.0}
  - Poly: C ∈ {0.1, 1.0, 10.0}, degree ∈ {2, 3, 4}, gamma='scale'
  - RBF: C ∈ {0.1, 1.0, 10.0}, gamma='scale'
- Selezione su Validation → Refit su Train+Val → Test una volta.
- Studio statistico: CV k‑fold su Train+Val (k=10).

# Slide 15 — classification.py (Baseline & Tuning)
- Baseline su Val (accuracy): stampa per Logistic, SVM linear/poly/rbf.
- GridSearchCV su Train (cv=5; accuracy): stampa `val_acc` e `best_params` per i tre SVM.
- Miglior modello scelto su Validation.
![val_base](plots_classification/val_baseline.png)
![val_tuned](plots_classification/val_tuned.png)

# Slide 16 — classification.py (Test & Confusion Matrix)
- Refit del migliore su Train+Val; valutazione su Test (accuracy stampata).
- Confusion Matrix salvata in `plots_classification/confusion_matrix.png`.
![cm](plots_classification/confusion_matrix.png)

# Slide 17 — classification.py (Studio Statistico CV k=10)
- Cross‑validation su Train+Val (k=10, accuracy): media, dev.std., mediana, min, max, range, IC95%.
- Grafici salvati in `plots_classification/`:
![cv_hist](plots_classification/cv_hist.png)
![cv_scat](plots_classification/cv_scatter.png)
![cv_box](plots_classification/cv_box.png)

# Slide 20 — regressione_lineare.py (Flusso)
- `preprocess_eur` → SLR WorkExp → CompTotal (EUR).
- Stampa coefficienti, R², MSE, RMSE; scatter + retta.
- Diagnostica residui: residuals vs fitted, QQ‑plot.

# Slide 21 — Regressione (Grafici)
![reg_scatter](plots_regression/scatter_fit_WorkExp_CompTotal.png)
![resid](plots_regression/residuals_vs_fitted_eur.png)
![qq](plots_regression/qq_residuals_eur.png)
- Segnale positivo modesto; eteroschedasticità; code leggere.

# Slide 22 — Regressione (Per Paese)
![DE](plots_regression/by_country/Germany_slr.png)
![FR](plots_regression/by_country/France_slr.png)
![NL](plots_regression/by_country/Netherlands_slr.png)
![IT](plots_regression/by_country/Italy_slr.png)
![ES](plots_regression/by_country/Spain_slr.png)
- Differenze di baseline e pendenza per Paese.

# Slide 23 — Preprocessing (Riepilogo)
- EUR: plausibilità e IQR trim.
- Class.: bounds numeriche, mapping ordinale, top‑N capping, one‑hot in Pipeline.
- Pipeline evita leakage (fit solo su Train).

# Slide 24 — Splitting & Tuning (Riepilogo traccia)
- 60/20/20 (classification.py) e 70/15/15 (slides) — validi.
- Selezione su validation; test una sola volta (no peeking).
- Tuning: GridSearchCV su Train; confronto su Val.

# Slide 25 — Metriche & Studio Statistico
- Class.: accuracy (Test), confusion matrix.
- Regr.: R², MSE (e RMSE), residui/QQ‑plot.
- k run / k‑fold: media ± IC95, istogrammi/boxplot.

# Slide 26 — Conclusioni
- Classificazione: Logistic leggermente migliore (~0.74 test), stabile.
- Regressione: trend positivo ma R² basso (molta varianza non spiegata).
- EDA ha guidato scelte e interpretazioni (Employment con segnale; WorkExp correlato ma debole).

# Slide 27 — Lavori Futuri (Facolt.)
- Class.: multi‑hot per AI usage; griglie un po’ più ampie.
- Regr.: log(CompTotal), SE robusti (HC3).
- Controlli geografici più fini.

# Slide 28 — Riproducibilità
- Esecuzione:
  - `python eda.py`
  - `python classification.py`
  - `python classification_slides.py`
  - `python regressione_lineare.py`
- Headless: `MPLBACKEND=Agg python <script>.py`
- Dipendenze: pandas, numpy, matplotlib, seaborn, scikit‑learn, statsmodels.
