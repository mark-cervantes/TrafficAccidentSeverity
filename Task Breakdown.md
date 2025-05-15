## EDSA Severe Traffic Accident Forecasting: Task Breakdown (v2.2 Compliant)

**Project Goal:** Forecast severe traffic accidents on EDSA, identify key predictors, and synthesize model suitability.
**Deliverables:** Documented Jupyter notebooks (as per PRD 9.1), summary report with model synthesis, detailed model performance Excel sheet, utility scripts.

---

### Phase 1: Project Setup & Initial Data Handling

*   **Commit 1: Project Initialization & Environment Setup**
    *   **Task:** Create main project directory (`edsa_traffic_forecasting/`) and subdirectories: `data/`, `notebooks/`, `src/`, `reports/`, `models/`.
    *   **Task:** Initialize Git repository.
    *   **Task:** Create `requirements.txt` (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, JupyterLab, openpyxl, XGBoost, LightGBM, SHAP).
    *   **Task:** Create `.gitignore`.
    *   **Task:** Set up Python virtual environment and install dependencies.
    *   **Task:** Create basic `README.md`.
    *   **PRD Ref:** 7, 8; NFR1, NFR2, NFR4.
    *   **Notebook:** N/A

*   **Commit 2: Data Ingestion & Initial Inspection (`00_data_ingestion_and_initial_exploration.ipynb`)**
    *   **Task:** Create `notebooks/00_data_ingestion_and_initial_exploration.ipynb`.
    *   **Task:** Load CSV into Pandas DataFrame. (FR1)
    *   **Task:** Perform initial inspection: `head()`, `tail()`, `info()`, `describe(include='all')`, `shape`, `duplicated().sum()`.
    *   **Task:** Document initial observations in the notebook.
    *   **PRD Ref:** 3.1.1, 4.1; FR1.
    *   **Notebook:** `00_data_ingestion_and_initial_exploration.ipynb`

*   **Commit 3: Target Variable Definition & Creation (`01_target_variable_definition.ipynb`)**
    *   **Task:** Create `notebooks/01_target_variable_definition.ipynb`.
    *   **Task:** Investigate `SEVERITY`, `killed_total`, `injured_total` columns. (5.2)
    *   **Task:** Implement logic for binary target `is_severe_accident` (1 if `killed_total > 0` OR `injured_total > 0`, else 0). (3.1.3, 5.1)
    *   **Task:** Verify against `SEVERITY` column, document rationale. (5.2)
    *   **Task:** Add `is_severe_accident` to DataFrame.
    *   **Task:** Analyze and visualize target variable distribution (check for imbalance). (11.2)
    *   **PRD Ref:** 3.1.3, 5.1, 5.2.
    *   **Notebook:** `01_target_variable_definition.ipynb`

---

### Phase 2: Data Cleaning & Preprocessing (`02_data_preprocessing.ipynb` & `src/preprocessing_utils.py`)

*   **Commit 4: Feature Exclusion & Datetime Processing (`02_data_preprocessing.ipynb` - Part 1)**
    *   **Task:** Begin `notebooks/02_data_preprocessing.ipynb`.
    *   **Task:** Drop outcome-related, identifier, and redundant datetime features as per PRD 4.3. Document reasons. (11.1)
    *   **Task:** Convert `DATE_UTC`, `TIME_UTC` to datetime objects; combine into `DATETIME_UTC`.
    *   **PRD Ref:** 4.3, 3.1.2.
    *   **Notebook:** `02_data_preprocessing.ipynb`

*   **Commit 5: Temporal Feature Engineering (`02_data_preprocessing.ipynb` - Part 2)**
    *   **Task:** In `notebooks/02_data_preprocessing.ipynb`, engineer temporal features from `DATETIME_UTC` (hour, day_of_week, month, year, season, is_weekend, etc.). (3.1.2, 4.2)
    *   **Task:** Verify correctness and data types of new temporal features.
    *   **PRD Ref:** 3.1.2, 4.2.
    *   **Notebook:** `02_data_preprocessing.ipynb`

*   **Commit 6: Missing Value Handling (`02_data_preprocessing.ipynb` - Part 3)**
    *   **Task:** In `notebooks/02_data_preprocessing.ipynb`, analyze and impute missing values for `WEATHER`, `LIGHT`, and other key categorical features. Document strategy (e.g., 'Unknown', mode) and rationale. (3.1.2, 2.2.3)
    *   **PRD Ref:** 3.1.2, 2.2.3.
    *   **Notebook:** `02_data_preprocessing.ipynb`

*   **Commit 7: Categorical Feature Encoding (`02_data_preprocessing.ipynb` - Part 4)**
    *   **Task:** In `notebooks/02_data_preprocessing.ipynb`, identify and encode categorical features (e.g., `ROAD`, `MAIN_CAUSE`, `COLLISION_TYPE`, `WEATHER`, `LIGHT`). Apply One-Hot Encoding or other suitable methods. Document choices. (3.1.2)
    *   **PRD Ref:** 3.1.2.
    *   **Notebook:** `02_data_preprocessing.ipynb`

*   **Commit 8: Numerical Scaling, `DESC` Parsing & Final Preprocessing (`02_data_preprocessing.ipynb` - Part 5)**
    *   **Task:** In `notebooks/02_data_preprocessing.ipynb`, identify and plan scaling for numerical features (`Y`, `X`). (Scaler to be fit on train set later).
    *   **Task:** Perform basic keyword-based feature engineering from `DESC` if deemed useful and not leaky. Document decision. (4.2)
    *   **Task:** Drop original `DESC` if processed.
    *   **Task:** Save the fully preprocessed DataFrame to a file (e.g., `data/processed/preprocessed_data.csv`) for easier loading in subsequent notebooks.
    *   **PRD Ref:** 3.1.2, 4.2.
    *   **Notebook:** `02_data_preprocessing.ipynb`

*   **Commit 9: Preprocessing Utility Script (`src/preprocessing_utils.py`)**
    *   **Task:** Create `src/preprocessing_utils.py`.
    *   **Task:** Refactor core preprocessing steps from `notebooks/02_data_preprocessing.ipynb` into well-documented, reusable functions in `src/preprocessing_utils.py`.
    *   **Task:** Update `notebooks/02_data_preprocessing.ipynb` to use these utility functions, demonstrating their usage.
    *   **PRD Ref:** 9.4; NFR1, NFR4.
    *   **Notebook:** `02_data_preprocessing.ipynb` (updated)

---

### Phase 3: Exploratory Data Analysis (`03_exploratory_data_analysis.ipynb`)

*   **Commit 10: Univariate & Bivariate EDA (`03_exploratory_data_analysis.ipynb` - Part 1)**
    *   **Task:** Create `notebooks/03_exploratory_data_analysis.ipynb`. Load preprocessed data.
    *   **Task:** Perform univariate analysis (histograms, box plots for numerical; bar charts for categorical). (FR2)
    *   **Task:** Perform bivariate analysis (features vs. target `is_severe_accident` using stacked bars, box/violin plots). (FR2)
    *   **Task:** Generate clear visualizations (NFR3) and document observations.
    *   **PRD Ref:** 2.2.2, 3.1.4; FR2; NFR3.
    *   **Notebook:** `03_exploratory_data_analysis.ipynb`

*   **Commit 11: Temporal EDA & Correlation Analysis (`03_exploratory_data_analysis.ipynb` - Part 2)**
    *   **Task:** In `notebooks/03_exploratory_data_analysis.ipynb`, visualize accident trends over time (hour, day, month, season). (3.1.4)
    *   **Task:** Compute and visualize numerical feature correlation matrix. (3.1.4)
    *   **Task:** Investigate associations between key categorical features.
    *   **Task:** Document all significant findings.
    *   **PRD Ref:** 3.1.4; FR2.
    *   **Notebook:** `03_exploratory_data_analysis.ipynb`

---

### Phase 4: Modeling Framework (`04_modeling_pipeline_setup.ipynb` & `src/modeling_utils.py`)

*   **Commit 12: Train-Test Split, Evaluation & Results Tracking Setup (`04_modeling_pipeline_setup.ipynb`)**
    *   **Task:** Create `notebooks/04_modeling_pipeline_setup.ipynb`.
    *   **Task:** Create/enhance `src/modeling_utils.py`.
    *   **Task:** Implement robust train-test split (stratified, fixed `random_state`). (NFR2)
    *   **Task:** Implement evaluation metric functions (Precision, Recall, F1, ROC-AUC) in `src/modeling_utils.py`. (3.1.7)
    *   **Task:** Define Excel sheet (`reports/model_performance_summary.xlsx`) structure. **Crucially, include columns for `Hyperparameter_Set_Tried (JSON/string)`, `CV_Score_for_Set`, in addition to `Selected_Final_Hyperparameters`, `Training_Time_Seconds`, metrics, etc.** (3.1.5, 9.3, 10.5)
    *   **Task:** Create utility function in `src/modeling_utils.py` to append rows (including detailed hyperparameter trials) to this Excel sheet.
    *   **Task:** In `04_modeling_pipeline_setup.ipynb`, demonstrate train-test split. Load preprocessed data. Apply numerical scaling (fit `StandardScaler` or `MinMaxScaler` on X_train, transform X_train & X_test).
    *   **PRD Ref:** 3.1.5, 3.1.7, 9.3, 10.5; NFR2.
    *   **Notebook:** `04_modeling_pipeline_setup.ipynb`

*   **Commit 13: Class Imbalance Handling Strategy (`04_modeling_pipeline_setup.ipynb` - Cont.)**
    *   **Task:** Continue in `notebooks/04_modeling_pipeline_setup.ipynb`.
    *   **Task:** Research, select, and implement class imbalance technique(s) (e.g., SMOTE, class weighting). Apply to training data *after* split. (3.1.7, 11.2)
    *   **Task:** Document rationale and demonstrate application on `X_train_scaled`, `y_train`.
    *   **PRD Ref:** 3.1.7, 11.2.
    *   **Notebook:** `04_modeling_pipeline_setup.ipynb`

---

### Phase 5: Model Development, Tuning & Evaluation (Individual Notebooks per Model)

*(General pattern for each model notebook: Load data, apply preprocessing utilities, split, scale, handle imbalance. Implement model, train, evaluate. Perform hyperparameter tuning, meticulously log experiments to Excel. Save best model and final results.)*

*   **Commit 14: Logistic Regression (`05_model_logistic_regression.ipynb`)**
    *   **Task:** Create `notebooks/05_model_logistic_regression.ipynb`.
    *   **Task:** Implement, train, evaluate Logistic Regression. Consider `class_weight='balanced'`. (3.1.5.1, FR3)
    *   **Task:** Perform basic hyperparameter tuning (e.g., `C`, `solver`).
    *   **Task:** **Log all hyperparameter sets tried, their CV scores (if applicable), selected hyperparameters, training time, and evaluation metrics to the Excel sheet.** (3.1.5, FR3, 9.3, 10.5)
    *   **Task:** Save model. Document process and results.
    *   **PRD Ref:** 3.1.5.1; FR3; 9.1, 9.3, 10.5.
    *   **Notebook:** `05_model_logistic_regression.ipynb`

*   **Commit 15: Decision Tree - Initial & Tuning (`06_model_decision_tree.ipynb`)**
    *   **Task:** Create `notebooks/06_model_decision_tree.ipynb`.
    *   **Task:** Implement Decision Tree. Perform hyperparameter tuning (`max_depth`, `min_samples_split`, etc.) using `GridSearchCV` or `RandomizedSearchCV`. (3.1.5.2, FR3)
    *   **Task:** **For each hyperparameter combination evaluated by the search, log it and its CV score to the Excel sheet. Log selected best hyperparameters, training time, and final train/test evaluation metrics.** (3.1.5, FR3, 9.3, 10.5)
    *   **Task:** Save best model. Document.
    *   **PRD Ref:** 3.1.5.2; FR3; 9.1, 9.3, 10.5.
    *   **Notebook:** `06_model_decision_tree.ipynb`

*   **Commit 16: Random Forest - Initial & Tuning (`07_model_random_forest.ipynb`)**
    *   **Task:** Create `notebooks/07_model_random_forest.ipynb`.
    *   **Task:** Implement Random Forest. Tune (`n_estimators`, `max_depth`, etc.). (3.1.5.3, FR3)
    *   **Task:** **Log all hyperparameter experiments and final results to Excel as detailed for Decision Tree.** (3.1.5, FR3, 9.3, 10.5)
    *   **Task:** Save best model. Document.
    *   **PRD Ref:** 3.1.5.3; FR3; 9.1, 9.3, 10.5.
    *   **Notebook:** `07_model_random_forest.ipynb`

*   **Commit 17: LightGBM - Initial & Tuning (`08_model_lightgbm.ipynb`)**
    *   **Task:** Create `notebooks/08_model_lightgbm.ipynb`.
    *   **Task:** Implement LightGBM. Tune (`n_estimators`, `learning_rate`, `num_leaves`, etc.). (3.1.5.4, FR3)
    *   **Task:** **Log all hyperparameter experiments and final results to Excel.** (3.1.5, FR3, 9.3, 10.5)
    *   **Task:** Save best model. Document.
    *   **PRD Ref:** 3.1.5.4; FR3; 9.1, 9.3, 10.5.
    *   **Notebook:** `08_model_lightgbm.ipynb`

*   **Commit 18: XGBoost - Initial & Tuning (`09_model_xgboost.ipynb`)**
    *   **Task:** Create `notebooks/09_model_xgboost.ipynb`.
    *   **Task:** Implement XGBoost. Tune (`n_estimators`, `learning_rate`, `max_depth`, etc.). (3.1.5.4, 3.1.5.6, FR3)
    *   **Task:** **Log all hyperparameter experiments and final results to Excel.** (3.1.5, FR3, 9.3, 10.5)
    *   **Task:** Save best model. Document.
    *   **PRD Ref:** 3.1.5.4, 3.1.5.6; FR3; 9.1, 9.3, 10.5.
    *   **Notebook:** `09_model_xgboost.ipynb`

*   **Commit 19: Bagging Ensemble - Initial & Tuning (`10_model_bagging_classifier.ipynb`)**
    *   **Task:** Create `notebooks/10_model_bagging_classifier.ipynb`.
    *   **Task:** Implement `BaggingClassifier` (e.g., with Decision Trees). Tune (`n_estimators`, `max_samples`, etc.). (3.1.5.5, FR3)
    *   **Task:** **Log all hyperparameter experiments and final results to Excel.** (3.1.5, FR3, 9.3, 10.5)
    *   **Task:** Save best model. Document.
    *   **PRD Ref:** 3.1.5.5; FR3; 9.1, 9.3, 10.5.
    *   **Notebook:** `10_model_bagging_classifier.ipynb`

*   **Commit 20: AdaBoost Ensemble - Initial & Tuning (`11_model_adaboost.ipynb`)**
    *   **Task:** Create `notebooks/11_model_adaboost.ipynb`.
    *   **Task:** Implement `AdaBoostClassifier`. Tune (`n_estimators`, `learning_rate`, base estimator params). (3.1.5.6, FR3)
    *   **Task:** **Log all hyperparameter experiments and final results to Excel.** (3.1.5, FR3, 9.3, 10.5)
    *   **Task:** Save best model. Document.
    *   **PRD Ref:** 3.1.5.6; FR3; 9.1, 9.3, 10.5.
    *   **Notebook:** `11_model_adaboost.ipynb`

*   **Commit 21: Model Persistence Utilities Enhancement (`src/modeling_utils.py`)**
    *   **Task:** In `src/modeling_utils.py`, refine/ensure robust functions to save (e.g., `joblib`) and load all types of trained models.
    *   **Task:** Ensure best tuned models are saved with clear naming conventions in `models/`.
    *   **PRD Ref:** NFR4.
    *   **Notebook:** N/A (utilities used by model notebooks)

---

### Phase 6: Feature Importance Analysis (`12_feature_importance_analysis.ipynb`)

*   **Commit 22: Model-Based Feature Importance (`12_feature_importance_analysis.ipynb` - Part 1)**
    *   **Task:** Create `notebooks/12_feature_importance_analysis.ipynb`.
    *   **Task:** Load best tuned tree-based models and Logistic Regression.
    *   **Task:** Extract and visualize feature importances (`feature_importances_`, `coef_`). Compare across models. (3.1.6, FR4)
    *   **PRD Ref:** 3.1.6; FR4; 9.1.
    *   **Notebook:** `12_feature_importance_analysis.ipynb`

*   **Commit 23: SHAP Value Analysis (`12_feature_importance_analysis.ipynb` - Part 2)**
    *   **Task:** In `notebooks/12_feature_importance_analysis.ipynb`, implement SHAP value analysis for 1-2 top models.
    *   **Task:** Generate and interpret SHAP summary and dependence plots. (3.1.6, FR4)
    *   **PRD Ref:** 3.1.6; FR4; 8.
    *   **Notebook:** `12_feature_importance_analysis.ipynb`

*   **Commit 24: Statistical Feature Significance Tests (`12_feature_importance_analysis.ipynb` - Part 3)**
    *   **Task:** In `notebooks/12_feature_importance_analysis.ipynb`, perform Chi-squared and ANOVA F-tests for feature significance.
    *   **Task:** Document and compare with model-based importances. (3.1.6, FR4)
    *   **PRD Ref:** 3.1.6; FR4.
    *   **Notebook:** `12_feature_importance_analysis.ipynb`

---

### Phase 7: Reporting & Finalization (`13_results_summary_and_visualization.ipynb` & Report)

*   **Commit 25: Finalize Excel Sheet & Comparative Visualizations (`13_results_summary_and_visualization.ipynb` - Part 1)**
    *   **Task:** Create `notebooks/13_results_summary_and_visualization.ipynb`.
    *   **Task:** Ensure `reports/model_performance_summary.xlsx` is complete, accurate, and contains all detailed hyperparameter experiments and final results. (3.1.5, 9.3, 10.5)
    *   **Task:** In the notebook, load data from Excel. Create comparative visualizations of model performances (F1, ROC-AUC, etc.). Plot ROC curves. (2.2.1)
    *   **Task:** Save visualizations to `reports/figures/`.
    *   **PRD Ref:** 2.2.1, 3.1.5, 9.3, 10.5; FR3.
    *   **Notebook:** `13_results_summary_and_visualization.ipynb`

*   **Commit 26: Model Synthesis & Suitability Analysis (`13_results_summary_and_visualization.ipynb` - Part 2)**
    *   **Task:** In `notebooks/13_results_summary_and_visualization.ipynb`:
    *   **Task:** Analyze the comprehensive results from the Excel sheet and visualizations.
    *   **Task:** **Synthesize findings: Discuss the relative strengths and weaknesses of each model for this specific problem.**
    *   **Task:** **Consider trade-offs: predictive performance (Precision, Recall, F1, ROC-AUC), interpretability, training time/computational cost (from Excel logs), and robustness to hyperparameter choices (from detailed Excel logs).**
    *   **Task:** **Formulate a recommendation for the most suitable model(s) for forecasting severe EDSA accidents, justifying the choice.** (2.2.1, FR5, 10.6)
    *   **Task:** Document this synthesis clearly in the notebook, preparing it for the summary report.
    *   **PRD Ref:** 2.2.1; FR5; 9.2, 10.6.
    *   **Notebook:** `13_results_summary_and_visualization.ipynb`

*   **Commit 27: Draft Summary Report (`reports/EDSA_Traffic_Accident_Forecasting_Report.md`)**
    *   **Task:** Create `reports/EDSA_Traffic_Accident_Forecasting_Report.md`.
    *   **Task:** Draft main sections: Intro, Data, Methodology, Model Performance, Feature Importance.
    *   **Task:** **Integrate the detailed Model Synthesis and Suitability Analysis from the previous commit into a dedicated section of the report.** (FR5, 9.2, 10.6)
    *   **Task:** Include actionable insights from feature importance. (10.4)
    *   **PRD Ref:** 2.1.2, 9.2; FR5; 10.4, 10.6.
    *   **Notebook:** N/A (Report document)

*   **Commit 28: Notebook Cleaning, Documentation & Utility Script Finalization**
    *   **Task:** Review all Jupyter notebooks for clarity, execution, comments, and visualization quality. (9.1, NFR1, NFR3)
    *   **Task:** Review utility scripts (`src/`) for PEP 8, docstrings, and robustness. (NFR1, NFR4)
    *   **Task:** Ensure consistent use of fixed random seeds for reproducibility. (NFR2)
    *   **PRD Ref:** 9.1, 9.4; NFR1, NFR2, NFR3, NFR4.
    *   **Notebooks:** All

*   **Commit 29: Finalize Summary Report**
    *   **Task:** In `reports/EDSA_Traffic_Accident_Forecasting_Report.md`:
    *   **Task:** Write Executive Summary (goal, best model, top predictors, key insights, model suitability recommendation).
    *   **Task:** Add Conclusion (summary, limitations, future work).
    *   **Task:** Review entire report for coherence, grammar, and completeness, ensuring the model synthesis is prominent. (FR5, 9.2)
    *   **PRD Ref:** FR5; 9.2.
    *   **Notebook:** N/A (Report document)

*   **Commit 30: Update `README.md`, Final Review & Version Tagging**
    *   **Task:** Expand `README.md` (project details, setup, execution, summary of findings, link to report/Excel).
    *   **Task:** Final check of project structure, deliverables.
    *   **Task:** Update `requirements.txt` with exact versions (`pip freeze > requirements.txt`). (NFR2)
    *   **Task:** Git tag `v1.0.0` (or appropriate version).
    *   **PRD Ref:** NFR1, NFR2, NFR4; 10.3.
    *   **Notebook:** N/A