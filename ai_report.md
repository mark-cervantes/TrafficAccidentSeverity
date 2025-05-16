# AI Project Report: EDSA Traffic Accident Forecasting

This report details the execution of the EDSA Traffic Accident Forecasting project, addressing each stage of the machine learning process as outlined in the project tasks.

## Stage 1: Defining the Problem and Business Goal

This stage focuses on clearly establishing the project's objectives and scope.

*   **Clearly define the business problem you're trying to solve.**
    *   The primary business problem is to **forecast when severe traffic accidents are likely to occur on EDSA, Metro Manila**. This is based on contextual features available in the provided dataset ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:14), Section 1). The aim is to provide actionable insights for road safety measures.

*   **Identify key stakeholders and their objectives.**
    *   The PRD identifies the **Project Owner as "User"** and the **Target AI Assistant as "AI Pair Programmer"** ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:7-8)).
    *   The primary objectives are:
        1.  **Forecast Severe Accidents:** Develop ML models to predict the likelihood of severe accidents (injury or fatality) ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:23)).
        2.  **Feature Importance Analysis:** Identify and rank significant predictors of severe accidents for actionable insights ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:24)).

*   **Translate the business problem into a specific ML task (classification, regression, etc.).**
    *   The business problem is translated into a **binary classification task** ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:43-44)). The model will predict whether an accident is 'Severe' (1) or 'Non-severe' (0).

*   **Define success metrics from a business perspective.**
    *   Success is measured by:
        1.  Accurate forecasting of severe accidents (high Precision, Recall, F1-score on the test set) ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:212)).
        2.  Clear identification and ranking of significant predictors ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:213)).
        3.  Actionable insights derived from feature importance analysis ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:215)).
        4.  Well-documented, reproducible code and comprehensive tracking of experiments ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:214), [`Product Requirements Document.md`](Product%20Requirements%20Document.md:216)).
        5.  A clear synthesis identifying the most suitable model(s) ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:217)).

*   **Consider feasibility and potential impact.**
    *   Feasibility is addressed by using a provided dataset and standard ML libraries. The scope is defined to exclude real-time deployment or UI development, focusing on model building and analysis ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:71-75)).
    *   The potential impact is significant: providing insights for road safety interventions to reduce severe accidents ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:14)).

## Stage 2: Gathering and Preparing data for ML

This stage covers all aspects of data acquisition and preparation for modeling.

*   **Identifying and sourcing relevant data (databases, APIs, logs, etc.).**
    *   The data was provided as a single CSV file: `RTA_EDSA_2007-2016.csv`. This is loaded in [`notebooks/00_data_ingestion_and_initial_exploration.ipynb`](notebooks/00_data_ingestion_and_initial_exploration.ipynb:76) and referenced in the PRD ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:82)).

*   **Data collection strategies and challenges.**
    *   Data collection was not part of this project's scope as the dataset was provided. Challenges would relate to the inherent quality and completeness of the provided historical data.

*   **Data cleaning: Handling missing values, outliers, inconsistencies.**
    *   **Missing Values:**
        *   Categorical features (`WEATHER`, `LIGHT`, `MAIN_CAUSE`, etc.) had missing values imputed with the string 'Unknown' as detailed in [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb:405) and the function `impute_missing_categorical` in [`src/preprocessing_utils.py`](src/preprocessing_utils.py:64).
        *   Numerical casualty counts (`killed_total`, `injured_total`) were filled with 0 before defining the target variable in [`notebooks/01_target_variable_definition.ipynb`](notebooks/01_target_variable_definition.ipynb:269-270).
    *   **Outliers/Inconsistencies:** While not explicitly detailed as a separate outlier detection phase, data inspection occurs in Notebook 00. The primary focus was on robust handling of defined categories and missing data. The `SEVERITY` column was cross-verified against casualty counts in [`notebooks/01_target_variable_definition.ipynb`](notebooks/01_target_variable_definition.ipynb:397) to ensure consistency for the target variable.

*   **Data preprocessing.**
    *   **Target Variable Definition:** A binary target `is_severe_accident` (renamed to `SEVERITY` in later preprocessed data) was created in [`notebooks/01_target_variable_definition.ipynb`](notebooks/01_target_variable_definition.ipynb:272), where 1 indicates `killed_total > 0` or `injured_total > 0`.
    *   **Feature Exclusion:** Outcome-related columns (casualty counts), identifiers (`INCIDENTDETAILS_ID`, `LOCATION_TEXT`, `ADDRESS`), and redundant datetime (`DATETIME_PST`) were dropped as per PRD Section 4.3 and implemented in [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb:71-80) and [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb:108-112) (using functions from [`src/preprocessing_utils.py`](src/preprocessing_utils.py)).
    *   **Datetime Engineering:** `DATE_UTC` and `TIME_UTC` were combined into `DATETIME_UTC`. Temporal features (hour, day_of_week, day, month, year, is_weekend, season) were extracted in [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb:280-303) (using `extract_temporal_features` from [`src/preprocessing_utils.py`](src/preprocessing_utils.py:45)).
    *   **Categorical Encoding:** Features like `ROAD`, `MAIN_CAUSE`, `COLLISION_TYPE`, `WEATHER`, `LIGHT`, `REPORTING_AGENCY`, and `season` were One-Hot Encoded in [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb:458-462) (using `encode_categorical` from [`src/preprocessing_utils.py`](src/preprocessing_utils.py:74)).
    *   **Numerical Scaling:** Numerical features were scaled using `StandardScaler` in [`notebooks/04_modeling_pipeline_setup.ipynb`](notebooks/04_modeling_pipeline_setup.ipynb:72-81) after the train-test split.
    *   The fully preprocessed data is saved to [`data/processed/preprocessed_data.csv`](data/processed/preprocessed_data.csv).

*   **Data splitting: Training, Validation, and test sets.**
    *   The data was split into training (80%) and test (20%) sets using a stratified approach based on the target variable `SEVERITY`. A `random_state=42` was used for reproducibility. This is implemented in [`notebooks/04_modeling_pipeline_setup.ipynb`](notebooks/04_modeling_pipeline_setup.ipynb:49-58) and repeated in individual modeling notebooks. Validation sets are implicitly created by `GridSearchCV` during hyperparameter tuning.

*   **Addressing data privacy and security.**
    *   The primary privacy consideration was preventing data leakage. This was addressed by dropping outcome-related features (casualty counts) before modeling, as specified in PRD Section 4.3 and implemented in [`notebooks/02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb:71-80). No other explicit data privacy measures for anonymization were detailed, as the dataset was provided for research/modeling purposes.

## Stage 3 - Model Development & Training

This stage details the selection, training, and tuning of machine learning models.

*   **Selecting appropriate ML algorithms based on the task and data.**
    *   The PRD (Section 3.1.5, [`Product Requirements Document.md`](Product%20Requirements%20Document.md:51-58)) specified the following algorithms for this binary classification task:
        1.  Logistic Regression
        2.  Decision Tree Classifier
        3.  Random Forest Classifier
        4.  Gradient Boosting (XGBoost, LightGBM)
        5.  Bagging Ensemble (BaggingClassifier with Decision Trees)
        6.  Boosting Ensemble (AdaBoost)
    *   These models were implemented in their respective notebooks: [`notebooks/05_model_logistic_regression.ipynb`](notebooks/05_model_logistic_regression.ipynb) through `notebooks/11_model_adaboost.ipynb`.

*   **Model architecture design (if applicable, e.g., neural networks).**
    *   Not applicable for the selected traditional ML algorithms.

*   **Hyperparameter tuning.**
    *   `GridSearchCV` was used for hyperparameter tuning for each model, as demonstrated in [`notebooks/05_model_logistic_regression.ipynb`](notebooks/05_model_logistic_regression.ipynb:278) (and similarly in other model notebooks).
    *   The primary scoring metric for tuning was `roc_auc_ovr` or F1-score, depending on the model notebook's specific setup (PRD Section 3.3, [`Product Requirements Document.md`](Product%20Requirements%20Document.md:46)).
    *   Key hyperparameter combinations explored and selected final hyperparameters are logged in [`reports/model_performance_summary.xlsx`](reports/model_performance_summary.xlsx), managed by functions in [`src/modeling_utils.py`](src/modeling_utils.py:123) (PRD Section 6.FR3, 9.3).

*   **Training the model on the training data.**
    *   Each model was trained on the (potentially resampled, e.g., with SMOTE) training data (`X_train_scaled`, `y_train` or `X_train_resampled`, `y_train_resampled`) using the `.fit()` method of the respective scikit-learn estimator. This is shown in each modeling notebook (e.g., [`notebooks/05_model_logistic_regression.ipynb`](notebooks/05_model_logistic_regression.ipynb:151)).

*   **Using techniques to prevent overfitting.**
    *   **Cross-validation:** `GridSearchCV` inherently uses k-fold cross-validation (cv=5 was common, e.g., [`notebooks/05_model_logistic_regression.ipynb`](notebooks/05_model_logistic_regression.ipynb:284)) during hyperparameter tuning, which helps in selecting parameters that generalize well.
    *   **Regularization:** For models like Logistic Regression, regularization strength (parameter `C`) was part of the hyperparameter search space ([`notebooks/05_model_logistic_regression.ipynb`](notebooks/05_model_logistic_regression.ipynb:196)).
    *   **Tree Pruning/Complexity Control:** For tree-based models (Decision Tree, Random Forest, Gradient Boosting), hyperparameters controlling tree depth, minimum samples per leaf, etc., were tuned to prevent overfitting.
    *   **Train/Test Evaluation:** Comparing performance on training and test sets (as planned in [`notebooks/13_results_summary_and_visualization.ipynb`](notebooks/13_results_summary_and_visualization.ipynb:353-360)) helps identify overfitting.

## Stage 4 - Model Evaluation & Selection

This stage focuses on assessing model performance and choosing the best model(s).

*   **Evaluating model performance using relevant metrics (aligned with business goals).**
    *   Models were evaluated using Precision, Recall, F1-score, and ROC-AUC. These are aligned with the business goals of accurately forecasting severe accidents and understanding predictor importance ([`Product Requirements Document.md`](Product%20Requirements%20Document.md:212)).
    *   The function `compute_classification_metrics` in [`src/modeling_utils.py`](src/modeling_utils.py:15) is used for calculating these metrics.
    *   Performance on the test set for each model is logged in the `Model_Summaries` sheet of [`reports/model_performance_summary.xlsx`](reports/model_performance_summary.xlsx). [`notebooks/13_results_summary_and_visualization.ipynb`](notebooks/13_results_summary_and_visualization.ipynb:73) loads and displays this summary.

*   **Comparing different models and their performance on the validation set.**
    *   [`notebooks/13_results_summary_and_visualization.ipynb`](notebooks/13_results_summary_and_visualization.ipynb:119-155) generates bar charts comparing Test F1, Test ROC-AUC, Precision, and Recall across all trained models, using data from [`reports/model_performance_summary.xlsx`](reports/model_performance_summary.xlsx).
    *   Comparative ROC curves are also plotted in Notebook 13 ([`notebooks/13_results_summary_and_visualization.ipynb`](notebooks/13_results_summary_and_visualization.ipynb:211-293)) and saved to [`reports/figures/summary_visualizations/roc_curves_comparison.png`](reports/figures/summary_visualizations/roc_curves_comparison.png).
    *   The (placeholder) Section 4 of [`reports/EDSA_Traffic_Accident_Forecasting_Report.md`](reports/EDSA_Traffic_Accident_Forecasting_Report.md:51) is intended to summarize these comparisons.

*   **Performing error analysis to understand model weaknesses.**
    *   **Feature Importance:** [`notebooks/12_feature_importance_analysis.ipynb`](notebooks/12_feature_importance_analysis.ipynb:192) extracts and visualizes model-based feature importances (e.g., `feature_importances_` for tree models, coefficients for Logistic Regression). This helps understand what drives model predictions.
    *   **SHAP Value Analysis:** For selected tree-based models (XGBoost, LightGBM), SHAP summary plots and dependence plots are generated in [`notebooks/12_feature_importance_analysis.ipynb`](notebooks/12_feature_importance_analysis.ipynb:354) to provide nuanced insights into feature contributions and interactions. Plots are saved in `reports/figures/shap_analysis/`.
    *   **Statistical Tests:** Chi-squared and ANOVA F-tests are performed in [`notebooks/12_feature_importance_analysis.ipynb`](notebooks/12_feature_importance_analysis.ipynb:476) to assess feature significance statistically, offering another perspective on feature relevance.
    *   These analyses help understand which features are most influential and potentially where models might be making errors or relying on specific data characteristics.

*   **Selecting the best model for deployment based on relevant criteria.**
    *   The criteria for model selection are outlined in [`notebooks/13_results_summary_and_visualization.ipynb`](notebooks/13_results_summary_and_visualization.ipynb:300) (Section 5) and include:
        1.  Predictive Performance (Test F1, ROC-AUC, Precision, Recall).
        2.  Interpretability.
        3.  Training Time/Computational Cost.
        4.  Robustness to Hyperparameters (from CV trial analysis).
        5.  Complexity.
        6.  Feature Importance Consistency.
    *   The (placeholder) Section 6 of [`reports/EDSA_Traffic_Accident_Forecasting_Report.md`](reports/EDSA_Traffic_Accident_Forecasting_Report.md:75) and Section 5.5 of [`notebooks/13_results_summary_and_visualization.ipynb`](notebooks/13_results_summary_and_visualization.ipynb:450) are designated for the final model synthesis and recommendation.
    *   **The actual selection of the "best" model and its specific performance metrics would be detailed in these filled-out sections, based on the quantitative results from the `Model_Summaries` sheet in [`reports/model_performance_summary.xlsx`](reports/model_performance_summary.xlsx) and the analyses in Notebooks 12 and 13.** Without executing the final summarization steps or having the populated Excel file, I can confirm the *process* for selection is in place.

## Conclusion

The project comprehensively addresses all stages of the machine learning lifecycle outlined in `tasks.md`. Methodologies for data preparation, model training, hyperparameter tuning, evaluation, and feature analysis are well-defined and implemented across the various Jupyter notebooks and utility scripts. The final selection of the best model and detailed quantitative results would be found in the completed `reports/model_performance_summary.xlsx` and the finalized `reports/EDSA_Traffic_Accident_Forecasting_Report.md`.