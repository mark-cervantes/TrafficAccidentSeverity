Okay, here is the updated Product Requirements Document (PRD) incorporating your requests.

## Product Requirements Document (PRD): EDSA Severe Traffic Accident Forecasting and Feature Importance Analysis

**Version:** 2.2
**Date:** 2025-05-15
**Project Owner:** User
**Target AI Assistant:** AI Pair Programmer

---

### 1. Introduction

This document outlines the requirements for a project to **forecast when severe traffic accidents are likely to occur** on EDSA, Metro Manila, based on contextual features available in the provided dataset. The project will also include an **analysis of the most significant predictors** of severe accidents, providing actionable insights for road safety measures.

The primary focus is on **forecasting severe accidents** and identifying the **key contextual factors** that contribute to their occurrence. The project will leverage machine learning models, including ensemble methods, to achieve these goals.

---

### 2. Goals

#### 2.1 Primary Goals
1.  **Forecast Severe Accidents:** Develop machine learning models to predict the likelihood of severe accidents (e.g., injury or fatality) based on contextual features such as time, location, weather, and other available data.
2.  **Feature Importance Analysis:** Identify and rank the most significant predictors of severe accidents to provide actionable insights for road safety interventions.

#### 2.2 Secondary Goals
1.  Perform a comparative analysis of different machine learning algorithms for forecasting severe accidents, *including a synthesis of which models are most suitable for this specific task based on performance, interpretability, and computational cost.*
2.  Conduct exploratory data analysis (EDA) to uncover patterns and trends in the dataset.
3.  Address data quality issues (e.g., missing values, inconsistent formats) to ensure robust model performance.

---

### 3. Scope

#### 3.1 In Scope
1.  **Data Ingestion:** Load the provided CSV dataset into a suitable data structure for analysis.
2.  **Data Cleaning & Preprocessing:**
    *   Handle missing values (e.g., `WEATHER`, `LIGHT`).
    *   Encode categorical features (e.g., `ROAD`, `MAIN_CAUSE`, `COLLISION_TYPE`).
    *   Engineer features from temporal data (e.g., hour, day of the week, month, season).
    *   Scale/normalize numerical features as needed.
3.  **Target Variable Definition:**
    *   Define "severe accidents" as a binary classification problem:
        *   Severe: Accidents involving injuries or fatalities.
        *   Non-severe: Accidents involving only property damage.
    *   Derive the target variable from the `SEVERITY`, `killed_total`, and `injured_total` columns.
4.  **Exploratory Data Analysis (EDA):**
    *   Analyze feature distributions and relationships with the target variable.
    *   Visualize trends over time (e.g., accident frequency by hour, day, or month).
    *   Investigate correlations between features.
5.  **Model Development:**
    *   Implement, train, and evaluate the following models:
        1.  Logistic Regression
        2.  Decision Tree Classifier
        3.  Random Forest Classifier
        4.  Gradient Boosting (e.g., XGBoost, LightGBM)
        5.  Bagging Ensemble (e.g., BaggingClassifier with Decision Trees)
        6.  Boosting Ensemble (e.g., AdaBoost, Gradient Boosting, XGBoost)
    *   Perform hyperparameter tuning for each model.
    *   Save training progress details (e.g., *key hyperparameter combinations tested during tuning, selected final hyperparameters*, training time, evaluation metrics) for each model in an **Excel sheet** for documentation, *detailed experimentation tracking,* and comparison.
6.  **Feature Importance Analysis:**
    *   Use model-based techniques (e.g., feature importance from tree-based models, SHAP values) to rank predictors.
    *   Perform statistical tests (e.g., chi-square, ANOVA) to validate feature significance.
7.  **Model Evaluation:**
    *   Evaluate models using appropriate metrics (e.g., Precision, Recall, F1-score, ROC-AUC).
    *   Address class imbalance using techniques like SMOTE or class weighting.
8.  **Deliverables:**
    *   Well-documented Jupyter notebooks for each model.
    *   A summary report on feature importance and model performance.

#### 3.2 Out of Scope
1.  Real-time prediction system deployment.
2.  Development of a user interface (UI) or dashboard.
3.  Integration of external data sources beyond the provided CSV.
4.  Advanced NLP techniques for the `DESC` column beyond basic parsing.

---

### 4. Data

#### 4.1 Source
The dataset is provided as a CSV file. Sample structure:

```csv
LOCATION_TEXT,ROAD,WEATHER,LIGHT,DESC,REPORTING_AGENCY,MAIN_CAUSE,INCIDENTDETAILS_ID,DATE_UTC,TIME_UTC,ADDRESS,killed_driver,killed_passenger,killed_pedestrian,injured_driver,injured_passenger,injured_pedestrian,killed_uncategorized,injured_uncategorized,killed_total,injured_total,DATETIME_PST,SEVERITY,Y,X,COLLISION_TYPE
None,EDSA,None,None,"No Accident Factor, No Collision Stated (based on Police Blotter Book), 0.0 drivers_injured, 0.0 drivers_killed, 0.0 passengers_injured, 0.0 passengers_killed, 0.0 pedestrian_injured, 0.0 pedestrian_killed, Fair (Day), Congressional Ave. Edsa Quezon City",MMDA Road Safety Unit,Human error,2414f311-e805-40c6-b8b9-116b51c35be7,2014-06-30,05:40:00,Congressional Ave. Edsa Quezon City,0,0,0,0,0,0,0,0,0,0,2014-06-30 13:40,Property,14.65771436,121.019788,No Collision Stated
```

#### 4.2 Key Features for Prediction
-   **Temporal Features:** Extracted from `DATE_UTC` and `TIME_UTC` (e.g., hour, day of the week, month, season).
-   **Location Features:** `ROAD`, `Y`, `X` (latitude, longitude).
-   **Environmental Features:** `WEATHER`, `LIGHT`.
-   **Cause and Context Features:** `MAIN_CAUSE`, `COLLISION_TYPE`, `REPORTING_AGENCY`.
-   **Derived Features:** Extracted from `DESC` if useful (e.g., keywords indicating severity).

#### 4.3 Features to Exclude
-   **Outcome Features:** `killed_total`, `injured_total` (to avoid data leakage).
-   **Identifiers:** `INCIDENTDETAILS_ID`, `LOCATION_TEXT`, `ADDRESS` (if redundant with `Y`, `X`).
-   **Datetime Redundancy:** `DATETIME_PST` (if `DATE_UTC` and `TIME_UTC` are used).

---

### 5. Target Variable Definition

#### 5.1 Definition
-   **Severe Accident (1):** Any accident involving injuries or fatalities (`killed_total > 0` or `injured_total > 0`).
-   **Non-Severe Accident (0):** Accidents involving only property damage (`killed_total == 0` and `injured_total == 0`).

#### 5.2 Task
-   Investigate the `SEVERITY` column and casualty counts (`killed_total`, `injured_total`) to define the target variable.
-   Document the rationale for the chosen definition.

---

### 6. Functional Requirements

#### FR1: Data Loading and Preprocessing
-   Load the dataset and inspect its structure.
-   Handle missing values and encode categorical features.
-   Engineer temporal features and scale numerical features.

#### FR2: Exploratory Data Analysis (EDA)
-   Analyze feature distributions and relationships with the target variable.
-   Visualize trends over time and correlations between features.

#### FR3: Model Development
-   Train and evaluate the specified machine learning models:
    1.  Logistic Regression
    2.  Decision Tree Classifier
    3.  Random Forest Classifier
    4.  Gradient Boosting (e.g., XGBoost, LightGBM)
    5.  Bagging Ensemble (e.g., BaggingClassifier with Decision Trees)
    6.  Boosting Ensemble (e.g., AdaBoost, Gradient Boosting, XGBoost)
-   Perform hyperparameter tuning for each model.
-   Save training progress details (e.g., *key hyperparameter combinations tested during tuning, selected final hyperparameters*, training time, evaluation metrics) for each model in an **Excel sheet** for documentation, *detailed experimentation tracking,* and comparison.

#### FR4: Feature Importance Analysis
-   Use model-based techniques (e.g., SHAP, feature importance scores) to rank predictors.
-   Validate findings with statistical tests.

#### FR5: Results and Summary
-   Present model evaluation metrics and feature importance rankings.
-   Summarize insights in a clear and actionable format, *including a synthesis of model suitability for the task based on performance, interpretability, and computational cost.*

---

### 7. Non-Functional Requirements

1.  **Code Quality:** Write clean, readable, and well-commented Python code adhering to PEP 8 standards.
2.  **Reproducibility:** Use fixed random seeds and document library versions.
3.  **Visualization:** Use clear and informative plots for EDA and results presentation.
4.  **Maintainability:** Structure code for easy understanding and future modifications.

---

### 8. Technology Stack

-   **Language:** Python 3.x
-   **Core Libraries:**
    -   Pandas, NumPy (Data manipulation)
    -   Scikit-learn (Machine learning, preprocessing, metrics)
    -   Matplotlib, Seaborn (Visualization)
    -   openpyxl (for Excel interaction)
-   **Boosting Libraries:** XGBoost, LightGBM (for advanced models)
-   **Explainability Libraries:** SHAP (optional, for feature importance)

---

### 9. Deliverables

1.  **Jupyter Notebooks:**
    *   `00_data_ingestion_and_initial_exploration.ipynb`
    *   `01_target_variable_definition.ipynb`
    *   `02_data_preprocessing.ipynb`
    *   `03_exploratory_data_analysis.ipynb`
    *   `04_modeling_pipeline_setup.ipynb` (covers train-test split, imbalance handling, evaluation framework)
    *   `05_model_logistic_regression.ipynb`
    *   `06_model_decision_tree.ipynb`
    *   `07_model_random_forest.ipynb`
    *   `08_model_lightgbm.ipynb`
    *   `09_model_xgboost.ipynb`
    *   `10_model_bagging_classifier.ipynb`
    *   `11_model_adaboost.ipynb`
    *   `12_feature_importance_analysis.ipynb`
    *   `13_results_summary_and_visualization.ipynb`
2.  **Summary Report (`reports/EDSA_Traffic_Accident_Forecasting_Report.md` or `.docx`):**
    *   Comparative analysis of model performance, *including a discussion on the relative strengths and weaknesses of each model for this specific problem, leading to a recommendation of the most suitable model(s) considering performance, interpretability, and computational cost.*
    *   Ranked list of significant predictors with explanations.
    *   Actionable insights derived from the analysis.
3.  **Excel Sheet (`reports/model_performance_summary.xlsx`):**
    *   Detailed log of training progress for each model, including:
        *   Model Name
        *   Timestamp
        *   *Key hyperparameter combinations explored during tuning (e.g., as a JSON string or separate columns if manageable)*
        *   *Selected final hyperparameters (e.g., as a JSON string)*
        *   Training Time (seconds)
        *   CV Mean F1-Score (or other primary metric used for tuning)
        *   Train Precision, Recall, F1, ROC-AUC
        *   Test Precision, Recall, F1, ROC-AUC
        *   Class Imbalance Strategy Used
        *   Notes/Observations for each run/model.
4.  **Utility Scripts (`src/` directory):**
    *   `preprocessing_utils.py`: Common functions for data cleaning and feature engineering.
    *   `modeling_utils.py`: Common functions for model training, evaluation, results logging, and model persistence.
5.  **Requirements File (`requirements.txt`):** Listing all necessary Python libraries and their versions.
6.  **README.md:** Comprehensive project overview, setup instructions, and guide to deliverables.

---

### 10. Success Metrics

1.  Accurate forecasting of severe accidents (e.g., high Precision, Recall, F1-score on the test set for the best model).
2.  Clear identification and ranking of the most significant predictors.
3.  Well-documented, clean, and reproducible code and notebooks.
4.  Actionable insights derived from feature importance analysis and model synthesis.
5.  *Comprehensive tracking of hyperparameter experiments in the Excel sheet.*
6.  *A clear synthesis in the summary report identifying the most suitable model(s) for the task.*

---

### 11. Notes and Pitfalls

1.  **Data Leakage:** Avoid using outcome-related features (e.g., casualty counts) as predictors.
2.  **Class Imbalance:** Address imbalances in the target variable using appropriate techniques and evaluate their impact.
3.  **Feature Engineering:** Ensure derived features are meaningful, not overly complex, and do not inadvertently leak information from the target.
4.  **Interpretability vs. Performance:** Balance model complexity with interpretability for actionable insights. The model synthesis should address this trade-off.
5.  **Hyperparameter Tuning:** Be thorough but also mindful of computational resources. Document the search space and strategy.
6.  **Reproducibility:** Strictly use fixed random seeds for all stochastic operations and document library versions.