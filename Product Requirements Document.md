s a comprehensive Product Requirements Document (PRD) tailored for an AI pair programmer to develop a traffic accident severity prediction model.

## Product Requirements Document: EDSA Traffic Accident Severity Prediction

**Version:** 1.0
**Date:** 2025-05-09
**Project Owner:** User
**Target AI Assistant:** AI Pair Programmer

### 1. Introduction
This document outlines the requirements for a project to predict the severity of traffic accidents on EDSA, Metro Manila. The primary goal is to develop and compare multiple machine learning models using historical accident data. The insights gained will help understand factors contributing to accident severity and can inform proactive road safety measures.

### 2. Goals
*   **Primary Goal:** Develop machine learning models to predict accident severity based on contextual features available prior to or at the time of the incident.
*   **Secondary Goals:**
    *   Perform a comparative analysis of different machine learning algorithms for this prediction task.
    *   Provide insights into the most effective way to define and quantify "accident severity" using the available dataset.
    *   Conduct necessary data preprocessing and exploratory data analysis (EDA) to prepare the data and uncover insights.

### 3. Scope

#### 3.1. In Scope
*   **Data Ingestion:** Loading the provided CSV dataset.
*   **Data Cleaning & Preprocessing:**
    *   Handling missing values (e.g., `WEATHER`, `LIGHT`).
    *   Encoding categorical features (e.g., `ROAD`, `MAIN_CAUSE`, `COLLISION_TYPE`).
    *   Feature engineering:
        *   Extracting temporal features (hour, day of week, month, season) from `DATE_UTC`, `TIME_UTC`, `DATETIME_PST`.
        *   Potentially parsing relevant information from the `DESC` column if it provides unique contextual data not present in other dedicated columns.
    *   Scaling/Normalizing numerical features (`Y`, `X`, engineered numerical features) as appropriate for specific algorithms.
*   **Target Variable Definition ("Severity"):**
    *   Analyze the existing `SEVERITY` column.
    *   Propose and implement a clear definition for the target variable (e.g., 2-class: Property Damage, Injury; or 3-class: Property Damage, Injury, Fatality, if derivable and distinct).
*   **Exploratory Data Analysis (EDA):**
    *   Univariate and bivariate analysis of features.
    *   Visualization of feature distributions and relationships with the target variable.
    *   Correlation analysis.
*   **Model Development (Individual Notebooks):**
    *   Implement, train, and evaluate the following models:
        1.  Logistic Regression
        2.  Decision Tree Classifier
        3.  Random Forest Classifier
        4.  Multi-Layer Perceptron (MLP) Classifier
        5.  Ensemble Learning:
            *   Bagging (e.g., BaggingClassifier with Decision Trees)
            *   Boosting (e.g., AdaBoost, Gradient Boosting, XGBoost)
            *   Stacking (using a combination of diverse base learners and a meta-learner)
    *   Data splitting into training and testing sets.
    *   Hyperparameter tuning for each model (e.g., using GridSearchCV or RandomizedSearchCV).
*   **Model Evaluation:**
    *   Utilize appropriate classification metrics: Accuracy, Precision, Recall, F1-score (macro/weighted averages for multi-class), Confusion Matrix, ROC-AUC score.
    *   Address potential class imbalance in the target variable during modeling and evaluation (e.g., using SMOTE, class weights, or appropriate metrics).
*   **Comparative Analysis:**
    *   Summarize and compare the performance of all trained models.
    *   Discuss trade-offs, strengths, and weaknesses of each approach for this specific problem.
*   **Deliverables:** Separate, well-commented Jupyter notebooks for each model, and a summary of the comparative analysis.

#### 3.2. Out of Scope
*   Real-time prediction system deployment.
*   Development of a user interface (UI) or dashboard.
*   Integration of external data sources beyond the provided CSV.
*   Advanced NLP techniques for the `DESC` column beyond basic parsing for specific keywords/phrases if deemed necessary. Deep semantic understanding of `DESC` is out of scope unless it proves highly valuable with simple methods.

### 4. Data
*   **Source:** Provided CSV file. Sample:
    ```csv
    LOCATION_TEXT,ROAD,WEATHER,LIGHT,DESC,REPORTING_AGENCY,MAIN_CAUSE,INCIDENTDETAILS_ID,DATE_UTC,TIME_UTC,ADDRESS,killed_driver,killed_passenger,killed_pedestrian,injured_driver,injured_passenger,injured_pedestrian,killed_uncategorized,injured_uncategorized,killed_total,injured_total,DATETIME_PST,SEVERITY,Y,X,COLLISION_TYPE
    None,EDSA,None,None,"No Accident Factor, No Collision Stated (based on Police Blotter Book), 0.0 drivers_injured, 0.0 drivers_killed, 0.0 passengers_injured, 0.0 passengers_killed, 0.0 pedestrian_injured, 0.0 pedestrian_killed, Fair (Day), Congressional Ave. Edsa Quezon City",MMDA Road Safety Unit,Human error,2414f311-e805-40c6-b8b9-116b51c35be7,2014-06-30,05:40:00,Congressional Ave. Edsa Quezon City,0,0,0,0,0,0,0,0,0,0,2014-06-30 13:40,Property,14.65771436,121.019788,No Collision Stated
    None,EDSA,None,None,"No Accident Factor, No Collision Stated (based on Police Blotter Book), 0.0 drivers_injured, 0.0 drivers_killed, 0.0 passengers_injured, 0.0 passengers_killed, 0.0 pedestrian_injured, 0.0 pedestrian_killed, Fair (Day), Congressional Ave. near EDSA Exit Gate of SM Warehouse Bahay Toro Quezon City",MMDA Road Safety Unit,Human error,c359744e-e0e7-4541-8db8-5f334c2b4fd7,2014-03-17,01:00:00,Congressional Ave. near EDSA Exit Gate of SM Warehouse Bahay Toro Quezon City,0,0,0,0,0,0,0,0,0,0,2014-03-17 9:00,Property,14.65771436,121.019788,No Collision Stated
    None,EDSA,None,None,"No Accident Factor, No Collision Stated (based on Police Blotter Book), 0.0 drivers_injured, 0.0 drivers_killed, 1.0 passengers_injured, 0.0 passengers_killed, 0.0 pedestrian_injured, 0.0 pedestrian_killed, Fair (Day), Congressional Ave. EDSA Quezon City",MMDA Road Safety Unit,Human error,84e0c352-c029-4393-9252-7516c77f471f,2013-11-26,02:00:00,Congressional Ave. EDSA Quezon City,0,0,0,0,1,0,0,0,0,1,2013-11-26 10:00,Injury,14.65771436,121.019788,No Collision Stated
    ```
*   **Key Contextual Features for Prediction (Initial List - to be refined during EDA):**
    *   `ROAD`
    *   `WEATHER`
    *   `LIGHT`
    *   `MAIN_CAUSE`
    *   Temporal features (derived from `DATE_UTC`, `TIME_UTC`)
    *   `Y`, `X` (Latitude, Longitude)
    *   `COLLISION_TYPE`
    *   `REPORTING_AGENCY`
    *   Potentially, features derived from `DESC` if they offer unique contextual information (e.g., "No Accident Factor").
*   **Features to EXCLUDE as predictors (to avoid data leakage):**
    *   `killed_driver`, `killed_passenger`, `killed_pedestrian`
    *   `injured_driver`, `injured_passenger`, `injured_pedestrian`
    *   `killed_uncategorized`, `injured_uncategorized`
    *   `killed_total`, `injured_total`
    *   `INCIDENTDETAILS_ID` (identifier)
    *   `LOCATION_TEXT` (if redundant with `ADDRESS` or `Y,X`, and often `None`)
    *   `ADDRESS` (if `Y,X` are used and provide sufficient spatial information)
    *   `DATETIME_PST` (if `DATE_UTC`, `TIME_UTC` are used for temporal features)

### 5. Defining "Severity" (Target Variable)

*   **Primary Approach:** Utilize the existing `SEVERITY` column.
    *   Analyze its unique values (e.g., "Property", "Injury" from sample).
    *   Verify its consistency with `killed_total` and `injured_total`. For example:
        *   If `killed_total > 0`, does `SEVERITY` reflect "Fatality"? If not, can we create this category?
        *   If `killed_total == 0` and `injured_total > 0`, does `SEVERITY` reflect "Injury"?
        *   If `killed_total == 0` and `injured_total == 0`, does `SEVERITY` reflect "Property Damage"?
*   **Recommended Target Variable:** A categorical variable with 2 or 3 classes:
    1.  **Property Damage:** No injuries or fatalities.
    2.  **Injury:** One or more injuries, no fatalities.
    3.  **Fatality:** One or more fatalities.
    This may involve mapping or creating this target based on `SEVERITY`, `killed_total`, and `injured_total`. Clearly document the derivation.
*   **Task:** The AI programmer should investigate the `SEVERITY` column and casualty counts (`killed_total`, `injured_total`) to define the most appropriate multi-class target variable. Document the chosen definition and its rationale.

### 6. Functional Requirements (Workflow for AI Pair Programmer)

For each of the specified ML algorithms, a dedicated Jupyter Notebook should be created following this general structure:

#### FR1: Data Loading and Initial Inspection
*   Load the dataset from the provided CSV file into a Pandas DataFrame.
*   Display DataFrame shape, `head()`, `info()`, `describe()`.
*   Perform an initial check for missing values (`isnull().sum()`).

#### FR2: Exploratory Data Analysis (EDA) & Target Variable Definition
*   **Target Variable (`SEVERITY`):**
    *   Investigate the existing `SEVERITY` column.
    *   Analyze `killed_total` and `injured_total` columns.
    *   Define and create the final target variable for classification (e.g., "Property Damage", "Injury", "Fatality"). Justify the definition.
    *   Analyze the class distribution of the target variable. Note any imbalances.
*   **Feature Analysis:**
    *   Analyze each potential predictor column:
        *   Distribution (histograms for numerical, bar plots for categorical).
        *   Number of unique values.
        *   Relationship with the defined target variable (e.g., stacked bar charts, box plots).
    *   Visualize correlations between numerical features and with the target (if encoded).
    *   Examine the `DESC` column:
        *   Assess its content and potential for extracting useful contextual features (e.g., "No Accident Factor", specific light/weather mentions if `LIGHT`/`WEATHER` columns are sparse).
        *   Decide on a strategy: parse specific info, use basic text features, or drop if too noisy/redundant. Document this decision.

#### FR3: Data Preprocessing
*   **Handle Missing Values:**
    *   For columns like `WEATHER`, `LIGHT`, `MAIN_CAUSE`, `COLLISION_TYPE`, devise an imputation strategy (e.g., mode, "Unknown" category, or more advanced imputation if appropriate). Justify the choice.
*   **Feature Engineering:**
    *   Extract temporal features from `DATE_UTC` and `TIME_UTC`:
        *   Hour of day
        *   Day of week
        *   Month
        *   Year (consider its utility - if trends over years are significant)
        *   Season
        *   Weekend/Weekday
    *   If `DESC` parsing is chosen, create new features from it.
*   **Encode Categorical Features:**
    *   Apply appropriate encoding techniques (e.g., One-Hot Encoding for nominal, Ordinal Encoding for ordinal if applicable). Be mindful of dimensionality.
*   **Feature Selection:**
    *   Based on EDA and domain understanding, select the final set of features for modeling. Explicitly drop columns that are identifiers, direct outcomes, or highly redundant.
*   **Scale Numerical Features:**
    *   Apply scaling (e.g., StandardScaler, MinMaxScaler) to numerical features, especially for algorithms like Logistic Regression, MLP, and SVM (if used in Stacking). Tree-based models are generally less sensitive.
*   **Train-Test Split:**
    *   Split the data into training and testing sets (e.g., 80/20 split). Use `stratify` on the target variable if there's class imbalance. Use a fixed `random_state` for reproducibility.

#### FR4: Model Training, Tuning, and Evaluation (Repeated in each dedicated notebook)
*   **Instantiate Model:** Initialize the specific model for the notebook (Logistic Regression, Decision Tree, etc.).
*   **Handle Class Imbalance (if significant):**
    *   Consider techniques like:
        *   Using class weights in model parameters (e.g., `class_weight='balanced'`).
        *   Resampling techniques (e.g., SMOTE for oversampling the minority class on the training set only, or undersampling). Evaluate impact carefully.
*   **Hyperparameter Tuning:**
    *   Implement hyperparameter tuning (e.g., `GridSearchCV` or `RandomizedSearchCV` with cross-validation) to find optimal parameters for the model. Define a reasonable search space for hyperparameters.
*   **Train Model:** Train the model on the (potentially resampled/weighted) training data using the best hyperparameters.
*   **Make Predictions:** Predict on the test set.
*   **Evaluate Model:**
    *   Calculate and report:
        *   Accuracy
        *   Precision, Recall, F1-score (report per-class and macro/weighted averages)
        *   Confusion Matrix (visualize it)
        *   ROC-AUC score (for multi-class, use appropriate averaging like One-vs-Rest or One-vs-One, and plot ROC curves if feasible).
    *   Interpret the results in the context of the problem.

#### FR5: Ensemble Learning Specifics (in their respective notebooks)
*   **Bagging:**
    *   Implement `BaggingClassifier`. Choose a suitable base estimator (e.g., `DecisionTreeClassifier`). Tune parameters like `n_estimators` and `max_samples`.
*   **Boosting:**
    *   Implement at least two boosting algorithms (e.g., `AdaBoostClassifier`, `GradientBoostingClassifier`, `XGBClassifier`). Tune key parameters (e.g., `n_estimators`, `learning_rate`, `max_depth`).
*   **Stacking:**
    *   Implement `StackingClassifier`.
    *   Select a diverse set of base learners (e.g., Logistic Regression, Decision Tree, k-NN, an SVM).
    *   Choose a meta-learner (e.g., Logistic Regression or a simple Random Forest).
    *   Ensure proper cross-validation setup for training base learners to prevent leakage.

#### FR6: Results and Summary (within each notebook and a final comparison)
*   Clearly present the evaluation metrics for the model in that notebook.
*   Optionally, list important features if the model allows (e.g., feature importance from tree-based models, coefficients from logistic regression).

### 7. Comparative Analysis (can be a separate summary section or notebook)
*   Compile the key performance metrics (F1-score, ROC-AUC, Precision, Recall) for all models into a summary table.
*   Visualize the comparison (e.g., bar charts of F1-scores).
*   Discuss the overall findings:
    *   Which model(s) performed best for predicting accident severity?
    *   What were the trade-offs (e.g., training time, interpretability, performance)?
    *   Any insights gained about important predictive features.

### 8. Non-Functional Requirements
*   **Code Quality:**
    *   Write clean, readable, and well-commented Python code adhering to PEP 8 standards.
    *   Use meaningful variable and function names.
    *   Structure code into logical cells in Jupyter notebooks with Markdown explanations for steps and rationale.
*   **Reproducibility:**
    *   Use fixed `random_state` seeds for all operations involving randomness (train-test split, model initialization, resampling).
    *   List versions of key libraries used (e.g., pandas, scikit-learn).
*   **Maintainability:** Code should be organized for easy understanding and future modifications.
*   **Visualization:** Use clear and informative plots (Matplotlib, Seaborn) for EDA and results presentation.

### 9. Technology Stack
*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Pandas (Data manipulation)
    *   NumPy (Numerical operations)
    *   Scikit-learn (Machine learning, preprocessing, metrics)
    *   Matplotlib, Seaborn (Visualization)
*   **Boosting (Optional but Recommended):**
    *   XGBoost
    *   LightGBM
*   **MLP (Optional):**
    *   TensorFlow with Keras API or PyTorch (if Scikit-learn's `MLPClassifier` is deemed insufficient for desired complexity, though `MLPClassifier` should be the first attempt).

### 10. Deliverables
*   A separate Jupyter Notebook (`.ipynb`) for each of the following models:
    1.  Logistic Regression
    2.  Decision Tree
    3.  Random Forest
    4.  MLP
    5.  Ensemble Learning - Bagging
    6.  Ensemble Learning - Boosting (can include multiple boosting algorithms in one notebook or separate ones)
    7.  Ensemble Learning - Stacking
*   Each notebook should include:
    *   Data loading and preprocessing steps (or import from a common utility script/notebook).
    *   Model training, hyperparameter tuning, and evaluation.
    *   Clear explanations, visualizations, and interpretation of results.
*   A summary section or a dedicated notebook for the comparative analysis of all models.
*   Any utility Python scripts created for common functions (e.g., preprocessing).

### 11. Success Metrics for this PRD Implementation
*   All specified machine learning models are successfully implemented, trained, and evaluated.
*   The definition of "accident severity" is well-reasoned and clearly implemented.
*   Data preprocessing steps are appropriate and well-documented.
*   The comparative analysis provides clear insights into model performance.
*   Notebooks are well-structured, commented, and reproducible.
*   Potential pitfalls like data leakage are avoided, and class imbalance is appropriately addressed.

### 12. Notes and Pitfalls to Consider
*   **Data Leakage:** Be extremely careful not to use information that would not be available at the time of prediction as a feature (e.g., casualty counts if they define the target).
*   **Imbalanced Classes:** The `SEVERITY` target variable is likely imbalanced (fewer fatalities than property damage). Employ strategies to handle this (e.g., appropriate metrics, resampling, class weighting) and discuss their impact.
*   **Missing Data Imputation:** The choice of imputation strategy for `WEATHER`, `LIGHT`, etc., can impact model performance. Justify choices and consider sensitivity.
*   **Feature Scaling:** Remember to fit scalers on the training data only and then transform both training and test data.
*   **Hyperparameter Tuning:** This can be computationally intensive. Start with `RandomizedSearchCV` for a broader search or a smaller grid for `GridSearchCV`.
*   **Interpretability vs. Performance:** Some models (e.g., Decision Trees, Logistic Regression) are more interpretable than others (e.g., complex Ensembles, MLP). Discuss this trade-off.
*   **`DESC` Column:** This column is a mix of potentially useful and redundant information. Approach its parsing cautiously. If initial attempts to extract value are complex or yield little improvement, prioritize other features and consider dropping `DESC` or using only very reliably parsed elements.
*   **Cross-Validation:** Use cross-validation consistently, especially during hyperparameter tuning and for robust performance estimation.
