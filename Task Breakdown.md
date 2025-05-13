## Per-Commit Plan for Target Variable Definition

This plan breaks down the PRD into atomic commits, each representing a logical step in the process of investigating, defining, and documenting the "Severity" target variable.

**Assumptions:**
*   The AI programmer will use appropriate tools for data analysis (e.g., Python with Pandas, Jupyter Notebooks for exploration) and version control (Git).
*   Code will be written with maintainability and testing in mind.

---

**Commit 1: Initial Data Loading and Preliminary Inspection**

*   **Commit Message:** `feat: Load dataset and perform initial inspection of severity-related columns`
*   **PRD Points Covered:**
    *   Partially addresses point 2: "Utilize the existing SEVERITY column" (by loading it).
    *   Initiates point 3: "Analyze its unique values" (basic inspection).
*   **Tasks:**
    1.  - [x] Implement a script/notebook cell to load the dataset.
    2.  - [x] Perform basic data inspection:
        *   - [x] Display column names and data types.
        *   - [x] Check for missing values in `SEVERITY`, `killed_total`, and `injured_total`.
        *   - [x] Get a first look at unique values for `SEVERITY` (e.g., `df['SEVERITY'].unique()`).
        *   - [x] Get basic descriptive statistics for `killed_total` and `injured_total`.
*   **Rationale:** Establishes the baseline by loading data and getting a first understanding of the key columns involved.
*   **Potential Files Affected:**
    *   `data_processing/load_data.py`
    *   `notebooks/01_initial_data_exploration.ipynb`

---

**Commit 2: Detailed Analysis of Existing `SEVERITY` Column**

*   **Commit Message:** `feat: Conduct detailed analysis of the existing SEVERITY column's values`
*   **PRD Points Covered:**
    *   Completes point 3: "Analyze its unique values (e.g., "Property", "Injury" from sample)."
*   **Tasks:**
    1.  Generate a frequency distribution of unique values in the `SEVERITY` column.
    2.  Document these unique values and their counts.
    3.  Visually inspect a sample of rows for each unique `SEVERITY` value to understand context, if feasible.
*   **Rationale:** Provides a clear understanding of the current state and diversity of the `SEVERITY` column before consistency checks.
*   **Potential Files Affected:**
    *   `notebooks/02_severity_column_analysis.ipynb`
    *   `reports/severity_column_initial_findings.md`

---

**Commit 3: Implement Consistency Check: `killed_total` vs. `SEVERITY`**

*   **Commit Message:** `feat: Implement consistency check between killed_total and SEVERITY`
*   **PRD Points Covered:**
    *   Addresses point 4: "Verify its consistency with killed_total and injured_total." (Part 1)
    *   Addresses point 5 (first bullet): "If `killed_total > 0`: ... `SEVERITY` *should ideally* indicate "Fatality"."
*   **Tasks:**
    1.  Develop a script/function to filter records where `killed_total > 0`.
    2.  For these records, analyze the corresponding `SEVERITY` values.
    3.  Quantify and document discrepancies (e.g., percentage of records where `killed_total > 0` but `SEVERITY` is not "Fatality" or a similar expected value).
*   **Rationale:** Begins the critical verification process by checking the most severe outcome against the existing `SEVERITY` data.
*   **Potential Files Affected:**
    *   `data_processing/consistency_checks.py`
    *   `notebooks/03_consistency_checks.ipynb`

---

**Commit 4: Implement Consistency Check: `injured_total` (no fatalities) vs. `SEVERITY`**

*   **Commit Message:** `feat: Implement consistency check for injuries (no fatalities) vs. SEVERITY`
*   **PRD Points Covered:**
    *   Addresses point 4: "Verify its consistency with killed_total and injured_total." (Part 2)
    *   Addresses point 5 (second bullet): "If `killed_total == 0` and `injured_total > 0`: ... `SEVERITY` *should ideally* indicate "Injury"."
*   **Tasks:**
    1.  Extend/develop script/function to filter records where `killed_total == 0` AND `injured_total > 0`.
    2.  For these records, analyze the corresponding `SEVERITY` values.
    3.  Quantify and document discrepancies (e.g., percentage of records meeting injury criteria but `SEVERITY` is not "Injury" or similar).
*   **Rationale:** Continues the verification for injury-related incidents.
*   **Potential Files Affected:**
    *   `data_processing/consistency_checks.py` (updated)
    *   `notebooks/03_consistency_checks.ipynb` (updated)

---

**Commit 5: Implement Consistency Check: No Casualties vs. `SEVERITY`**

*   **Commit Message:** `feat: Implement consistency check for no casualties vs. SEVERITY`
*   **PRD Points Covered:**
    *   Addresses point 4: "Verify its consistency with killed_total and injured_total." (Part 3)
    *   Addresses point 5 (third bullet): "If `killed_total == 0` and `injured_total == 0`: ... `SEVERITY` *should ideally* indicate "Property Damage"."
*   **Tasks:**
    1.  Extend/develop script/function to filter records where `killed_total == 0` AND `injured_total == 0`.
    2.  For these records, analyze the corresponding `SEVERITY` values.
    3.  Quantify and document discrepancies (e.g., percentage of records meeting no-casualty criteria but `SEVERITY` is not "Property Damage" or similar).
*   **Rationale:** Completes the consistency checks against the existing `SEVERITY` column. The findings from commits 3-5 will heavily inform the target variable definition.
*   **Potential Files Affected:**
    *   `data_processing/consistency_checks.py` (updated)
    *   `notebooks/03_consistency_checks.ipynb` (updated)
    *   `reports/consistency_check_summary.md`

---

**Commit 6: Define Initial Target Variable Based on Casualty Counts**

*   **Commit Message:** `feat: Define initial target variable based on casualty counts as per recommended classes`
*   **PRD Points Covered:**
    *   Addresses point 6: "Recommended Target Variable: A categorical variable with 2 or 3 classes."
    *   Implements point 7: "Property Damage: No injuries or fatalities."
    *   Implements point 8: "Injury: One or more injuries, no fatalities."
    *   Implements point 9: "Fatality: One or more fatalities."
    *   Partially addresses point 10: "...creating this target based on ... killed_total, and injured_total."
*   **Tasks:**
    1.  Implement a function in a feature engineering script to create a new column (e.g., `derived_severity`).
    2.  Apply the following logic:
        *   If `killed_total > 0`, `derived_severity` = "Fatality".
        *   Else if `killed_total == 0` AND `injured_total > 0`, `derived_severity` = "Injury".
        *   Else (`killed_total == 0` AND `injured_total == 0`), `derived_severity` = "Property Damage".
    3.  Generate this new column for the dataset.
*   **Rationale:** Creates the primary candidate for the target variable based on the objective casualty counts, as recommended by the PRD.
*   **Potential Files Affected:**
    *   `data_processing/feature_engineering.py`
    *   `notebooks/04_target_variable_creation.ipynb`

---

**Commit 7: Investigate Mapping/Refinement of Target Variable using Original `SEVERITY`**

*   **Commit Message:** `feat: Investigate potential mapping or refinement of target variable using original SEVERITY`
*   **PRD Points Covered:**
    *   Addresses point 10: "This may involve mapping or creating this target based on SEVERITY, killed_total, and injured_total." (focus on mapping/using `SEVERITY`)
    *   Initiates point 12: "The AI programmer should investigate... to define the most appropriate multi-class target variable."
*   **Tasks:**
    1.  Compare the newly created `derived_severity` with the original `SEVERITY` column.
    2.  Analyze discrepancies:
        *   Where do they differ?
        *   Does the original `SEVERITY` column offer meaningful distinctions not captured by the count-based `derived_severity` (e.g., different types of "Property Damage" if `derived_severity` is "Property Damage")?
        *   Assess the consistency and reliability of these distinctions in the original `SEVERITY` column, referencing findings from commits 3-5.
    3.  Document the investigation findings and a preliminary decision on whether/how to incorporate information from the original `SEVERITY` column or if the count-based derivation is superior due to consistency.
*   **Rationale:** This is the core investigative step to determine if the original `SEVERITY` column, despite potential inconsistencies, holds valuable information for the final target variable.
*   **Potential Files Affected:**
    *   `notebooks/05_target_refinement_investigation.ipynb`
    *   `reports/target_refinement_analysis.md`

---

**Commit 8: Finalize Target Variable Definition and Document Derivation**

*   **Commit Message:** `docs: Finalize target variable definition, document derivation and rationale`
*   **PRD Points Covered:**
    *   Completes point 10.
    *   Addresses point 11: "Clearly document the derivation."
    *   Completes point 12: "Document the chosen definition and its rationale."
*   **Tasks:**
    1.  Based on the investigation in Commit 7, make the final decision on the target variable's definition (e.g., stick to the 3 classes derived purely from counts, or a modified version if the original `SEVERITY` column proved useful and reliable for certain distinctions).
    2.  If any changes are made to the logic from Commit 6, update the `feature_engineering.py` script.
    3.  Create comprehensive documentation:
        *   State the final, exact definition of the target variable and its classes.
        *   Detail the rules used for its derivation (e.g., "Fatality is defined as `killed_total > 0`").
        *   Explain the rationale for this chosen definition, referencing the consistency checks, the count-based derivation, and the investigation into the original `SEVERITY` column. Justify why this definition is the "most appropriate."
*   **Rationale:** Solidifies the target variable and ensures its creation process is transparent, reproducible, and well-justified.
*   **Potential Files Affected:**
    *   `data_processing/feature_engineering.py` (if updated)
    *   `docs/target_variable_definition.md` (primary documentation artifact)
    *   `README.md` (linking to or summarizing the definition)

---

**Commit 9: Implement Unit Tests for Target Variable Creation Logic**

*   **Commit Message:** `test: Add unit tests for target variable creation function`
*   **PRD Points Covered:**
    *   Implied by general software engineering best practices ("Maintainability: Produce code ready for testing").
*   **Tasks:**
    1.  Write unit tests for the function responsible for creating the final target variable (in `data_processing/feature_engineering.py`).
    2.  Test cases should cover:
        *   Each defined class (Fatality, Injury, Property Damage).
        *   Edge cases (e.g., zero counts for all).
        *   Potentially scenarios with missing `killed_total` or `injured_total` if the data might contain them and a specific handling strategy was decided (though not explicitly in PRD).
*   **Rationale:** Ensures the target variable is consistently and correctly generated according to the finalized definition, preventing regressions.
*   **Potential Files Affected:**
    *   `tests/test_feature_engineering.py`

