# Project Conventions

This document outlines the conventions to be followed for this project, with a particular focus on Jupyter Notebooks (`.ipynb` files) to ensure consistency, readability, and cross-platform compatibility.

## 1. General Project Conventions

*   **Version Control:**
    *   Use Git for version control.
    *   Commit messages should be clear and follow conventional commit formats (e.g., `feat: ...`, `fix: ...`, `docs: ...`, `style: ...`, `refactor: ...`, `test: ...`, `chore: ...`).
    *   Ensure notebooks are committed in a clean state: outputs cleared, and cells run sequentially from top to bottom before committing, unless the output is essential for understanding the commit.
*   **File and Directory Naming:**
    *   Use `snake_case` for file and directory names (e.g., `my_notebook.ipynb`, `data_processing/`).
    *   Prefix operational notebooks with numbers to indicate order if they are part of a sequence (e.g., `01_data_loading.ipynb`, `02_exploratory_data_analysis.ipynb`).
*   **Python Coding Standards:**
    *   Follow PEP 8 style guidelines for Python code.
    *   Use a linter (e.g., Flake8, Pylint) and a formatter (e.g., Black, autopep8) to maintain consistent code style.
*   **Documentation:**
    *   Maintain a `README.md` in the project root with an overview, setup instructions, and how to run the project.
    *   Document functions, classes, and complex code blocks with clear docstrings and comments.

## 2. Jupyter Notebook (.ipynb) Conventions

The primary goal for notebook conventions is to ensure they are readable, reproducible, and runnable across different environments, specifically **Jupyter Lab** and **Google Colab**.

### 2.1. Notebook Structure and Content

*   **Clear Headings:** Use Markdown cells with clear headings (`#`, `##`, `###`) to structure the notebook logically.
*   **Introduction Cell:** Start each notebook with a Markdown cell that includes:
    *   The notebook's title/purpose.
    *   A brief description of its objectives.
    *   Any prerequisites (e.g., other notebooks to be run, specific data files needed).
    *   Key libraries imported.
*   **Markdown for Explanation:** Use Markdown cells extensively to explain steps, methodologies, reasoning behind choices, and interpretation of results. Code should not stand alone without context.
*   **Logical Flow:** Organize cells in a logical sequence. The notebook should be runnable from top to bottom without errors after restarting the kernel.
*   **Code Comments:** Comment Python code within cells where necessary to explain complex logic.
*   **Concise Cells:** Keep code cells relatively short and focused on a single logical step or a few related steps.

### 2.2. Python in Notebooks

*   **Imports:**
    *   Import all necessary libraries in the first few code cells.
    *   Group imports (standard library, third-party, project-specific).
*   **Reproducibility:**
    *   Set random seeds (e.g., `np.random.seed(42)`, `random.seed(42)`) when stochastic processes are involved to ensure consistent results.
    *   Clearly state versions of key libraries if compatibility issues are anticipated. This can be done in the introductory Markdown cell or by printing versions in a code cell.

### 2.3. Cross-Platform Compatibility (Jupyter Lab & Google Colab)

This is crucial for ensuring that notebooks can be executed in different environments without significant modifications.

*   **Dependency Management:**
    *   List all Python dependencies in a `requirements.txt` file in the project root.
    *   For Google Colab, if specific libraries are not pre-installed, include a cell at the beginning of the notebook to install them using `!pip install -q package_name`. Use the `-q` flag for a quieter installation.
        ```python
        # Example for Colab:
        # try:
        #     import some_package
        # except ImportError:
        #     !pip install -q some_package
        #     import some_package
        ```
    *   Avoid libraries with platform-specific dependencies if possible. If unavoidable, provide clear conditional logic or instructions for each platform.

*   **Data Access:**
    *   **Local Paths (Jupyter Lab):** Use relative paths from the notebook's location to access data files (e.g., `../data/raw/my_data.csv`). Avoid absolute paths.
        ```python
        # Example for local relative path
        # import pandas as pd
        # df = pd.read_csv('../data/raw/your_dataset.csv')
        ```
    *   **Google Colab Data Handling:**
        *   **Google Drive:** Provide clear instructions or code snippets for mounting Google Drive and accessing files.
            ```python
            # Example for Google Drive in Colab
            # from google.colab import drive
            # drive.mount('/content/drive')
            # file_path = '/content/drive/MyDrive/path/to/your_dataset.csv'
            # df = pd.read_csv(file_path)
            ```
        *   **Direct Upload:** For smaller files, users can upload directly to the Colab environment. Mention this as an option.
        *   **Cloud Storage (GCS, S3, etc.):** If using cloud storage, provide authenticated access methods that work in both environments. Colab has helpers for GCS authentication.
    *   **Conditional Path Logic:** Implement logic to switch data paths based on the environment.
        ```python
        # Example for conditional data path
        # import os
        # if 'COLAB_GPU' in os.environ:
        #     # Running in Google Colab
        #     from google.colab import drive
        #     drive.mount('/content/drive', force_remount=True) # Add force_remount=True if needed
        #     data_path = '/content/drive/MyDrive/Colab_Notebooks/TrafficAccidentSeverity/data/raw/RTA_EDSA_2007-2016.csv'
        # else:
        #     # Running locally (Jupyter Lab or other)
        #     data_path = '../data/raw/RTA_EDSA_2007-2016.csv'
        #
        # try:
        #     df = pd.read_csv(data_path)
        #     print(f"Successfully loaded data from: {data_path}")
        # except FileNotFoundError:
        #     print(f"Error: Data file not found at {data_path}")
        #     if 'COLAB_GPU' in os.environ:
        #         print("Please ensure the file exists in your Google Drive at the specified path and that Drive is mounted.")
        #     else:
        #         print("Please ensure the file exists at the specified relative path for your local environment.")

        ```

*   **Environment Variables:**
    *   Avoid hardcoding sensitive information (API keys, credentials).
    *   **Local:** Use `.env` files (added to `.gitignore`) and libraries like `python-dotenv` to load environment variables.
    *   **Google Colab:** Use Colab's "Secrets" feature (recommended) or prompt for input using `getpass.getpass()`. Provide clear instructions.

*   **Magic Commands:**
    *   Be mindful that some magic commands might behave differently or might not be available. Standard magics like `%matplotlib inline`, `%load_ext`, `%run` are generally safe.
    *   Shell commands via `!` (e.g., `!ls`, `!pip install`) work in both but paths might differ.

*   **Output and Visualization:**
    *   Use libraries like Matplotlib, Seaborn, and Plotly, which render well in both Jupyter Lab and Colab.
    *   Ensure plots are displayed inline (`%matplotlib inline` for Matplotlib).
    *   Test visualizations on both platforms to ensure they render correctly.

*   **Authentication for External Services:**
    *   If accessing services like Google BigQuery, AWS services, etc., use authentication methods that are compatible with both environments.
    *   Colab provides `google.colab.auth` for authenticating to Google Cloud services.
    *   For local environments, standard SDK authentication methods (e.g., service accounts, `gcloud auth application-default login`) should be used.

*   **File System Operations:**
    *   Use `os` and `pathlib` modules for platform-agnostic file and path manipulations.
    *   Be aware of the default working directory differences (Colab is typically `/content/`).

### 2.4. Notebook Hygiene

*   **Kernel Restarts:** Regularly restart the kernel and run all cells from top to bottom to ensure reproducibility and catch any state-related bugs.
*   **Clear Outputs:** Before committing, consider clearing all cell outputs to reduce file size and merge conflicts, unless the outputs are critical for understanding the notebook's state at that commit (e.g., specific plots or results for a report). This can be configured in `nbstripout` or similar tools.
*   **Variable Naming:** Use descriptive variable names.
*   **Avoid Long Lines:** Keep code lines to a reasonable length (e.g., <100 characters) for better readability.

## 3. Tools and Configuration

*   **`.gitignore`:** Include a comprehensive `.gitignore` file to exclude virtual environments, IDE/editor configuration files, OS-specific files, data files (unless small and essential for direct execution), and cache files.
    ```
    # Example .gitignore entries
    __pycache__/
    *.py[cod]
    *$py.class

    .env
    *.env

    # Virtual environments
    .venv/
    venv/
    env/

    # IDE and editor specific
    .vscode/
    .idea/
    *.sublime-project
    *.sublime-workspace

    # Jupyter Notebook checkpoints
    .ipynb_checkpoints/

    # Data (if large or private)
    # data/raw/
    # data/processed/

    # OS-specific
    .DS_Store
    Thumbs.db
    ```

*   **(Optional) `nbstripout`:** Consider using `nbstripout` as a pre-commit hook to automatically clear notebook outputs before committing.
    ```bash
    # Installation
    # pip install nbstripout
    # nbstripout --install # To install for the current repo
    ```

By adhering to these conventions, we aim to create a more collaborative, maintainable, and robust project.
