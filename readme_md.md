# Mincer Wage Model Analysis

This repository contains the data processing, statistical modeling, and visualizations for analyzing wage determinants using the Mincer Earnings Function. The analysis explores how factors like education, age, social category, and employment sector influence wages, utilizing Ordinary Least Squares (OLS) regression.

## Project Overview

The primary objective of this project is to estimate a Mincer wage equation to understand income disparities and returns to education. The model includes an interaction term between the Social Group and Employment Sector to analyze structural wage differences. Additionally, macro-level employment trends from RBI and Economic Survey data are visualized to provide broader labor market context.

## Repository Structure

```text
├── data/
│   ├── chhv1.csv                  # Household-level survey data
│   ├── cperv1.csv                 # Person-level survey data
│   ├── EmploymentTrend.xlsx       # Macro employment trends (RBI DBIE)
│   └── EcoSurvey2011-12.xlsx      # Public sector employment data
├── outputs/
│   ├── df_output.xlsx             # Cleaned and merged dataset
│   ├── Model_Coefficients.xlsx    # Exported OLS coefficients and p-values
│   └── Final_Regression_Summary.txt # Full statsmodels regression output
├── Mincer_Wage_Model.ipynb        # Main Jupyter/Colab notebook
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Methodology

1.  **Data Wrangling**: 
    *   Merged household (`chhv1.csv`) and person-level (`cperv1.csv`) datasets using composite keys (State, District, FSU, etc.).
    *   Filtered target population to working-age individuals (15-65 years) reporting salaried earnings.
2.  **Feature Engineering**:
    *   Calculated `Log_Wage` as the dependent variable.
    *   Created dummy variables for Education (`Below_Secondary`, `Secondary_Plus`, `Graduate`, `PostGraduate`), Social Category (`General` vs. `SC/ST/OBC`), Sector (`Public` vs. `Private`), and Geography (`Urban_Dummy`).
    *   Added `Age_Sq` to capture the non-linear relationship between age and earnings.
    *   Generated an `Interaction_Term` (Category * Sector).
3.  **Modeling**:
    *   Ran an OLS Regression using `statsmodels`.
    *   Applied clustered standard errors (`cov_type='cluster'`) grouped by State/UT to account for regional correlation.
    *   Conducted post-estimation diagnostics, including Variance Inflation Factor (VIF) checks for multicollinearity.
4.  **Visualization**:
    *   Created scatter plots for Actual vs. Predicted Log Wages.
    *   Visualized wage premiums by Education, Age, and Sector using `seaborn` and `matplotlib`.
    *   Plotted historical macroeconomic employment trends from auxiliary Excel datasets.

## Key Variables

| Variable | Type | Description |
| :--- | :--- | :--- |
| **Log_Wage** | Continuous | Logarithm of Current Wage/Salary (Dependent Variable) |
| **Age / Age_Sq** | Continuous | Age of the individual and its squared term |
| **Category_Dummy** | Binary | 0 = General, 1 = SC/ST/OBC |
| **Sector_Dummy** | Binary | 0 = Public Sector, 1 = Private Sector |
| **Urban_Dummy** | Binary | 0 = Rural, 1 = Urban |
| **Education Dummies**| Binary | Categorical bins for highest education level attained |

## Requirements

To run the notebook locally, you need Python 3.x and the following libraries:

```text
pandas
numpy
statsmodels
patsy
matplotlib
seaborn
```

*You can install these via:* `pip install -r requirements.txt`

## Usage Instructions

1. Clone this repository: `git clone <your-repo-url>`
2. Ensure the data files are placed in the `data/` directory. 
3. **Note on File Paths**: The original script was written in Google Colab and references Google Drive paths (e.g., `/content/drive/MyDrive/...`). If running locally, update the `pd.read_csv` and `pd.read_excel` file paths in the notebook to point to the local `data/` directory.
4. Run the cells in `Mincer_Wage_Model.ipynb` sequentially.