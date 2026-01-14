# ADAPTA: Advanced Data Analysis & Prediction for Health Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/loxleyftsck/brfss-heart-disease-ml?style=social)](https://github.com/loxleyftsck/brfss-heart-disease-ml)

> 🏥 **Machine Learning for Heart Disease Risk Prediction using CDC BRFSS 2020 Data**

## 📋 Project Overview

**ADAPTA** is an academic data mining project focused on **predicting heart disease risk** using machine learning techniques applied to the **BRFSS 2020** (Behavioral Risk Factor Surveillance System) dataset from the CDC.

### Authors
- **Herald Michain Samuel Theo** (NIM: 225314142)
- **Fera Cisca Wanda Hamid** (NIM: 215314017)

### Institution
Data Mining Course - Advanced Machine Learning Applications

---

## 🎯 Project Objectives

1. **Develop predictive models** for heart disease risk assessment
2. **Handle class imbalance** in medical datasets effectively
3. **Compare multiple ML algorithms** (Random Forest vs Logistic Regression)
4. **Optimize model performance** through hyperparameter tuning
5. **Provide interpretable results** for clinical decision support

---

## 📊 Dataset Description

### Source
- **Name**: Behavioral Risk Factor Surveillance System (BRFSS) 2020
- **Provider**: Centers for Disease Control and Prevention (CDC), USA
- **Size**: ~315 MB (331,045,949 bytes)
- **Records**: 400,000+ individuals
- **Features**: 279 variables (demographics, health status, lifestyle)

### Target Variable
- **`_MICHD`**: Myocardial Infarction or Coronary Heart Disease
  - `1`: Has history of heart disease
  - `0`: No history of heart disease

### Selected Features
| Feature | Description | Type |
|---------|-------------|------|
| `_SEX` | Gender | Categorical |
| `_AGEG5YR` | Age group (5-year intervals) | Ordinal |
| `_RFSMOK3` | Smoking status | Binary |
| `_RFBMI5` | BMI category | Categorical |
| `_TOTINDA` | Physical activity | Binary |

### Data Characteristics
- **Class Distribution**:
  - Negative (No disease): 260,246 (91%)
  - Positive (Has disease): 25,700 (9%)
- **Challenge**: Highly imbalanced dataset

---

## 🔬 Methodology

### 1. Data Preprocessing
```
Raw Data → Cleaning → Feature Selection → Normalization → Train/Test Split
```

**Steps**:
- Remove missing values (codes 7, 9 = Don't know/Refused)
- Select relevant health risk factors
- Apply StandardScaler for normalization
- Stratified split (70% train, 30% test)

### 2. Model Development

#### Models Evaluated
1. **Random Forest Classifier**
   - Ensemble method using decision trees
   - Handles non-linear relationships
   - Provides feature importance

2. **Logistic Regression**
   - Linear probabilistic model
   - Fast training and inference
   - Interpretable coefficients

#### Class Imbalance Handling
- Strategy: `class_weight='balanced'`
- Effect: Penalizes misclassification of minority class
- Alternative considered: SMOTE (not used due to risk of overfitting)

### 3. Hyperparameter Optimization

**Random Forest**:
```python
{
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 10],
    'class_weight': ['balanced']
}
```

**Logistic Regression**:
```python
{
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
```

**Optimization Technique**: GridSearchCV with 5-fold cross-validation

---

## 📈 Results

### Model Performance (Class = 1: Heart Disease Positive)

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **Random Forest (Baseline)** | 0.18 | 0.81 | 0.30 | 0.66 |
| **Random Forest (Tuned)** | 0.18 | 0.81 | 0.30 | 0.66 |
| **Logistic Regression (Baseline)** | 0.18 | 0.80 | 0.30 | 0.66 |
| **Logistic Regression (Tuned)** | 0.18 | 0.80 | 0.30 | 0.66 |

### Best Hyperparameters

**Random Forest**:
```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "min_samples_split": 2,
  "class_weight": "balanced"
}
```

**Logistic Regression**:
```json
{
  "C": 0.01,
  "penalty": "l2",
  "solver": "lbfgs"
}
```

### Key Findings

✅ **High Recall (80-81%)**:
- Successfully identifies **4 out of 5** individuals with heart disease
- Critical for medical screening applications (minimize false negatives)

⚠️ **Low Precision (18%)**:
- Many false positives (healthy individuals flagged as at-risk)
- **Trade-off accepted**: In healthcare, missing a diagnosis is worse than over-testing

🔍 **Model Similarity**:
- Both RF and LR show nearly identical performance
- Suggests linear relationships dominate in selected features
- More complex features or interactions may be needed for improvement

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/loxleyftsck/brfss-heart-disease-ml.git
cd brfss-heart-disease-ml
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download dataset**:
```bash
# Place brfss2020.csv in data/raw/
# Or download from CDC: https://www.cdc.gov/brfss/
```

### Quick Start

#### Run Full Pipeline:
```bash
python src/main.py
```

#### Or Step-by-Step in Notebooks:
```bash
jupyter notebook
# Open notebooks/ and run sequentially:
# 01_exploration.ipynb → 02_cleaning.ipynb → ... → 05_evaluation.ipynb
```

---

## 📁 Project Structure

```
adapta-datamining/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
│
├── data/
│   ├── raw/                     # Original dataset (do not modify)
│   │   └── brfss2020.csv
│   ├── processed/               # Cleaned data (generated by code)
│   │   └── train_test_split.pkl
│   └── external/                # External datasets or references
│
├── notebooks/
│   ├── 01_exploration.ipynb     # EDA and data understanding
│   ├── 02_cleaning.ipynb        # Data cleaning pipeline
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb        # Model training
│   └── 05_evaluation.ipynb      # Results and metrics
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load raw data
│   ├── preprocessing.py         # Cleaning and transformation
│   ├── features.py              # Feature engineering
│   ├── models.py                # Model definitions
│   ├── evaluation.py            # Metrics and evaluation
│   ├── utils.py                 # Helper functions
│   └── main.py                  # End-to-end pipeline
│
├── experiments/
│   └── results.csv              # Experiment tracking
│
├── reports/
│   ├── figures/                 # Plots and visualizations
│   │   ├── confusion_matrix_rf.png
│   │   ├── confusion_matrix_lr.png
│   │   └── performance_comparison.png
│   └── final_report.pdf         # Academic report
│
└── docs/
    ├── methodology.md           # Detailed methodology
    └── data_dictionary.md       # Feature descriptions
```

---

## 🔧 Technologies Used

- **Python 3.9**
- **pandas**: Data manipulation
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib/seaborn**: Visualization
- **Jupyter**: Interactive notebooks

---

## 📝 Reproducibility

All random processes are seeded for reproducibility:
```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

---

## 📚 References

1. CDC. (2020). *Behavioral Risk Factor Surveillance System*. https://www.cdc.gov/brfss/
2. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
3. Breiman, L. (2001). "Random Forests". *Machine Learning*, 45(1), 5-32.

---

## 📧 Contact

For questions or collaboration:
- **Herald M.S. Theo**: herald.theo@example.edu
- **Fera C.W. Hamid**: fera.hamid@example.edu

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Centers for Disease Control and Prevention (CDC) for providing the BRFSS dataset
- Course instructors for guidance and support
- Open-source community for Python libraries

---

**Last Updated**: January 2026
