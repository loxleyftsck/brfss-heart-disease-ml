# ADAPTA: Advanced Data Analysis & Prediction for Health Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/loxleyftsck/brfss-heart-disease-ml?style=social)](https://github.com/loxleyftsck/brfss-heart-disease-ml)

> 🏥 **Machine Learning for Heart Disease Risk Prediction using CDC BRFSS 2020 Data**

## 📋 Project Overview

**ADAPTA** is an academic data mining project focused on **predicting heart disease risk** using machine learning techniques applied to the **BRFSS 2020** (Behavioral Risk Factor Surveillance System) dataset from the CDC.

### Authors
- **Herald Michain Samuel Theo** (NIM: 215314017)
- **Fera Cisca Wanda Hamid** (NIM: 225314142)

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

## 📈 Results & Performance

### 🎯 Model Performance Comparison

#### Overall Metrics (Test Set: 85,784 samples)

| Model Configuration | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
|:-------------------|----------:|-------:|---------:|---------:|--------:|
| **Random Forest** (Baseline) | 0.18 | **0.81** | 0.30 | 0.66 | 0.74 |
| **Random Forest** (Tuned) | 0.18 | **0.81** | 0.30 | 0.66 | 0.74 |
| **Logistic Regression** (Baseline) | 0.18 | **0.80** | 0.30 | 0.66 | 0.73 |
| **Logistic Regression** (Tuned) | 0.18 | **0.80** | 0.30 | 0.66 | 0.73 |

> 📊 **Key Observation**: Hyperparameter tuning showed **no performance improvement**, indicating:
> - Default parameters were already optimal
> - Current feature set limits model ceiling
> - Linear patterns dominate (RF ≈ LR performance)

---

### 📊 Detailed Performance Breakdown

#### Random Forest vs Logistic Regression

```
┌─────────────────────────────────────────────────────────────┐
│                   PERFORMANCE COMPARISON                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Metric         │  Random Forest  │  Logistic Regression    │
│  ───────────────┼─────────────────┼─────────────────────    │
│  Precision      │      18%        │        18%              │
│  Recall         │      81% ⭐     │        80% ⭐           │
│  F1-Score       │      30%        │        30%              │
│  Accuracy       │      66%        │        66%              │
│  Training Time  │     ~4s         │        ~0.5s            │
│  Interpretability│     Medium     │        High ✓           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Legend: ⭐ = Strong Performance  │  ✓ = Better Choice
```

**Winner**: **Logistic Regression** ✅
- Same accuracy as Random Forest
- **8x faster** training time
- **More interpretable** coefficients
- Simpler model = easier deployment

---

### 🔍 Confusion Matrix Analysis

#### Predictions on 85,784 Test Samples

**Random Forest**:
```
                  Predicted
                  Neg    Pos
Actual  Neg    50,000  28,074  ← 36% False Positive Rate
        Pos     1,469   6,241  ← 19% False Negative Rate
                          ↑
                     81% Recall (Good!)
```

**Key Insights**:
- ✅ Catches **6,241 out of 7,710** heart disease cases (81% recall)
- ⚠️ **28,074 false alarms** (low precision acceptable for screening)
- **Critical**: Only **1,469 missed diagnoses** (19% false negative)

---

### 🏆 Best Model Configuration

**Recommended**: **Logistic Regression (Balanced)**

**Hyperparameters**:
```python
{
    "C": 0.01,              # Regularization strength
    "penalty": "l2",        # L2 regularization
    "solver": "lbfgs",      # Optimizer
    "class_weight": "balanced",  # Handle imbalance
    "random_state": 42      # Reproducibility
}
```

**Why This Model?**:
1. ⚡ **Fast**: 8x faster than Random Forest
2. 📊 **Interpretable**: Clear feature coefficients
3. 🎯 **Effective**: 80% recall for disease detection
4. 🚀 **Simple**: Easy to deploy and maintain

---

### 💡 Key Findings

#### ✅ Strengths
- **High Recall (80-81%)**: Detects 4 out of 5 heart disease cases
- **Consistent Performance**: Both models agree on predictions
- **Production Ready**: Fast inference time (< 1ms per prediction)

#### ⚠️ Trade-offs
- **Low Precision (18%)**: 82% of positive predictions are false alarms
- **Acceptable for Screening**: Better to over-test than miss diagnoses
- **Requires Follow-up**: Positive results need clinical confirmation

#### 🔍 Insights
- **Linear Patterns Dominate**: LR matches RF performance
- **Feature Engineering Needed**: Current 5 features limit ceiling
- **Class Imbalance Handled**: Balanced weights prevent majority-class bias

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

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Centers for Disease Control and Prevention (CDC) for providing the BRFSS dataset
- Course instructors for guidance and support
- Open-source community for Python libraries

---

**Last Updated**: January 2026
