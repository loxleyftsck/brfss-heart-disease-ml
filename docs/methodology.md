# Methodology Documentation

## ADAPTA: Advanced Data Analysis & Prediction for Health Assessment

This document provides detailed technical methodology for the heart disease prediction system.

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Data Description](#data-description)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Selection](#model-selection)
6. [Training Strategy](#training-strategy)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results Interpretation](#results-interpretation)

---

## 1. Problem Definition

### Objective
Develop a binary classification system to predict the presence of **Myocardial Infarction or Coronary Heart Disease** based on behavioral and demographic risk factors.

### Clinical Context
- **Early detection** of heart disease risk can save lives
- **False negatives** (missing a diagnosis) are more dangerous than **false positives**
- System designed for **screening**, not diagnosis
- Healthcare professionals make final decisions

### Research Questions
1. Can machine learning effectively identify heart disease risk from lifestyle data?
2. Which risk factors are most predictive?
3. How do different algorithms compare for this imbalanced medical dataset?

---

## 2. Data Description

### Source
**Behavioral Risk Factor Surveillance System (BRFSS) 2020**
- Collected by Centers for Disease Control and Prevention (CDC)
- Annual telephone health survey of U.S. adults
- Representative sample of population health status

### Dataset Characteristics
- **Size**: ~400,000 individuals
- **Features**: 279 variables (originally)
- **Domain**: Public health, chronic disease surveillance
- **Collection Period**: Calendar year 2020

### Target Variable
**`_MICHD`**: Ever Diagnosed with Angina or Coronary Heart Disease
- **Original encoding**:
  - `1` = Yes (has history)
  - `2` = No (no history)
- **Transformed to**:
  - `1` = Positive case
  - `0` = Negative case

### Selected Predictors

| Variable | Description | Type | Rationale |
|----------|-------------|------|-----------|
| `_SEX` | Gender (1=Male, 2=Female) | Categorical | Known risk factor difference by gender |
| `_AGEG5YR` | Age in 5-year intervals | Ordinal | Age is strongest predictor |
| `_RFSMOK3` | Current smoking status | Binary | Major modifiable risk factor |
| `_RFBMI5` | BMI category (obesity) | Categorical | Obesity linked to heart disease |
| `_TOTINDA` | Physical activity (yes/no) | Binary | Sedentary lifestyle increases risk |

**Feature Selection Criteria**:
- Clinical relevance to cardiovascular health
- Data availability (minimal missing values)
- Modifiable risk factors for intervention potential
- Balance between complexity and interpretability

---

## 3. Preprocessing Pipeline

### Step 1: Data Loading
```python
df = pd.read_csv("brfss2020.csv")
df.columns = df.columns.str.strip().str.upper()
```

**Purpose**: Ensure consistent column naming

### Step 2: Missing Value Handling

**BRFSS Encoding**:
- `7` = "Don't know"
- `9` = "Refused to answer"

**Strategy**: Replace with `NaN` and drop affected rows

```python
df = df.replace([7, 9], pd.NA).dropna()
```

**Justification**:
- Cannot impute clinical data reliably
- Imputation risks introducing bias in medical decisions
- Final dataset still has sufficient samples (>280,000)

**Data Loss**: ~28.5% of records removed

### Step 3: Target Variable Transformation

```python
df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x == 1 else 0)
```

**Result**:
- Positive class (disease): **25,700** (9%)
- Negative class (healthy): **260,246** (91%)
- **Imbalance ratio**: 1:10

### Step 4: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y  # Critical for imbalanced data
)
```

**Configuration**:
- **Train**: 70% (200,162 samples)
- **Test**: 30% (85,784 samples)
- **Stratified**: Preserves 9% positive class in both sets

### Step 5: Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

**Why StandardScaler?**
- Logistic Regression sensitive to feature scales
- Ensures mean=0, std=1 for all features
- Improves optimization convergence

**Data Leakage Prevention**:
- Scaler fitted **only** on training data
- Test data transformed with training statistics
- Prevents information flow from test to train

---

## 4. Feature Engineering

**Note**: This project uses domain-selected features without further engineering.

**Potential Enhancements** (Future Work):
- **Interaction Terms**:
  - Age × Smoking
  - BMI × Physical Activity
- **Polynomial Features**: Age², Age³
- **Binning**: Continuous age into risk groups
- **Feature Selection**: LASSO, Random Forest importance

---

## 5. Model Selection

### Model 1: Random Forest Classifier

**Algorithm**: Ensemble of decision trees with bagging

**Advantages**:
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Less prone to overfitting than single tree

**Disadvantages**:
- "Black box" model (less interpretable)
- Computationally expensive
- May overfit with too many trees

**Default Configuration**:
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
```

### Model 2: Logistic Regression

**Algorithm**: Linear probabilistic classifier

**Advantages**:
- Highly interpretable (coefficients = feature importance)
- Fast training and prediction
- Well-established in medical research
- Outputs probabilities naturally

**Disadvantages**:
- Assumes linear relationship
- May underfit complex patterns
- Sensitive to feature scaling

**Default Configuration**:
```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs'
)
```

---

## 6. Training Strategy

### Class Imbalance Handling

**Problem**: 91% negative, 9% positive → Models predict all as negative for 91% accuracy

**Solution**: `class_weight='balanced'`

**Effect**:
```python
class_weight[0] = n_samples / (n_classes * n_samples_0)
class_weight[1] = n_samples / (n_classes * n_samples_1)
```

- **Negative class weight**: ~0.55
- **Positive class weight**: ~5.5

This **penalizes** misclassifying minority class **10× more**

**Alternative Approaches** (Not Used):
- **SMOTE**: Synthetic minority oversampling (risk: generating unrealistic medical data)
- **Undersampling**: Discard majority class (wastes data)
- **Ensemble methods**: Balanced bagging (higher complexity)

### Training Process

1. **Initialize model** with balanced weights
2. **Fit on training data** (70% of dataset)
3. **Validate** on held-out test set (30%)
4. **No validation set** for hyperparameter tuning (GridSearch uses CV)

---

## 7. Hyperparameter Optimization

### Methodology: GridSearchCV

**Purpose**: Find optimal model configuration

**Process**:
1. Define parameter grid
2. Train model for each combination
3. Evaluate using cross-validation
4. Select best performing set

**Configuration**:
```python
GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring='f1',  # Optimize for F1-score
    cv=5,          # 5-fold cross-validation
    n_jobs=-1      # Parallel processing
)
```

### Random Forest Grid

```python
{
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 10],
    'class_weight': ['balanced']
}
```

**Total Combinations**: 12

### Logistic Regression Grid

```python
{
    'C': [0.01, 0.1, 1, 10],          # Regularization strength
    'penalty': ['l2'],                 # L2 regularization
    'solver': ['lbfgs']               # Optimization algorithm
}
```

**Total Combinations**: 4

### Scoring Metric: F1-Score

**Why F1?**
$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- Balances precision and recall
- More appropriate than accuracy for imbalanced data
- Focuses on positive class performance

---

## 8. Evaluation Metrics

### Confusion Matrix

|               | Predicted Negative | Predicted Positive |
|---------------|--------------------|--------------------|
| **Actual Negative** | True Negative (TN)  | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP)  |

### Primary Metrics

**Accuracy**:
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
- **Limitation**: Misleading for imbalanced data

**Precision** (Positive Predictive Value):
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
- "Of all predicted positive, how many are correct?"
- **Clinical interpretation**: Specificity of screening test

**Recall** (Sensitivity):
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
- "Of all actual positive, how many did we catch?"
- **Clinical interpretation**: Ability to detect disease

**F1-Score**:
$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- Harmonic mean of precision and recall

**ROC-AUC**:
- Area Under Receiver Operating Characteristic curve
- Measures discrimination ability across all thresholds

---

## 9. Results Interpretation

### Model Performance Summary

| Model | Precision | Recall | F1   | Accuracy |
|-------|-----------|--------|------|----------|
| RF (Baseline)  | 0.18 | 0.81 | 0.30 | 0.66 |
| RF (Tuned)     | 0.18 | 0.81 | 0.30 | 0.66 |
| LR (Baseline)  | 0.18 | 0.80 | 0.30 | 0.66 |
| LR (Tuned)     | 0.18 | 0.80 | 0.30 | 0.66 |

### Key Findings

#### 1. High Recall (80-81%)
✅ **Successfully identifies 4 out of 5 individuals with heart disease**

**Clinical Significance**:
- Minimizes **false negatives** (missed diagnoses)
- Appropriate for screening tools
- Aligns with medical practice: "better safe than sorry"

#### 2. Low Precision (18%)
⚠️ **Many false positives** (healthy individuals flagged as at-risk)

**Implications**:
- ~82% of positive predictions are false alarms
- Requires follow-up testing (acceptable in screening context)
- Trade-off: Sensitivity vs. Specificity

**Justification**:
- Missing a heart disease case can be fatal
- False positives lead to further (safer) testing
- This is **standard** in medical screening (e.g., mammograms)

#### 3. Model Similarity
Both Random Forest and Logistic Regression show **nearly identical performance**

**Interpretation**:
- Linear relationships dominate in selected features
- Complex patterns may not exist in current feature set
- Logistic Regression preferred for **interpretability**

**Implications**:
- Use Logistic Regression in production (faster, explainable)
- Current features may be insufficient for higher precision
- Need additional predictors or feature engineering

### Hyperparameter Tuning Impact

**Finding**: Tuning did **not improve** performance

**Possible Reasons**:
- Default parameters already near-optimal
- Dataset characteristics limit ceiling performance
- Feature set is the bottleneck, not model complexity

---

## Limitations & Future Work

### Limitations
1. **Low precision**: High false positive rate
2. **Limited features**: Only 5 predictors used
3. **No feature engineering**: Could explore interactions
4. **Single dataset**: External validation needed
5. **Temporal**: 2020 data may not generalize to other years

### Recommendations for Improvement

1. **Add more features**:
   - Cholesterol levels
   - Blood pressure
   - Family history
   - Diabetes status

2. **Advanced techniques**:
   - SMOTE for balanced sampling
   - XGBoost for gradient boosting
   - Neural networks for complex patterns

3. **Threshold optimization**:
   - Adjust classification threshold
   - Optimize for specific recall target (e.g., 90%)

4. **External validation**:
   - Test on BRFSS 2021+ data
   - Test on different population (e.g., European dataset)

---

## Reproducibility

All experiments use fixed random seeds:
```python
RANDOM_SEED = 42
np.random.seed(42)
```

**Environment**:
- Python 3.9
- scikit-learn 1.3.0
- pandas 2.0.3
- See `requirements.txt` for complete dependencies

---

## References

1. CDC. (2020). Behavioral Risk Factor Surveillance System.
2. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
3. Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
4. Hosmer, D. W., & Lemeshow, S. (2000). Applied Logistic Regression.

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Authors**: Herald M.S. Theo, Fera C.W. Hamid
