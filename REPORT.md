# 📊 BÁO CÁO DỰ ÁN: HEART STROKE PREDICTION

**Dự án Data Mining - HK251**  
**Ngày báo cáo**: 18/10/2025  
**Repository**: [heart-stroke-data-mining](https://github.com/hothanhnha256/heart-stroke-data-mining)

---

## 📑 MỤC LỤC

1. [Tổng quan Dataset](#1-tổng-quan-dataset)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Feature Selection](#4-feature-selection)
5. [Model Training & Results](#5-model-training--results)
6. [Kết luận và Kiến nghị](#6-kết-luận-và-kiến-nghị)

---

## 1. TỔNG QUAN DATASET

### 1.1 Nguồn dữ liệu

- **Dataset**: Healthcare Dataset Stroke Data (Kaggle)
- **Kích thước**: 5,110 bệnh nhân × 12 thuộc tính
- **Mục tiêu**: Dự đoán nguy cơ đột quỵ (stroke) dựa trên các yếu tố sức khỏe và nhân khẩu học
- **Loại bài toán**: Binary Classification (stroke: 0/1)

- **Loại bài toán**: Binary Classification (stroke: 0/1)

### 1.2 Cấu trúc dữ liệu

| Cột                 | Kiểu    | Mô tả                  | Phạm vi/Giá trị                                          | Missing |
| ------------------- | ------- | ---------------------- | -------------------------------------------------------- | ------- |
| `id`                | int     | Mã định danh bệnh nhân | 1-72940                                                  | 0       |
| `gender`            | object  | Giới tính              | Male, Female, Other                                      | 0       |
| `age`               | float   | Tuổi                   | 0.08-82                                                  | 0       |
| `hypertension`      | int     | Tăng huyết áp          | 0, 1                                                     | 0       |
| `heart_disease`     | int     | Bệnh tim               | 0, 1                                                     | 0       |
| `ever_married`      | object  | Tình trạng hôn nhân    | Yes, No                                                  | 0       |
| `work_type`         | object  | Loại công việc         | Private, Govt_job, Self-employed, Never_worked, children | 0       |
| `Residence_type`    | object  | Nơi cư trú             | Urban, Rural                                             | 0       |
| `avg_glucose_level` | float   | Mức glucose TB (mg/dL) | 55.12-271.74                                             | 0       |
| `bmi`               | float   | Chỉ số khối cơ thể     | 10.3-97.6                                                | **201** |
| `smoking_status`    | object  | Tình trạng hút thuốc   | formerly smoked, never smoked, smokes, Unknown           | 0       |
| **`stroke`**        | **int** | **Target - Đột quỵ**   | **0, 1**                                                 | **0**   |

### 1.3 Vấn đề chính của Dataset

#### ⚠️ **Problem 1: CLASS IMBALANCE nghiêm trọng**

```
No Stroke (0): 4,861 cases (95.13%)
Stroke (1):      249 cases (4.87%)
────────────────────────────────────
Tỷ lệ:         19.5 : 1
```

**Impact**:

- Model có thể "học" cách dự đoán tất cả là "No Stroke" và vẫn đạt 95% accuracy
- Cần metrics phù hợp: F1-Score, Precision, Recall (không chỉ Accuracy)
- Cần kỹ thuật xử lý: SMOTE, class weights, stratified sampling

#### ⚠️ **Problem 2: Missing Values**

```
BMI: 201 giá trị thiếu (3.93%)
```

**Impact**:

- Không thể loại bỏ vì mất 4% dữ liệu
- Cần imputation strategy thích hợp

#### ⚠️ **Problem 3: Outliers**

```
BMI:
  - Median: 28.1
  - Max: 97.6 (không hợp lý y học!)

avg_glucose_level:
  - Median: 91.9 mg/dL
  - Max: 271.7 mg/dL (có thể hợp lý trong trường hợp đặc biệt)
```

**Impact**:

- Có thể làm méo model
- Cần outlier capping/removal

---

## 2. EXPLORATORY DATA ANALYSIS (EDA)

### 2.1 Phân tích biến Target

**Phân phối Stroke:**

| Class         | Count | Percentage |
| ------------- | ----- | ---------- |
| No Stroke (0) | 4,861 | 95.13%     |
| Stroke (1)    | 249   | 4.87%      |

**Visualizations**: `eda/eda_target_distribution.png`

### 2.2 Phân tích Numeric Features

#### **Age (Tuổi)**

```
Count: 5,110
Mean:  43.23 years
Std:   22.61 years
Min:   0.08 years (infant)
25%:   25 years
50%:   45 years
75%:   61 years
Max:   82 years
```

**Insight**:

- Phân phối wide range từ infant đến 82 tuổi
- Stroke risk tăng mạnh theo tuổi (see age group analysis below)

#### **Average Glucose Level**

```
Count: 5,110
Mean:  106.15 mg/dL
Std:   45.28 mg/dL
Min:   55.12 mg/dL
Median: 91.89 mg/dL
Max:   271.74 mg/dL
```

**Insight**:

- Normal range: 70-100 mg/dL (fasting)
- Nhiều cases có glucose cao (pre-diabetes/diabetes)
- Có thể là risk factor quan trọng

#### **BMI (Body Mass Index)**

```
Count: 4,909 (201 missing)
Mean:  28.89
Std:   7.85
Min:   10.3 (underweight severe)
Median: 28.1
Max:   97.6 (outlier!)
```

**Insight**:

- Mean BMI = 28.89 → "Overweight" category
- Max = 97.6 là outlier rõ ràng (cần xử lý)

**Visualizations**: `eda/eda_numeric_analysis.png`

### 2.3 Phân tích Categorical Features

#### **Gender Distribution**

| Gender | Count | Stroke Count | Stroke Rate |
| ------ | ----- | ------------ | ----------- |
| Female | 2,994 | 141          | 4.71%       |
| Male   | 2,115 | 108          | 5.11%       |
| Other  | 1     | 0            | 0%          |

**Insight**: Stroke rate tương đương giữa Male/Female

#### **Marital Status**

| Status | Count | Stroke Count | Stroke Rate |
| ------ | ----- | ------------ | ----------- |
| Yes    | 3,353 | 220          | **6.56%**   |
| No     | 1,757 | 29           | **1.65%**   |

**Insight**: ⚠️ **Người đã kết hôn có stroke rate cao gấp 4 lần!**  
(Có thể do correlation với age - người lớn tuổi thường đã kết hôn)

#### **Work Type**

| Work Type     | Count | Stroke Count | Stroke Rate |
| ------------- | ----- | ------------ | ----------- |
| Self-employed | 819   | 65           | **7.94%**   |
| Private       | 2,925 | 149          | 5.09%       |
| Govt_job      | 657   | 33           | 5.02%       |
| Children      | 687   | 2            | **0.29%**   |
| Never_worked  | 22    | 0            | 0%          |

**Insight**: Self-employed có stroke rate cao nhất

#### **Residence Type**

| Type  | Count | Stroke Count | Stroke Rate |
| ----- | ----- | ------------ | ----------- |
| Urban | 2,596 | 135          | 5.20%       |
| Rural | 2,514 | 114          | 4.53%       |

**Insight**: Không có sự khác biệt lớn

#### **Smoking Status**

| Status          | Count | Stroke Count | Stroke Rate |
| --------------- | ----- | ------------ | ----------- |
| formerly smoked | 885   | 70           | **7.91%**   |
| smokes          | 789   | 42           | 5.32%       |
| never smoked    | 1,892 | 90           | 4.76%       |
| Unknown         | 1,544 | 47           | 3.04%       |

**Insight**: "Formerly smoked" có rate cao nhất (có thể do age factor)

**Visualizations**: `eda/eda_categorical_analysis.png`

### 2.4 Correlation Analysis

**Top correlations với Stroke (theo absolute value):**

| Feature               | Correlation | Ý nghĩa                |
| --------------------- | ----------- | ---------------------- |
| **age**               | **0.2453**  | ⭐⭐⭐ Quan trọng nhất |
| **heart_disease**     | **0.1349**  | ⭐⭐ Quan trọng        |
| **avg_glucose_level** | **0.1319**  | ⭐⭐ Quan trọng        |
| **hypertension**      | **0.1279**  | ⭐⭐ Quan trọng        |
| **ever_married**      | **0.1083**  | ⭐ Có ảnh hưởng        |
| bmi                   | 0.0361      | Ảnh hưởng nhỏ          |
| work_type             | 0.0323      | Ảnh hưởng nhỏ          |
| smoking_status        | 0.0281      | Ảnh hưởng nhỏ          |
| Residence_type        | 0.0155      | Gần như không          |
| gender                | 0.0089      | Gần như không          |

**Visualizations**: `eda/eda_correlation_matrix.png`

### 2.5 Age Group Analysis

**Stroke Rate theo nhóm tuổi:**

| Age Group | Total | Stroke Count | Stroke Rate  |
| --------- | ----- | ------------ | ------------ |
| <30       | 1,570 | 2            | **0.13%**    |
| 30-50     | 1,413 | 21           | **1.49%**    |
| 50-65     | 1,162 | 70           | **6.02%**    |
| **65+**   | 965   | 156          | **16.17%** � |

**Key Insights:**

- 🎯 **Age là predictor mạnh nhất**
- Stroke rate tăng **exponentially** với tuổi
- Nhóm 65+ có risk cao gấp **127 lần** so với <30
- **Implication**: Age phải là feature quan trọng trong model

**Visualizations**: `eda/eda_age_analysis.png`

### 2.6 Tóm tắt EDA Insights

✅ **Top Risk Factors** (theo thứ tự quan trọng):

1. **Age** (tuổi cao)
2. **Heart Disease** (bệnh tim)
3. **High Glucose** (đường huyết cao)
4. **Hypertension** (tăng huyết áp)
5. **Marital Status** (đã kết hôn - proxy cho age)

❌ **Weak Factors**:

- Gender (correlation gần 0)
- Residence type (Urban vs Rural không khác biệt)

---

## 3. PREPROCESSING PIPELINE

### 3.1 Kiến trúc Pipeline

Chúng tôi xây dựng preprocessing pipeline với **sklearn** sử dụng `ColumnTransformer`:

```python
# Feature categorization
target_col = "stroke"
drop_cols = ["id"]  # Không có giá trị dự đoán

numeric_cols = ["age", "avg_glucose_level", "bmi"]
binary_cols = ["hypertension", "heart_disease"]
categorical_cols = ["gender", "ever_married", "work_type",
                   "Residence_type", "smoking_status"]
```

**Pipeline Flow:**

```
Raw CSV (12 columns)
    ↓
[1] Missing Value Handling
    ↓
[2] Outlier Capping (optional)
    ↓
[3] Train/Test Split (stratified)
    ↓
[4] Feature Encoding & Scaling
    ↓
[5] SMOTE Balancing (train only)
    ↓
Processed Data (21 features)
```

### 3.2 Chi tiết từng bước

#### **Bước 1: Missing Value Handling**

**Problem**: BMI có 201 giá trị thiếu (3.93%)

**Solution**:

```python
# Quick imputation trước khi outlier capping
if df["bmi"].isna().any():
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Safety imputer trong pipeline
SimpleImputer(strategy="median")  # Cho numeric
SimpleImputer(strategy="most_frequent")  # Cho categorical
```

**Rationale**:

- Median robust hơn mean với outliers
- Most frequent giữ được distribution của categorical

#### **Bước 2: Outlier Treatment**

**Method**: IQR-based capping

```python
def cap_outliers_iqr(s, whisker=1.5):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return s.clip(lower=lower, upper=upper)
```

**Applied to**:

- `bmi`: Max 97.6 → capped to ~45
- `avg_glucose_level`: Extreme values capped

**Rationale**:

- Không loại bỏ data points
- Giữ được information nhưng giảm impact của outliers
- Standard statistical method

#### **Bước 3: Train/Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80-20 split
    stratify=y,         # Preserve 4.87% stroke rate
    random_state=42     # Reproducibility
)
```

**Results**:

- Training: 4,088 samples
- Test: 1,022 samples
- Both maintain ~4.87% stroke rate

**Rationale**: Stratification critical với imbalanced data

#### **Bước 4: Feature Encoding & Scaling**

**A. Numeric Pipeline**:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())  # Mean=0, Std=1
])
```

**B. Categorical Pipeline**:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",  # Quan trọng cho production!
        sparse_output=False
    ))
])
```

**C. Binary Pipeline**:

```python
"passthrough"  # Giữ nguyên 0/1
```

**Combined Transformer**:

```python
ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
    ("bin", "passthrough", binary_cols)
], remainder="drop")
```

**Feature Transformation**:

| Before         | After                                          | Example        |
| -------------- | ---------------------------------------------- | -------------- |
| 12 columns     | 21 features                                    | -              |
| `gender`       | `gender_Female`, `gender_Male`, `gender_Other` | OneHot         |
| `age`          | `age` (scaled)                                 | StandardScaler |
| `hypertension` | `hypertension`                                 | Passthrough    |

**⚠️ Critical**: Fit chỉ trên training set!

```python
# ✅ CORRECT
preprocessor.fit(X_train)  # Learn từ train only
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)  # Apply same transform

# ❌ WRONG - Data Leakage!
preprocessor.fit(X)  # Information leak từ test→train
```

#### **Bước 5: SMOTE Oversampling**

**Problem**: Training set có 4.87% stroke (197 positive / 3,891 negative)

**Solution**: SMOTE (Synthetic Minority Oversampling Technique)

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

**Results**:

| Metric        | Before SMOTE   | After SMOTE     |
| ------------- | -------------- | --------------- |
| Total samples | 4,088          | **7,778**       |
| Stroke (1)    | 197 (4.82%)    | **3,889 (50%)** |
| No Stroke (0) | 3,891 (95.18%) | **3,889 (50%)** |

**How SMOTE works**:

1. Lấy minority class samples
2. Tìm k-nearest neighbors
3. Tạo synthetic samples giữa sample và neighbors
4. Balance classes

**⚠️ Critical**: Chỉ apply trên TRAINING set!

```python
# ✅ CORRECT
sm.fit_resample(X_train, y_train)  # Train only
# Test set giữ nguyên distribution

# ❌ WRONG
sm.fit_resample(X, y)  # Before split = data leakage!
```

### 3.3 Output Artifacts

Pipeline tạo ra các files trong `data-pre/`:

```
data-pre/
├── train_preprocessed.csv        # 7,778 × 22 (21 features + target)
├── test_preprocessed.csv         # 1,022 × 22
├── preprocessor.joblib           # Fitted sklearn pipeline
├── feature_names.txt             # List of 21 feature names
└── prep_meta.json               # Metadata
```

**prep_meta.json example**:

```json
{
  "n_train": 7778,
  "n_test": 1022,
  "pos_rate_train": 0.5, // Balanced!
  "pos_rate_test": 0.0487, // Original distribution
  "n_features": 21,
  "scale": "standard",
  "cap_outliers": true,
  "smote": true
}
```

### 3.4 Command Usage

```powershell
python prepare-stroke.py `
  --input data-raw/healthcare-dataset-stroke-data.csv `
  --output-dir data-pre `
  --test-size 0.2 `
  --scale standard `
  --cap-outliers `
  --smote `
  --random-state 42
```

**Execution time**: ~5 seconds

---

## 4. FEATURE SELECTION

### 4.1 Methodology

Chúng tôi sử dụng **4 phương pháp độc lập** và kết hợp kết quả:

```
┌─────────────────────┐
│  1. Correlation     │ → Pearson correlation
│  2. Mutual Info     │ → Information Gain
│  3. RF Importance   │ → Tree-based
│  4. Statistical     │ → ANOVA + Chi-square
└─────────────────────┘
          ↓
    Normalize [0,1]
          ↓
   Average scores
          ↓
   Combined Ranking
```

### 4.2 Method Details

#### **Method 1: Correlation Analysis**

```python
correlations = df.corr()['stroke'].abs()
```

**Top 5 results**:

| Feature           | Correlation | Interpretation  |
| ----------------- | ----------- | --------------- |
| age               | 0.2453      | Strong positive |
| heart_disease     | 0.1349      | Moderate        |
| avg_glucose_level | 0.1319      | Moderate        |
| hypertension      | 0.1279      | Moderate        |
| ever_married      | 0.1083      | Weak-Moderate   |

**Visualization**: `feature/feature_correlation_analysis.png`

#### **Method 2: Mutual Information**

```python
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, y, random_state=42)
```

**Top 5 results**:

| Feature      | MI Score | Interpretation    |
| ------------ | -------- | ----------------- |
| age          | 0.0348   | Highest info gain |
| bmi          | 0.0112   | Moderate          |
| ever_married | 0.0093   | Moderate          |
| hypertension | 0.0091   | Moderate          |
| work_type    | 0.0074   | Low               |

**Insight**: BMI ranks higher in MI than correlation

**Visualization**: `feature/feature_mutual_info_analysis.png`

#### **Method 3: Random Forest Importance**

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42,
                            class_weight='balanced')
rf.fit(X, y)
importance = rf.feature_importances_
```

**Top 5 results**:

| Feature           | Importance | Interpretation |
| ----------------- | ---------- | -------------- |
| age               | 0.3840     | Dominant!      |
| avg_glucose_level | 0.2027     | Important      |
| bmi               | 0.1844     | Important      |
| smoking_status    | 0.0482     | Minor          |
| work_type         | 0.0477     | Minor          |

**Insight**: RF heavily weights age (38.4% của total importance!)

**Visualization**: `feature/feature_rf_importance_analysis.png`

#### **Method 4: Statistical Tests**

**For Numeric** (ANOVA F-test):

```python
from sklearn.feature_selection import f_classif
f_scores, p_values = f_classif(X_numeric, y)
```

| Feature           | F-score | p-value        |
| ----------------- | ------- | -------------- |
| age               | 326.92  | < 0.001 \*\*\* |
| avg_glucose_level | 90.50   | < 0.001 \*\*\* |
| bmi               | 6.67    | 0.0098 \*\*    |

**For Categorical** (Chi-square):

```python
from sklearn.feature_selection import chi2
chi2_scores, p_values = chi2(X_categorical, y)
```

| Feature        | χ² score | p-value        |
| -------------- | -------- | -------------- |
| heart_disease  | 87.99    | < 0.001 \*\*\* |
| hypertension   | 75.45    | < 0.001 \*\*\* |
| ever_married   | 20.62    | < 0.001 \*\*\* |
| smoking_status | 3.37     | 0.0664         |
| work_type      | 2.93     | 0.0872         |

**Visualization**: `feature/feature_statistical_analysis.png`

### 4.3 Combined Ranking

**Normalization Process**:

```python
# Min-Max normalize each method's scores to [0,1]
normalized = (score - min) / (max - min)

# Average across 4 methods
combined_score = mean([corr, mi, rf_imp, stat])
```

**Final Top 8 Features**:

| Rank | Feature               | Combined Score | Comment                 |
| ---- | --------------------- | -------------- | ----------------------- |
| 🥇 1 | **age**               | **1.0000**     | ⭐⭐⭐⭐⭐ CRITICAL     |
| 🥈 2 | **avg_glucose_level** | **0.3636**     | ⭐⭐⭐⭐ Very Important |
| 🥉 3 | **hypertension**      | **0.2471**     | ⭐⭐⭐ Important        |
| 4    | **heart_disease**     | **0.2428**     | ⭐⭐⭐ Important        |
| 5    | **bmi**               | **0.2198**     | ⭐⭐⭐ Important        |
| 6    | **ever_married**      | **0.1905**     | ⭐⭐ Moderate           |
| 7    | **work_type**         | **0.0898**     | ⭐ Minor                |
| 8    | **smoking_status**    | **0.0505**     | ⭐ Minor                |

**Dropped** (low scores):

- `Residence_type` (0.0239)
- `gender` (0.0009)

**Visualization**: `feature/feature_combined_ranking.png`

### 4.4 Key Insights

✅ **Age dominates**:

- Rank #1 in ALL 4 methods
- 10x more important than next feature
- Should definitely be included

✅ **Health metrics critical**:

- glucose, hypertension, heart_disease all rank high
- Reflects medical knowledge (expected)

✅ **BMI underrated by correlation**:

- Low Pearson correlation (0.036)
- But high in MI and RF importance
- **Non-linear relationship** with stroke!

❌ **Lifestyle factors weak**:

- Smoking: Lower than expected
- Work type: Minimal impact
- Possibly confounded by age

❌ **Demographic factors irrelevant**:

- Gender: Nearly zero importance
- Residence: Urban vs Rural không khác biệt

### 4.5 Recommendations

**For Modeling**:

1. **Must include**: age, avg_glucose_level, hypertension, heart_disease, bmi
2. **Consider**: ever_married (age proxy)
3. **Optional**: work_type, smoking_status
4. **Can drop**: gender, Residence_type

**Feature Engineering Ideas**:

- Age groups/bins (categorical)
- BMI categories (underweight/normal/overweight/obese)
- Glucose categories (normal/pre-diabetes/diabetes)
- Interaction features: age × heart_disease, age × hypertension

---

## 5. MODEL TRAINING & RESULTS

- **Đặc biệt**: BMI được impute trước để có thể xử lý outliers

#### 3.2.2 Xử lý Outliers

- **Phương pháp**: IQR-based capping với `whisker=1.5`
- **Áp dụng cho**: `bmi` và `avg_glucose_level`
- **Công thức**: `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

```python
def cap_outliers_iqr(s: pd.Series, whisker: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.clip(lower=q1 - whisker*iqr, upper=q3 + whisker*iqr)
```

#### 3.2.3 Encoding Categorical Variables

- **Phương pháp**: `OneHotEncoder(handle_unknown='ignore')`
- **Kết quả**: Từ 5 categorical columns → 14 encoded features
- **Ví dụ**: `gender` → `gender_Female`, `gender_Male`, `gender_Other`

#### 3.2.4 Feature Scaling

- **Options**: StandardScaler (default), MinMaxScaler, hoặc không scale
- **Áp dụng cho**: Chỉ numeric features
- **Lý do**: Binary features giữ nguyên (0/1)

#### 3.2.5 Train/Test Split

- **Phương pháp**: `train_test_split` với `stratify=y`
- **Tỷ lệ**: 80/20 (có thể cấu hình)
- **Random state**: 42 (reproducibility)

#### 3.2.6 Xử lý Class Imbalance

- **Phương pháp**: SMOTE (Synthetic Minority Oversampling Technique)
- **Thư viện**: `imbalanced-learn`
- **Áp dụng**: Chỉ trên training set
- **Kết quả**: Cân bằng tỷ lệ 50/50

### 3.3 Artifacts được tạo

1. `preprocessor.joblib`: Sklearn pipeline đã fit
2. `train_preprocessed.csv`, `test_preprocessed.csv`: Dữ liệu đã transform
3. `feature_names.txt`: Danh sách features sau encoding
4. `prep_meta.json`: Metadata và thống kê

**📋 Metadata ví dụ:**

```json
{
  "n_train": 7778, // Sau SMOTE
  "n_test": 1022,
  "pos_rate_train": 0.5, // Cân bằng sau SMOTE
  "pos_rate_test": 0.049, // Giữ nguyên phân phối gốc
  "n_features": 21
}
```

---

## 4. FEATURE SELECTION

### 4.1 Phương pháp áp dụng

Chúng tôi sử dụng **4 phương pháp** kết hợp:

1. **Correlation Analysis**: Tương quan Pearson với target
2. **Mutual Information**: Information gain giữa features và target
3. **Random Forest Importance**: Feature importance từ tree-based model
4. **Statistical Tests**: ANOVA F-test (numeric) + Chi-square (categorical)

### 4.2 Quy trình feature selection

```python
# 1. Normalize tất cả scores về [0,1]
# 2. Tính combined_score = average của 4 phương pháp
# 3. Rank features theo combined_score
# 4. Chọn top K features
```

### 4.3 Kết quả feature selection

**🏆 Top 8 Features quan trọng nhất:**

1. **age**: Yếu tố quan trọng nhất (tuổi)
2. **avg_glucose_level**: Mức glucose trung bình
3. **bmi**: Chỉ số khối cơ thể
4. **hypertension**: Tăng huyết áp
5. **heart_disease**: Bệnh tim
6. **work*type*\***: Một số loại công việc cụ thể
7. **ever*married*\***: Tình trạng hôn nhân
8. **smoking*status*\***: Tình trạng hút thuốc

**📊 Kết quả chi tiết**: Xem `feature_selection.py` và `feature_selection_results.json`

### 4.4 Insights từ Feature Selection

- **Age dominates**: Tuổi là predictor mạnh nhất
- **Health indicators**: Các chỉ số sức khỏe (glucose, BMI, blood pressure) quan trọng
- **Lifestyle factors**: Hút thuốc, hôn nhân có ảnh hưởng nhưng ít hơn
- **Gender**: Không nằm trong top features

---

## 5. MODEL TRAINING & RESULTS

### 5.1 Models Overview

Chúng tôi đã triển khai và so sánh **4 models** khác nhau:

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **SVM (RBF kernel)** - Support Vector Machine với kernel phi tuyến
4. **K-Nearest Neighbors (k=5)** - Instance-based learning

**Training Data**: 7,778 samples (balanced 50-50 sau SMOTE)  
**Test Data**: 1,022 samples (original distribution: 95.1% vs 4.9%)

### 5.2 Results Summary

#### **📊 Performance Metrics Table**

| Model                      | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Logistic Regression** ⭐ | **0.7495** | **0.1399** | **0.8000** | **0.2381** | **0.8456** |
| Random Forest              | 0.9266     | 0.1795     | 0.1400     | 0.1573     | 0.7615     |
| SVM (RBF)                  | 0.7945     | 0.1040     | 0.4200     | 0.1667     | 0.7648     |
| KNN (k=5)                  | 0.8415     | 0.0942     | 0.2600     | 0.1383     | 0.6202     |

**Visualization**: `model_metrics_comparison.png`

#### **🏆 Rankings**

**Theo F1-Score** (chính):

1. 🥇 **Logistic Regression**: 0.2381
2. 🥈 SVM (RBF): 0.1667
3. 🥉 Random Forest: 0.1573
4. KNN (k=5): 0.1383

**Theo ROC-AUC**:

1. 🥇 **Logistic Regression**: 0.8456
2. 🥈 SVM (RBF): 0.7648
3. 🥉 Random Forest: 0.7615
4. KNN (k=5): 0.6202

**Theo Recall** (quan trọng cho medical):

1. 🥇 **Logistic Regression**: 0.8000 (40/50 cases detected)
2. 🥈 SVM (RBF): 0.4200 (21/50)
3. 🥉 KNN (k=5): 0.2600 (13/50)
4. Random Forest: 0.1400 (7/50) ⚠️

### 5.3 Detailed Analysis

#### **Model 1: Logistic Regression** ⭐ BEST

**Confusion Matrix**:

```
              Predicted
              No    Yes
Actual No    726    246
       Yes    10     40
```

**Strengths**:

- ✅ **Best Recall (0.80)**: Detected 40/50 stroke cases → chỉ miss 10
- ✅ **Best ROC-AUC (0.8456)**: Excellent discrimination ability
- ✅ **Best F1-Score (0.2381)**: Best balance precision-recall
- ✅ **Low False Negatives (10)**: Critical cho medical context

**Weaknesses**:

- ❌ **Low Precision (0.14)**: 246 false alarms
- ❌ **High False Positives**: Nhiều người không có stroke bị dự đoán nhầm

**Medical Interpretation**:

- **Sensitivity**: 80% strokes detected → Excellent for screening
- **Trade-off**: "Over-alert" model → better safe than sorry

#### **Model 2: Random Forest**

**Confusion Matrix**:

```
              Predicted
              No    Yes
Actual No    940     32
       Yes    43      7
```

**Strengths**:

- ✅ **Highest Accuracy (0.93)**: Best overall correctness
- ✅ **Low False Positives (32)**: Fewest false alarms

**Weaknesses**:

- ❌ **Worst Recall (0.14)**: Chỉ detect 7/50 → **MISS 43 cases!** ⚠️
- ❌ **Not suitable for medical**: Too many missed strokes (86%)

#### **Model 3: SVM (RBF)**

**Confusion Matrix**:

```
              Predicted
              No    Yes
Actual No    791    181
       Yes    29     21
```

**Analysis**:

- Moderate performance across all metrics
- Better than RF về recall nhưng worse than LogReg
- Computationally expensive without significant improvement

#### **Model 4: KNN (k=5)**

**Confusion Matrix**:

```
              Predicted
              No    Yes
Actual No    847    125
       Yes    37     13
```

**Analysis**:

- Worst ROC-AUC (0.62) → Poor discrimination
- 74% strokes missed (37/50)
- Not recommended for production

### 5.4 Model Selection

#### **🏆 Recommended: Logistic Regression**

**Lý do chọn**:

1. **Best F1-Score (0.2381)** - Highest overall performance
2. **Best Recall (0.80)** - Critical cho medical diagnosis
3. **Best ROC-AUC (0.8456)** - Best discrimination ability
4. **Interpretable** - Linear coefficients hiểu được
5. **Fast** - Training + prediction rất nhanh

**Production Strategy**:

```python
# Option 1: High Sensitivity (Current)
threshold = 0.5
recall = 0.80  # 80% detection
precision = 0.14  # Acceptable false alarms

# Option 2: Adjusted Threshold
threshold = 0.3  # Lower threshold
expected_recall = 0.90+  # 90%+ detection
expected_precision = 0.08-0.10  # More false alarms
```

**Medical Justification**:

- False Positives → Extra tests (acceptable cost)
- False Negatives → Missed diagnosis (unacceptable!)
- **Better safe than sorry**

### 5.5 Evaluation Metrics Explanation

**Why F1-Score?**:

- Harmonic mean của Precision & Recall
- Balanced metric cho imbalanced data
- Penalizes extreme imbalance

**Why ROC-AUC?**:

- Measures discrimination across all thresholds
- Robust to class imbalance
- 0.8456 = "Good" classification

**Why NOT Accuracy?**:

- 95% No Stroke → predict tất cả "No Stroke" = 95% accuracy
- Hoàn toàn vô dụng!
- Misleading metric cho imbalanced data

---

## 6. KẾT LUẬN VÀ KIẾN NGHỊ

### 6.1 Tóm tắt Thành quả

✅ **Dataset Understanding**:

- Analyzed 5,110 patients với class imbalance nghiêm trọng (95.1% vs 4.9%)
- Xác định age là predictor mạnh nhất (correlation 0.2453)
- Phát hiện BMI có non-linear relationship với stroke
- Age group 65+ có risk cao gấp 127 lần so với <30

✅ **Preprocessing Pipeline**:

- Xử lý missing values: Median imputation cho BMI (201 missing)
- Outlier capping: IQR method cho BMI và glucose
- Feature encoding: 12 columns → 21 features
- SMOTE balancing: Train set balanced 50-50 (7,778 samples)
- **Zero data leakage**: Fit preprocessor on train only

✅ **Feature Selection**:

- Multi-method approach: 4 independent methods
- Top 8 features identified: age, glucose, hypertension, heart_disease, bmi, ever_married, work_type, smoking
- Validates medical knowledge: Age và health metrics dominate
- Gender và Residence_type có thể drop (minimal impact)

✅ **Model Training**:

- Implemented 4 models: LogReg, Random Forest, SVM, KNN
- Best model: **Logistic Regression**
  - F1-Score: 0.2381 (best)
  - Recall: 0.80 (80% stroke detection)
  - ROC-AUC: 0.8456 (excellent discrimination)
- Model comparison với comprehensive visualizations

✅ **Production-Ready**:

- Complete pipeline từ raw data → predictions
- Reproducible với random_state=42
- Documented code với Vietnamese comments
- Scripts: `run_all_models.py` cho full comparison

### 6.2 Challenges & Solutions

#### **Challenge 1: Class Imbalance (95:5)**

**Impact**:

- Models learn biased towards majority class
- High accuracy but poor minority class detection
- Random Forest achieved 93% accuracy nhưng chỉ detect 14% strokes!

**Solutions Implemented**:
✅ SMOTE oversampling trên training set  
✅ Stratified train/test split  
✅ Focus on F1-Score và Recall thay vì Accuracy  
✅ Logistic Regression with class_weight='balanced'

**Results**:

- Training balanced 50-50
- Test giữ original distribution (realistic evaluation)
- Recall improved to 80% với LogReg

#### **Challenge 2: Missing Values (BMI)**

**Impact**: 201/5110 (3.93%) missing BMI values

**Solution**:
✅ Median imputation (robust to outliers)  
✅ Impute BEFORE outlier capping  
✅ Preserve distribution

**Results**: Zero missing values after preprocessing

#### **Challenge 3: Feature Complexity**

**Impact**: Mixed data types (numeric, binary, categorical)

**Solution**:
✅ `ColumnTransformer` với separate pipelines  
✅ Numeric: Imputer → Scaler  
✅ Categorical: Imputer → OneHotEncoder  
✅ Binary: Passthrough

**Results**: Clean 21-feature matrix

#### **Challenge 4: Model Selection**

**Impact**: Trade-off giữa precision và recall

**Analysis**:

- Random Forest: 93% accuracy nhưng 86% missed strokes ❌
- Logistic Regression: 75% accuracy nhưng 80% stroke detection ✅

**Decision**:
✅ Choose Logistic Regression  
✅ Prioritize Recall cho medical context  
✅ Accept false positives for safety

### 6.3 Key Insights

#### **Medical Insights**

🏥 **Age is dominant predictor**:

- 10x more important than any other feature
- Age 65+ has 16.17% stroke rate vs 0.13% for <30
- Non-negotiable feature for any model

🏥 **Health metrics critical**:

- Glucose level: 2nd most important (correlation 0.132)
- Hypertension & Heart disease: Strong indicators
- BMI: Non-linear relationship (important in RF)

🏥 **Lifestyle factors surprising**:

- Smoking status: Lower impact than expected
- Possibly confounded by age (elderly people quit)
- "Formerly smoked" có highest rate (7.91%) → age effect

🏥 **Demographics less important**:

- Gender: Nearly zero importance
- Urban vs Rural: No significant difference
- Work type: Minimal impact

#### **Machine Learning Insights**

🤖 **Simpler is better**:

- Logistic Regression outperforms complex models
- Random Forest overfits to majority class
- SVM computationally expensive without gains

🤖 **SMOTE effectiveness**:

- Balanced training crucial for minority class learning
- Must apply AFTER train/test split
- Don't apply to test set (realistic evaluation)

🤖 **Metrics matter**:

- Accuracy is misleading (95% baseline)
- F1-Score balances precision-recall
- Recall prioritized for medical screening

🤖 **Threshold tuning potential**:

- Default 0.5 gives 80% recall
- Lowering to 0.3 could achieve 90%+ recall
- Trade-off: More false positives (acceptable)

### 6.4 Limitations

⚠️ **Data Limitations**:

- Dataset size: 5,110 samples (moderate)
- Temporal coverage: Single timepoint (no longitudinal)
- Missing BMI: 3.93% could introduce bias
- "Unknown" smoking status: 30% unclear classification

⚠️ **Model Limitations**:

- Low precision (0.14): High false alarm rate
- F1-Score 0.24: Room for improvement
- No feature interactions explored
- No hyperparameter tuning (Random Forest, SVM)

⚠️ **Generalization Concerns**:

- Dataset from single source (Kaggle)
- Population may not represent all demographics
- Geographic bias unknown
- Temporal validity unclear (year of data collection)

### 6.5 Future Improvements

#### **Short-term** (Immediate)

1. **Hyperparameter Tuning**:

   ```python
   GridSearchCV hoặc RandomizedSearchCV
   - Logistic Regression: C, penalty
   - Random Forest: n_estimators, max_depth, min_samples_split
   - SVM: C, gamma
   ```

2. **Threshold Optimization**:

   ```python
   # Find optimal threshold maximizing F1 or Recall
   from sklearn.metrics import precision_recall_curve
   precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
   ```

3. **Feature Engineering**:
   - Age bins: <30, 30-50, 50-65, 65+
   - BMI categories: Underweight, Normal, Overweight, Obese
   - Glucose categories: Normal, Pre-diabetes, Diabetes
   - Interaction features: age × heart_disease

#### **Medium-term**

4. **Ensemble Methods**:

   ```python
   # Voting Classifier
   VotingClassifier([
       ('lr', LogisticRegression()),
       ('svm', SVC(probability=True))
   ], voting='soft')
   ```

5. **Cross-Validation**:

   ```python
   StratifiedKFold(n_splits=5)
   # More robust performance estimates
   ```

6. **Cost-Sensitive Learning**:

   ```python
   # Assign higher cost to False Negatives
   class_weight = {0: 1, 1: 10}
   ```

7. **SHAP Analysis**:
   ```python
   import shap
   # Explain individual predictions
   # Feature importance với interactions
   ```

#### **Long-term** (Research)

8. **Deep Learning**:

   - Neural Networks cho complex patterns
   - Autoencoders cho anomaly detection
   - Caution: Needs more data (5K may be insufficient)

9. **Longitudinal Data**:

   - Track patients over time
   - Survival analysis
   - Time-to-stroke prediction

10. **External Validation**:

    - Test on different datasets
    - Multi-site validation
    - Cross-population generalization

11. **Clinical Integration**:
    - Risk calculator web app
    - Integration with EMR systems
    - Real-time prediction API

### 6.6 Production Deployment Recommendations

#### **Deployment Architecture**

```
User Input → Preprocessing Pipeline → Model → Risk Score → Clinical Decision Support
              (preprocessor.joblib)   (LogReg)   (0-1)
```

**Steps**:

1. **Input Validation**: Check all required features present
2. **Preprocessing**: Apply saved `preprocessor.joblib`
3. **Prediction**: LogisticRegression.predict_proba()
4. **Interpretation**:
   - probability > 0.5 → High Risk
   - 0.3-0.5 → Medium Risk
   - < 0.3 → Low Risk
5. **Output**: Risk score + feature contributions (SHAP values)

#### **Monitoring Strategy**

**Track metrics**:

- Prediction distribution over time
- Feature drift (data distribution changes)
- Model performance on new data
- False Negative rate (critical!)

**Retrain triggers**:

- Performance degrades > 5% F1-Score drop
- Feature distribution shift detected
- New data accumulated (> 20% of original)
- Quarterly scheduled retraining

### 6.7 Final Recommendations

#### **For Clinicians**

✅ **Use as screening tool**:

- High sensitivity (80%) good for initial screening
- Positive prediction → Further diagnostic tests
- Negative prediction → Lower risk but monitor

✅ **Focus on high-risk groups**:

- Age 65+ (16% stroke rate)
- Hypertension + Heart disease patients
- High glucose levels

⚠️ **Limitations to communicate**:

- Not diagnostic (14% precision)
- Many false alarms expected
- Clinical judgment essential

#### **For Data Scientists**

✅ **Key Lessons**:

- Class imbalance: Use SMOTE + stratified split
- Metrics: F1/Recall > Accuracy for medical
- Simple models: Often outperform complex ones
- Feature selection: Validates domain knowledge

✅ **Best Practices**:

- Zero data leakage (fit on train only)
- Reproducibility (random_state, seeds)
- Documentation (Vietnamese + English)
- Visualization (comprehensive charts)

✅ **Next Steps**:

1. Hyperparameter tuning
2. Ensemble methods
3. SHAP explanations
4. External validation

#### **For Stakeholders**

✅ **Business Value**:

- Early stroke detection → Better outcomes
- Cost-effective screening tool
- Scalable to large populations

✅ **Risk Management**:

- High false positives → Extra tests cost
- Low false negatives → Missed diagnosis risk
- Current model: Conservative (better safe)

✅ **Deployment Path**:

- Pilot study with clinical validation
- Integration with existing workflows
- Continuous monitoring and improvement

### 6.8 Acknowledgments

**Dataset**: Healthcare Dataset Stroke Data (Kaggle)  
**Libraries**: scikit-learn, pandas, numpy, imbalanced-learn, matplotlib, seaborn  
**Tools**: Python 3.11.4, VS Code, Git  
**Repository**: https://github.com/hothanhnha256/heart-stroke-data-mining

---

## APPENDIX

### A. Command Reference

**Complete Workflow**:

```powershell
# 1. Setup environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. EDA
python eda_analysis.py

# 3. Preprocessing
python prepare-stroke.py `
  --input data-raw/healthcare-dataset-stroke-data.csv `
  --output-dir data-pre `
  --scale standard `
  --cap-outliers `
  --smote

# 4. Feature Selection
python feature_selection.py

# 5. Model Training & Comparison
python run_all_models.py

# 6. Individual Models
python model-A/logistics_reg.py
python model-A/random_forest.py
python model-B/svm.py
```

### B. File Structure

```
heart-stroke/
├── data-raw/
│   └── healthcare-dataset-stroke-data.csv    # Raw data (5,110 rows)
├── data-pre/
│   ├── train_preprocessed.csv               # 7,778 rows (SMOTE)
│   ├── test_preprocessed.csv                # 1,022 rows
│   ├── preprocessor.joblib                  # Fitted pipeline
│   ├── feature_names.txt                    # 21 features
│   └── prep_meta.json                       # Metadata
├── eda/
│   ├── eda_target_distribution.png
│   ├── eda_numeric_analysis.png
│   ├── eda_categorical_analysis.png
│   ├── eda_correlation_matrix.png
│   └── eda_age_analysis.png
├── feature/
│   ├── feature_correlation_analysis.png
│   ├── feature_mutual_info_analysis.png
│   ├── feature_rf_importance_analysis.png
│   ├── feature_statistical_analysis.png
│   ├── feature_combined_ranking.png
│   └── feature_selection_results.json
├── model-A/
│   ├── logistics_reg.py
│   └── random_forest.py
├── model-B/
│   ├── svm.py
│   └── svm-and-knn.ipynb
├── prepare-stroke.py                        # Preprocessing pipeline
├── eda_analysis.py                          # EDA script
├── feature_selection.py                     # Feature selection
├── run_all_models.py                        # Model comparison
├── model_comparison_results.csv             # Results table
├── model_roc_curves_comparison.png          # ROC curves
├── model_metrics_comparison.png             # Metrics chart
├── model_confusion_matrices.png             # Confusion matrices
├── models_final_report.txt                  # Detailed report
├── models_results.json                      # JSON results
├── README.md                                # Documentation
├── REPORT.md                                # This report
├── QUICKSTART.md                            # Quick start guide
└── requirements.txt                         # Dependencies
```

### C. Key Metrics Summary

| Model                      | Accuracy | Precision | Recall     | F1-Score   | ROC-AUC    | Strokes Detected |
| -------------------------- | -------- | --------- | ---------- | ---------- | ---------- | ---------------- |
| **Logistic Regression** ⭐ | 0.7495   | 0.1399    | **0.8000** | **0.2381** | **0.8456** | **40/50 (80%)**  |
| Random Forest              | 0.9266   | 0.1795    | 0.1400     | 0.1573     | 0.7615     | 7/50 (14%)       |
| SVM (RBF)                  | 0.7945   | 0.1040    | 0.4200     | 0.1667     | 0.7648     | 21/50 (42%)      |
| KNN (k=5)                  | 0.8415   | 0.0942    | 0.2600     | 0.1383     | 0.6202     | 13/50 (26%)      |

**Winner**: Logistic Regression (Best F1, Recall, ROC-AUC)

---

**END OF REPORT**

**Date**: October 18, 2025  
**Project**: Heart Stroke Prediction - Data Mining HK251  
**Team**: Data Mining Project Group

- Phát hiện BMI có non-linear relationship

✅ **Preprocessing Pipeline**:

- Xử lý missing values (median/mode imputation)
- Outlier capping (IQR method)
- Feature encoding (12 cols → 21 features)
- SMOTE balancing (cân bằng 50-50 trên train set)
- **Zero data leakage** (fit trên train only)

✅ **Feature Selection**:

- Multi-method approach (4 methods)
- Top 8 features identified
- Validates medical knowledge (age, glucose, hypertension critical)

✅ **Infrastructure**:

- Modular, reusable code
- CLI interface
- Artifact management
- Team collaboration framework

### 6.2 Thách thức Đã Giải quyết

✅ Class Imbalance → SMOTE + stratification + proper metrics  
✅ Missing Data → Median/mode imputation  
✅ Outliers → IQR capping  
✅ Data Leakage → Careful pipeline design  
✅ Feature Selection → Multi-method consensus

### 6.3 Hạn chế và Đề xuất Cải tiến

❌ **Hạn chế**:

- Baseline model F1-Score thấp (0.10)
- Chưa có cross-validation
- Chưa exploit non-linear relationships đầy đủ
- Chưa có ensemble methods

🔄 **Đề xuất**:

1. **Short-term**: Hyperparameter tuning, threshold optimization, cross-validation
2. **Medium-term**: Feature engineering (interactions, polynomials), ensemble (stacking, voting), XGBoost/LightGBM
3. **Long-term**: Deep learning, AutoML, production deployment

### 6.4 Lessons Learned

**Technical**:

- Accuracy misleading với imbalanced data
- SMOTE phải apply sau train/test split
- Feature selection cần multiple methods
- Age >> all other features trong medical prediction

**Domain**:

- Stroke risk tăng exponentially với tuổi
- Health metrics (glucose, BP, heart disease) critical
- Lifestyle factors (smoking) confounded by age
- Gender surprisingly không quan trọng

**Project Management**:

- Modular code → team collaboration
- Git branches → parallel development
- Documentation → reduce confusion
- Automated consolidation → save time

### 6.5 Kết luận Cuối cùng

Dự án đã successfully xây dựng một **complete data mining pipeline** từ raw data đến model evaluation:

✅ Comprehensive EDA với insights rõ ràng  
✅ Robust preprocessing preventing data leakage  
✅ Scientific feature selection identifying top predictors  
✅ Scalable framework cho team collaboration  
✅ Production-ready code với documentation đầy đủ

**Next Steps**: Cải thiện model performance qua tuning, feature engineering, và ensemble methods để đạt clinical-grade predictions.

---

## 📚 REFERENCES

1. **Dataset**: Kaggle Stroke Prediction Dataset
2. **SMOTE**: Chawla et al. - Synthetic Minority Oversampling Technique
3. **Scikit-learn**: Machine Learning in Python
4. **Imbalanced-learn**: Tools for imbalanced datasets
5. **Medical Knowledge**: WHO Stroke Guidelines, American Heart Association

---

## 📎 APPENDIX

### A. Execution Commands

```powershell
# Environment Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Full Pipeline
python eda_analysis.py
python prepare-stroke.py --input data-raw/healthcare-dataset-stroke-data.csv --output-dir data-pre --scale standard --cap-outliers --smote
python feature_selection.py
python implement.py
python model_consolidation.py
```

### B. File Outputs

```
eda/*.png                          # EDA visualizations
feature/*.png                      # Feature selection charts
feature/feature_selection_results.json  # Top features ranking
data-pre/train_preprocessed.csv    # 7,778 × 22 (balanced)
data-pre/test_preprocessed.csv     # 1,022 × 22 (original dist)
data-pre/preprocessor.joblib       # Reusable pipeline
data-pre/prep_meta.json           # Metadata
model_results_comparison.png       # Model comparison charts
detailed_model_report.txt          # Detailed results
```

### C. Team Contributions

- **Data Pipeline**: Preprocessing, EDA, Feature Selection
- **Model A**: Logistic Regression, Random Forest
- **Model B**: SVM, KNN
- **Documentation**: README, REPORT, Copilot Instructions

---

**📊 End of Report**  
_Generated: 18/10/2025_  
_Project: Heart Stroke Prediction_  
_Team: HK251 Data Mining_
