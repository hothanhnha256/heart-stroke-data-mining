# BÁO CÁO PHẦN DATASET + PREPROCESSING + FEATURE SELECTION

## Dự án: Heart Stroke Prediction

---

## 1. TỔNG QUAN DATASET

### 1.1 Nguồn dữ liệu

- **Dataset**: Healthcare Dataset Stroke Data (Kaggle)
- **Kích thước**: 5,110 dòng × 12 cột
- **Mục tiêu**: Dự đoán nguy cơ đột quỵ (stroke) dựa trên các yếu tố sức khỏe và nhân khẩu học

### 1.2 Cấu trúc dữ liệu

| Cột               | Kiểu    | Mô tả                      | Ví dụ                                                    |
| ----------------- | ------- | -------------------------- | -------------------------------------------------------- |
| id                | int     | Mã định danh duy nhất      | 9046, 51676                                              |
| gender            | object  | Giới tính                  | Male, Female, Other                                      |
| age               | float   | Tuổi                       | 67, 61, 80                                               |
| hypertension      | int     | Tăng huyết áp (0/1)        | 0, 1                                                     |
| heart_disease     | int     | Bệnh tim (0/1)             | 0, 1                                                     |
| ever_married      | object  | Đã kết hôn                 | Yes, No                                                  |
| work_type         | object  | Loại công việc             | Private, Govt_job, Self-employed, Never_worked, children |
| Residence_type    | object  | Nơi cư trú                 | Urban, Rural                                             |
| avg_glucose_level | float   | Mức glucose trung bình     | 228.69, 202.21                                           |
| bmi               | float   | Chỉ số BMI                 | 36.6, 32.5                                               |
| smoking_status    | object  | Tình trạng hút thuốc       | formerly smoked, never smoked, smokes, Unknown           |
| **stroke**        | **int** | **Target - Đột quỵ (0/1)** | **0, 1**                                                 |

### 1.3 Đặc điểm chính của dataset

**Class Imbalance:**

- No Stroke (0): 4,861 cases (95.1%)
- Stroke (1): 250 cases (4.9%)
- **⚠️ Vấn đề**: Dataset có độ mất cân bằng lớp cao, cần xử lý đặc biệt

**Missing Values:**

- `bmi`: Có giá trị N/A cần được xử lý
- Các cột khác: Hoàn chỉnh

**Outliers:**

- `avg_glucose_level`: Có các giá trị bất thường cao
- `bmi`: Có các giá trị ngoại lai cần được kiểm soát

---

## 2. EXPLORATORY DATA ANALYSIS (EDA)

### 2.1 Phân tích biến Target

- **Tỷ lệ stroke**: 4.9% (rất thấp - typical của bài toán y tế)
- **Phân phối theo tuổi**: Nguy cơ stroke tăng đáng kể sau 50 tuổi
- **Yếu tố nguy cơ cao**: Tuổi cao, tăng huyết áp, bệnh tim

### 2.2 Các insights chính từ EDA

1. **Tuổi**: Yếu tố quan trọng nhất - tỷ lệ stroke tăng theo tuổi
2. **BMI và Glucose**: Có correlation với stroke risk
3. **Công việc**: Một số loại công việc có risk cao hơn
4. **Hút thuốc**: Có tác động nhưng không rõ ràng như mong đợi

**📊 Kết quả EDA chi tiết**: Xem file `eda_analysis.py` và các biểu đồ được tạo

---

## 3. PREPROCESSING PIPELINE

### 3.1 Kiến trúc xử lý dữ liệu

Chúng tôi sử dụng **sklearn Pipeline** với `ColumnTransformer`:

```python
# Schema cố định
target_col = "stroke"
drop_cols = ["id"]  # Loại bỏ ID
numeric_cols = ["age", "avg_glucose_level", "bmi"]
binary_cols = ["hypertension", "heart_disease"]
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
```

### 3.2 Các bước preprocessing

#### 3.2.1 Xử lý Missing Values

- **Numeric columns**: `SimpleImputer(strategy="median")`
- **Categorical columns**: `SimpleImputer(strategy="most_frequent")`
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

**📝 Ghi chú**: Báo cáo này tập trung vào Dataset + Preprocessing + Feature Selection theo yêu cầu. Phần modeling results sẽ được cập nhật sau khi tổng hợp từ các thành viên khác.
