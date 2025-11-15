# Heart Stroke Prediction - Data Mining Project

Dự án Data Mining hoàn chỉnh để dự đoán nguy cơ đột quỵ (stroke) dựa trên dữ liệu y tế và nhân khẩu học, sử dụng Machine Learning.

---

## Dataset Overview

- **Nguồn**: Healthcare Dataset Stroke Data (Kaggle)
- **Kích thước**: 5,110 bệnh nhân × 12 thuộc tính
- **Target**: Dự đoán đột quỵ (stroke: 0/1)
- **Vấn đề**: **Class Imbalance** nghiêm trọng (95.1% không đột quỵ, 4.9% đột quỵ)

---

## Project Structure

```
heart-stroke/
├── data-raw/
│   └── healthcare-dataset-stroke-data.csv    # Dataset gốc (5,111 rows × 12 cols)
├── data-pre/                                 # Dữ liệu đã preprocessing
│   ├── train_preprocessed.csv               # Training set đã xử lý
│   ├── test_preprocessed.csv                # Test set đã xử lý
│   ├── preprocessor.joblib                  # Sklearn pipeline
│   ├── feature_names.txt                    # Danh sách features
│   └── prep_meta.json                       # Metadata
├── eda/                                      # EDA visualizations
│   └── eda_*.png                            # Charts và plots
├── feature/                                  # Feature selection results
│   ├── feature_*.png                        # Feature importance plots
│   └── feature_selection_results.json       # Ranking results
├── model-A/                                  # Models - Team A
│   ├── logistics_reg.py                     # Logistic Regression
│   └── random_forest.py                     # Random Forest
├── model-B/                                  # Models - Team B
│   ├── svm.py                               # Support Vector Machine
│   └── svm-and-knn.ipynb                    # SVM + KNN notebook
├── report/                                   # LaTeX academic report
│   ├── main.tex                             # Main document
│   ├── Section 2/ ... Section 8/            # Report chapters
│   └── image/                               # Report images
├── prepare-stroke.py                        # Main preprocessing pipeline
├── implement.py                             # Simple model implementation
├── eda_analysis.py                          # Exploratory Data Analysis
├── feature_selection.py                     # Multi-method feature selection
├── model_consolidation.py                   # Tổng hợp kết quả từ team
├── README.md                                # Documentation (this file)
└── requirements.txt                         # Dependencies
```

````

---

## Quick Start

### Environment Setup

**Windows (PowerShell):**

```powershell
# Clone repository
git clone https://github.com/hothanhnha256/heart-stroke-data-mining.git
cd heart-stroke-data-mining

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Upgrade pip and install dependencies
pip install -U pip
pip install -r requirements.txt
````

**Ubuntu/Linux:**

```bash
# Clone repository
git clone https://github.com/hothanhnha256/heart-stroke-data-mining.git
cd heart-stroke-data-mining

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install -U pip
pip install -r requirements.txt
```

### Verify Installation

```powershell
# Check Python version (cần >= 3.9)
python --version

# Check installed packages
pip list | Select-String "pandas|numpy|scikit-learn"
```

---

## Complete Workflow

### Step 1: Exploratory Data Analysis (EDA)

Phân tích và hiểu dataset trước khi xử lý:

```powershell
python eda_analysis.py
```

**Outputs:**

- Thống kê mô tả dataset
- Phân tích class imbalance (4.9% stroke)
- Distributions của numeric features
- Correlation analysis
- Age group analysis
- Visualizations: `eda/*.png`

**Key Insights:**

- Class imbalance nghiêm trọng: 95.1% No Stroke, 4.9% Stroke
- Missing values: BMI có 201 giá trị thiếu
- Outliers: `bmi` và `avg_glucose_level` cần xử lý
- Age là yếu tố quan trọng nhất

---

### Step 2: Data Preprocessing

Tiền xử lý dữ liệu với pipeline hoàn chỉnh:

```powershell
python prepare-stroke.py `
  --input data-raw/healthcare-dataset-stroke-data.csv `
  --output-dir data-pre `
  --scale standard `
  --cap-outliers `
  --smote
```

**Parameters:**

- `--input`: Đường dẫn tới CSV gốc
- `--output-dir`: Thư mục lưu artifacts (default: `data-pre/`)
- `--test-size`: Tỷ lệ test set (default: 0.2)
- `--scale`: Phương pháp scaling `standard|minmax|none` (default: standard)
- `--cap-outliers`: Bật outlier capping bằng IQR method
- `--smote`: Bật SMOTE oversampling cho training set
- `--random-state`: Random seed (default: 42)

**Processing Steps:**

1. **NaN Handling**
   - Numeric: Median imputation
   - Categorical: Most frequent imputation
2. **Outlier Treatment**

   - IQR-based capping cho `bmi` và `avg_glucose_level`
   - Formula: `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

3. **Feature Encoding**

   - Categorical → OneHotEncoder (14 features)
   - Numeric → StandardScaler (3 features)
   - Binary → Passthrough (2 features)

4. **Train/Test Split**

   - Stratified split (80/20)
   - Preserves class distribution

5. **SMOTE Oversampling**
   - Applied only on training set
   - Balances classes to 50-50

**Outputs:**

```
data-pre/
├── train_preprocessed.csv    # 7,778 rows (balanced)
├── test_preprocessed.csv     # 1,022 rows (original dist)
├── preprocessor.joblib       # Fitted sklearn pipeline
├── feature_names.txt         # 21 feature names
└── prep_meta.json           # Metadata & statistics
```

---

### Step 3: Feature Selection

Tìm features quan trọng nhất với 4 phương pháp:

```powershell
python feature_selection.py
```

**Methods:**

1. **Correlation Analysis** - Pearson correlation với target
2. **Mutual Information** - Information gain measurement
3. **Random Forest Importance** - Tree-based feature importance
4. **Statistical Tests** - ANOVA (numeric) + Chi-square (categorical)

**Combined Ranking:**

- Normalize tất cả scores về [0, 1]
- Tính trung bình của 4 phương pháp
- Rank theo combined score

**Top 8 Features:**

1. **age** - Tuổi (quan trọng nhất!)
2. **avg_glucose_level** - Mức glucose
3. **bmi** - Chỉ số BMI
4. **hypertension** - Tăng huyết áp
5. **heart_disease** - Bệnh tim
6. **ever_married_Yes** - Đã kết hôn
7. **work*type*\*** - Loại công việc
8. **smoking*status*\*** - Tình trạng hút thuốc

**Outputs:**

```
feature/
├── feature_correlation_analysis.png
├── feature_mutual_info_analysis.png
├── feature_rf_importance_analysis.png
├── feature_statistical_analysis.png
├── feature_combined_ranking.png
└── feature_selection_results.json
```

---

### Step 4: Model Training

#### A. Baseline Model

Quick baseline với Logistic Regression:

```powershell
python implement.py
```

**Model Configuration:**

- Algorithm: Logistic Regression
- Hyperparameters: `max_iter=500`, `class_weight='balanced'`
- Metrics: Precision, Recall, F1-Score, Accuracy

#### **B. Advanced Models**

**Model A (Logistic Regression & Random Forest):**

```powershell
cd model-A
python logistics_reg.py
python random_forest.py
```

**Model B (SVM & KNN):**

```powershell
cd model-B
python svm.py
# Hoặc chạy notebook:
jupyter notebook svm-and-knn.ipynb
```

---

### Step 5: Results Consolidation

Tổng hợp và so sánh kết quả từ tất cả models:

```powershell
python model_consolidation.py
```

**Features:**

- Tổng hợp metrics từ nhiều models
- So sánh performance (Accuracy, F1, Precision, Recall)
- Visualizations (bar charts, heatmaps, scatter plots)
- Detailed report generation
- Export results to JSON

**Outputs:**

```
model_results_comparison.png
detailed_model_report.txt
model_results_consolidated.json
```

---

## Step 6: Generate Academic Report (LaTeX)

### Report Structure

```
report/
├── main.tex                    # Main LaTeX document
├── division_of_work.tex        # Phân công công việc
├── resources.tex               # Tài liệu tham khảo
├── Section 2/
│   └── index.tex              # Giới thiệu
├── Section 3/
│   └── index.tex              # Cơ sở lý thuyết
├── Section 4/
│   └── index.tex              # Khảo sát và phân tích dữ liệu (EDA)
├── Section 5/
│   └── index.tex              # Tiền xử lý dữ liệu
├── Section 6/
│   └── index.tex              # Xây dựng mô hình
├── Section 7/
│   └── index.tex              # Kết quả và đánh giá
└── Section 8/
    └── index.tex              # Kết luận
```

### Compile LaTeX Report

**Windows (PowerShell):**

```powershell
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex  # Chạy 2 lần để cập nhật TOC
```

**Ubuntu/Linux:**

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex  # Chạy 2 lần để cập nhật TOC
```

**Notes:**

- Flag `-interaction=nonstopmode`: Tự động bỏ qua errors và tiếp tục compile
- Chạy 2 lần để cập nhật Table of Contents và cross-references
- Output: `main.pdf` trong thư mục `report/`
- Cần cài đặt MiKTeX (Windows) hoặc TeX Live (Linux/Mac)

### Report Content

- **Section 2**: Giới thiệu về bài toán dự đoán đột quỵ
- **Section 3**: Cơ sở lý thuyết (Binary Classification, Metrics, SMOTE, Algorithms: LogReg, RF, SVM, KNN)
- **Section 4**: EDA với Professional Theme visualizations
- **Section 5**: Tiền xử lý dữ liệu (Missing values, Outliers, Scaling, SMOTE)
- **Section 6**: Xây dựng 4 mô hình ML
- **Section 7**: So sánh kết quả và đánh giá
- **Section 8**: Kết luận và hướng phát triển

### Troubleshooting LaTeX Compile

**Compile timeout:**

- Kiểm tra file paths trong `\input{}` commands
- Đảm bảo tất cả images tồn tại trong `report/image/`
- Tắt draft mode nếu đang bật

**Missing packages:**

```powershell
# MiKTeX sẽ tự động cài đặt packages thiếu
# Hoặc cài thủ công qua MiKTeX Console
```

**Permission errors:**

```powershell
# Đảm bảo không mở PDF đang compile
# Xóa các file tạm: *.aux, *.log, *.toc
cd report
Remove-Item *.aux, *.log, *.toc, *.out
```

---

## Dataset Information

| Column              | Type    | Description               | Example Values                                           |
| ------------------- | ------- | ------------------------- | -------------------------------------------------------- |
| `id`                | int     | Patient ID (dropped)      | 9046, 51676                                              |
| `gender`            | object  | Gender                    | Male, Female, Other                                      |
| `age`               | float   | Age in years              | 67, 61, 80                                               |
| `hypertension`      | int     | Hypertension (0/1)        | 0, 1                                                     |
| `heart_disease`     | int     | Heart disease (0/1)       | 0, 1                                                     |
| `ever_married`      | object  | Marital status            | Yes, No                                                  |
| `work_type`         | object  | Type of work              | Private, Govt_job, Self-employed, Never_worked, children |
| `Residence_type`    | object  | Residence type            | Urban, Rural                                             |
| `avg_glucose_level` | float   | Average glucose level     | 228.69, 202.21                                           |
| `bmi`               | float   | Body Mass Index           | 36.6, 32.5                                               |
| `smoking_status`    | object  | Smoking status            | formerly smoked, never smoked, smokes, Unknown           |
| **`stroke`**        | **int** | **Target variable (0/1)** | **0, 1**                                                 |

### Column Categorization for Preprocessing

```python
# Schema configuration
target_col = "stroke"
drop_cols = ["id"]  # Excluded from training

# Feature types
numeric_cols = ["age", "avg_glucose_level", "bmi"]
binary_cols = ["hypertension", "heart_disease"]
categorical_cols = ["gender", "ever_married", "work_type",
                   "Residence_type", "smoking_status"]
```

### After Preprocessing (21 Features)

**Numeric (3):**

- age
- avg_glucose_level
- bmi

**OneHot Encoded (14):**

- gender_Female, gender_Male, gender_Other
- ever_married_No, ever_married_Yes
- work_type_Govt_job, work_type_Never_worked, work_type_Private, work_type_Self-employed, work_type_children
- Residence_type_Rural, Residence_type_Urban
- smoking_status_Unknown, smoking_status_formerly smoked, smoking_status_never smoked, smoking_status_smokes

**Binary (2):**

- hypertension
- heart_disease

**Target (1):**

- stroke

---

## Key Technical Decisions

### 1. Class Imbalance Handling

**Problem:** 95.1% No Stroke vs 4.9% Stroke

**Solutions:**

- **SMOTE** oversampling (chỉ trên training set)
- **class_weight='balanced'** trong models
- **Stratified sampling** trong train/test split
- **Metrics focus**: F1-Score, Precision, Recall (không chỉ Accuracy)

### 2. Data Leakage Prevention

```python
# CORRECT: Fit on train only
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# WRONG: Fit on all data
preprocessor.fit(X)  # Information leakage!
```

### 3. Pipeline Architecture

```
Raw Data
    ↓
Missing Value Imputation
    ↓
Outlier Capping (optional)
    ↓
Train/Test Split (stratified)
    ↓
Feature Encoding & Scaling
    ↓
SMOTE (training only)
    ↓
Model Training
```

---

## Performance Metrics

### Priority Metrics (for imbalanced data)

1. **F1-Score** - Harmonic mean of Precision & Recall
2. **Precision** - Tỷ lệ dự đoán đúng trong các positive predictions
3. **Recall** - Tỷ lệ tìm được trong các positive cases
4. **ROC-AUC** - Area under ROC curve
5. **Accuracy** - Chỉ dùng để tham khảo (misleading với imbalanced data)

### Why not Accuracy?

```
Example với dataset:
- 95% No Stroke, 5% Stroke
- Model dự đoán TẤT CẢ là "No Stroke"
- Accuracy = 95% (looks good!)
- Nhưng: Recall = 0% (completely useless!)

→ F1-Score là metric tốt hơn!
```

---

## Development Tools

### Required Dependencies

```txt
# Core
numpy==1.26.4
pandas==2.2.2

# ML Framework
scikit-learn==1.4.2
scipy==1.11.4
joblib==1.3.2

# Imbalanced Learning
imbalanced-learn==0.12.3

# Visualization
matplotlib==3.8.4
seaborn==0.13.2
```

### Optional Tools

```powershell
# Jupyter for notebooks
pip install jupyter

# Code formatting
pip install black

# Linting
pip install pylint
```

---

## Documentation

- **README.md** - Project overview và quick start (this file)
- **REPORT.md** - Detailed analysis report với findings
- **REPORT_TEMPLATE.md** - Template cho báo cáo đầy đủ
- **.github/copilot-instructions.md** - AI coding guidelines

---

## Team Collaboration

Tran The Nhan
Ho Thanh Nha
Dang Minh Nhat

### Branch Strategy

```
main (hoặc master)
├── model_A - Logistic Regression & Random Forest
└── model_B - SVM & KNN
```

### Adding Your Model Results

```python
from model_consolidation import ModelResultsConsolidator

consolidator = ModelResultsConsolidator()

# Add your model
consolidator.add_model_result(
    model_name="Your Model Name",
    member_name="Your Name",
    y_true=y_test,
    y_pred=y_pred,
    model_params={"param1": value1},
    preprocessing_info={"scaler": "Standard", "smote": True}
)

# Generate reports
consolidator.print_summary()
consolidator.visualize_results()
consolidator.save_results_to_json()
```

---

## Common Issues & Solutions

### Issue 1: Import Error

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**

```powershell
# Activate venv first!
.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue 2: SMOTE Not Found

```
RuntimeError: Bạn đã bật --smote nhưng chưa cài imbalanced-learn
```

**Solution:**

```powershell
pip install imbalanced-learn
```

### Issue 3: File Not Found

```
FileNotFoundError: data-raw/healthcare-dataset-stroke-data.csv
```

**Solution:**

- Đảm bảo file CSV ở đúng thư mục `data-raw/`
- Check tên file chính xác

### Issue 4: Memory Error (large dataset)

**Solution:**

```python
# Giảm SMOTE samples
sm = SMOTE(sampling_strategy=0.5)  # Instead of 1.0

# Hoặc dùng batch processing
```

---

## To-Do List

- Dataset exploration & EDA (completed)
- Preprocessing pipeline (completed)
- Feature selection (completed)
- Train/test split với stratification (completed)
- SMOTE implementation (completed)
- Baseline model - Logistic Regression (completed)
- Advanced models - SVM, KNN, RF (in progress)
- Hyperparameter tuning (planned)
- Cross-validation (planned)
- Ensemble methods (planned)
- Final report writing (planned)

---

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [Imbalanced Learning](https://imbalanced-learn.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

## Contributors

- **Member A** - Logistic Regression & Random Forest
- **Member B** - SVM & KNN
- **Team** - EDA, Preprocessing, Feature Selection, Consolidation

---

## Acknowledgments

- Dataset: Kaggle Healthcare Stroke Dataset
- Framework: Scikit-learn, Pandas, NumPy
- Inspiration: Data Mining coursework HK251
  binary_cols = ["hypertension", "heart_disease"]
  categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

````

---

## Key Insights

### Top Risk Factors (từ Feature Selection):

1. **Age** - Yếu tố quan trọng nhất
2. **Average Glucose Level** - Chỉ số glucose
3. **BMI** - Chỉ số khối cơ thể
4. **Hypertension** - Tăng huyết áp
5. **Heart Disease** - Bệnh tim

### EDA Findings:

- Nguy cơ stroke tăng đáng kể sau 50 tuổi
- Class imbalance nghiêm trọng cần SMOTE
- BMI và glucose levels có outliers cần xử lý

---

## Advanced Usage

### Custom Preprocessing

```powershell
# Không SMOTE, sử dụng MinMax scaling
python prepare-stroke.py --input data-raw/healthcare-dataset-stroke-data.csv --output-dir data-pre --scale minmax

# Test size 30%, không cap outliers
python prepare-stroke.py --input data-raw/healthcare-dataset-stroke-data.csv --output-dir data-pre --test-size 0.3
````

### Team Collaboration

```python
from model_consolidation import ModelResultsConsolidator

consolidator = ModelResultsConsolidator()
consolidator.add_model_result("Random Forest", "Member A", y_true, y_pred)
consolidator.print_summary()
consolidator.visualize_results()
```

---

## Dependencies

**Core ML Stack:**

- pandas==2.2.2
- scikit-learn==1.4.2
- numpy==1.26.4

**Visualization & Analysis:**

- matplotlib==3.8.4
- seaborn==0.13.2

**Optional:**

- imbalanced-learn==0.12.3 (cho SMOTE)

---

## Documentation

- **Detailed Report**: Xem `REPORT_TEMPLATE.md`
- **AI Guidelines**: `.github/copilot-instructions.md`
- **Code Structure**: Tất cả scripts có docstrings Vietnamese

---

## Project Highlights

- **Reproducible Pipeline**: Seed-controlled, artifact-based
- **Class Imbalance Handling**: SMOTE + Stratified sampling
- **Multi-method Feature Selection**: 4 different approaches
- **Comprehensive EDA**: Statistical + Visual analysis
- **Team Collaboration**: Results consolidation framework
- **Production Ready**: Error handling, Vietnamese docs

---

**Note**: Đây là pipeline hoàn chỉnh cho phân tích dữ liệu stroke prediction. Mỗi script có thể chạy độc lập hoặc theo workflow trên.
