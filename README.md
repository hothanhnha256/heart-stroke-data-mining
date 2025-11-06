# Heart Stroke Data Mining Project

Dự án phân tích và dự đoán nguy cơ đột quỵ sử dụng Machine Learning.

## 📁 Project Structure

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

## 🚀 Quick Start

### Environment Setup

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Ubuntu:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Main Workflow

## 📊 Step 1: Exploratory Data Analysis

```powershell
python eda_analysis.py
```

**Outputs:**

- Thống kê mô tả dataset
- Phân tích target distribution (class imbalance: 4.9% stroke)
- Visualizations: distributions, correlations, age analysis
- Files: `eda_*.png`

## 🔧 Step 2: Data Preprocessing

```powershell
python prepare-stroke.py --input data-raw/healthcare-dataset-stroke-data.csv --output-dir data-pre --scale standard --cap-outliers --smote
```

**Key Features:**

- **NaN handling**: Median imputer cho numeric, most_frequent cho categorical
- **Outlier capping**: IQR-based cho `bmi` và `avg_glucose_level`
- **Encoding**: OneHotEncoder với `handle_unknown='ignore'`
- **Scaling**: StandardScaler/MinMaxScaler options
- **Train/Test Split**: Stratified split (80/20)
- **Class Balancing**: SMOTE oversampling (optional)

**Parameters:**

- `--input`: Path đến CSV gốc
- `--output-dir`: Thư mục lưu artifacts
- `--test-size`: Tỷ lệ test set (default: 0.2)
- `--scale`: `standard|minmax|none` (default: standard)
- `--cap-outliers`: Bật outlier capping
- `--smote`: Bật SMOTE oversampling
- `--random-state`: Random seed (default: 42)

## 🎯 Step 3: Feature Selection

```powershell
python feature_selection.py
```

**Methods:**

1. **Correlation Analysis**: Pearson correlation với target
2. **Mutual Information**: Information gain
3. **Random Forest Importance**: Tree-based importance
4. **Statistical Tests**: ANOVA (numeric) + Chi-square (categorical)

**Outputs:**

- Combined ranking của tất cả features
- Top K features quan trọng nhất
- Visualizations: `feature_*.png`
- Results: `feature_selection_results.json`

## 🤖 Step 4: Model Training

```powershell
python implement.py
```

Simple LogisticRegression baseline model.

## 📋 Step 5: Results Consolidation

```powershell
python model_consolidation.py
```

Framework để tổng hợp kết quả từ các thành viên trong team.

---

## 📄 Step 6: Generate Academic Report (LaTeX)

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

## 📊 Dataset Information

- **Source**: Healthcare Dataset Stroke Data (Kaggle)
- **Size**: 5,111 rows × 12 columns
- **Target**: `stroke` (binary: 0/1)
- **Class Imbalance**: 95.1% No Stroke, 4.9% Stroke
- **Missing Values**: `bmi` column có N/A values

### Schema

```python
target_col = "stroke"
drop_cols = ["id"]  # Không dùng để train
numeric_cols = ["age", "avg_glucose_level", "bmi"]
binary_cols = ["hypertension", "heart_disease"]
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
```

---

## 🔍 Key Insights

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

## 🛠️ Advanced Usage

### Custom Preprocessing

```powershell
# Không SMOTE, sử dụng MinMax scaling
python prepare-stroke.py --input data-raw/healthcare-dataset-stroke-data.csv --output-dir data-pre --scale minmax

# Test size 30%, không cap outliers
python prepare-stroke.py --input data-raw/healthcare-dataset-stroke-data.csv --output-dir data-pre --test-size 0.3
```

### Team Collaboration

```python
from model_consolidation import ModelResultsConsolidator

consolidator = ModelResultsConsolidator()
consolidator.add_model_result("Random Forest", "Member A", y_true, y_pred)
consolidator.print_summary()
consolidator.visualize_results()
```

---

## 📚 Dependencies

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

## 📖 Documentation

- **Detailed Report**: Xem `REPORT_TEMPLATE.md`
- **AI Guidelines**: `.github/copilot-instructions.md`
- **Code Structure**: Tất cả scripts có docstrings Vietnamese

---

## 🏆 Project Highlights

- ✅ **Reproducible Pipeline**: Seed-controlled, artifact-based
- ✅ **Class Imbalance Handling**: SMOTE + Stratified sampling
- ✅ **Multi-method Feature Selection**: 4 different approaches
- ✅ **Comprehensive EDA**: Statistical + Visual analysis
- ✅ **Team Collaboration**: Results consolidation framework
- ✅ **Production Ready**: Error handling, Vietnamese docs

---

**📝 Note**: Đây là pipeline hoàn chỉnh cho phân tích dữ liệu stroke prediction. Mỗi script có thể chạy độc lập hoặc theo workflow trên.
