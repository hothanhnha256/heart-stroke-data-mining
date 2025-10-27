# 🚀 QUICK START GUIDE - Heart Stroke Prediction

## ⚡ Bắt đầu trong 5 phút

### 1. Setup Environment (2 phút)

```powershell
# Clone repository
git clone https://github.com/hothanhnha256/heart-stroke-data-mining.git
cd heart-stroke-data-mining

# Create và activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (3 phút)

```powershell
# Step 1: EDA (~30 giây)
python eda_analysis.py

# Step 2: Preprocessing (~1 phút)
python prepare-stroke.py `
  --input data-raw/healthcare-dataset-stroke-data.csv `
  --output-dir data-pre `
  --scale standard `
  --cap-outliers `
  --smote

# Step 3: Feature Selection (~30 giây)
python feature_selection.py

# Step 4: Baseline Model (~5 giây)
python implement.py

# Step 5: View Consolidation Demo (~5 giây)
python model_consolidation.py
```

### 3. Check Results

**Visualizations**:

- `eda/*.png` - Exploratory analysis charts
- `feature/*.png` - Feature selection plots
- `model_results_comparison.png` - Model comparison

**Data**:

- `data-pre/train_preprocessed.csv` - 7,778 balanced samples
- `data-pre/test_preprocessed.csv` - 1,022 test samples

**Reports**:

- `detailed_model_report.txt` - Detailed metrics
- `feature/feature_selection_results.json` - Top features

---

## 📊 Expected Outputs

### EDA

```
=== THÔNG TIN DATASET ===
Số dòng: 5,110
Số cột: 12
Tỷ lệ stroke: 0.0487
Missing: BMI (201 values)
```

### Preprocessing

```
=== PREP SUMMARY ===
Train size: 7,778 | Test size: 1,022
Train pos rate: 0.5000 | Test pos rate: 0.0487
#Features: 21
```

### Feature Selection

```
=== TOP 8 FEATURES ===
 1. age: 1.0000 ⭐⭐⭐⭐⭐
 2. avg_glucose_level: 0.3636 ⭐⭐⭐⭐
 3. hypertension: 0.2471 ⭐⭐⭐
 4. heart_disease: 0.2428 ⭐⭐⭐
 5. bmi: 0.2198 ⭐⭐⭐
 ...
```

---

## 🛠️ Common Commands

### Development

```powershell
# Check Python version
python --version  # Should be >= 3.9

# List installed packages
pip list

# Run specific model
python model-A/logistics_reg.py
python model-B/svm.py

# Open Jupyter notebook
jupyter notebook model-B/svm-and-knn.ipynb
```

### Customization

```powershell
# Different scaling method
python prepare-stroke.py --scale minmax

# No outlier capping
python prepare-stroke.py --scale standard

# Different test size
python prepare-stroke.py --test-size 0.25

# No SMOTE
python prepare-stroke.py --scale standard --cap-outliers
```

---

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Detailed Report**: See `REPORT.md`
- **AI Guidelines**: See `.github/copilot-instructions.md`

---

## 🆘 Troubleshooting

### Error: Module not found

```powershell
# Solution: Activate venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Error: File not found

```powershell
# Solution: Check working directory
pwd  # Should be in heart-stroke/
ls data-raw/  # Should see healthcare-dataset-stroke-data.csv
```

### Low Performance

```
# Normal với baseline model
# Try advanced models in model-A/, model-B/
```

---

**Happy Coding! 🎉**
