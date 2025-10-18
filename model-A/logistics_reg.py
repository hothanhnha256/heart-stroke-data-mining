import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Đường dẫn thư mục dữ liệu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data-pre")

# 1. Load dữ liệu train/test đã được xử lý
train = pd.read_csv(os.path.join(DATA_DIR, "train_preprocessed.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test_preprocessed.csv"))

X_train = train.drop("stroke", axis=1)
y_train = train["stroke"]

X_test = test.drop("stroke", axis=1)
y_test = test["stroke"]

# 2. Load meta (feature_names, preprocessor nếu cần)
feature_names = open(os.path.join(DATA_DIR, "feature_names.txt")).read().splitlines()
preprocessor = joblib.load(os.path.join(DATA_DIR, "preprocessor.joblib"))

print("Số đặc trưng đầu vào:", len(feature_names))

# 3. Huấn luyện Logistic Regression
log_reg = LogisticRegression(
    C=1.0, 
    penalty="l2", 
    solver="liblinear", 
    max_iter=1000, 
    random_state=42
)
log_reg.fit(X_train, y_train)

# 4. Dự đoán và đánh giá
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print("Báo cáo phân loại Logistic Regression:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"LogReg (AUC={auc:.2f})")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
