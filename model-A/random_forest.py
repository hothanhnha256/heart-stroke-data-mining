from sklearn.ensemble import RandomForestClassifier
import pandas as pd
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

# 2. Khởi tạo và huấn luyện
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    max_features="sqrt",
    random_state=42
)
rf.fit(X_train, y_train)

# 3. Dự đoán và đánh giá
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Báo cáo phân loại Random Forest:")
print(classification_report(y_test, y_pred_rf))

auc_rf = roc_auc_score(y_test, y_prob_rf)
print("ROC-AUC:", auc_rf)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr, tpr, label=f"RandomForest (AUC={auc_rf:.2f})")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()
