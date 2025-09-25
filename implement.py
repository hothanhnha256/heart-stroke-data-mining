import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train = pd.read_csv("data-pre/train_preprocessed.csv")
test  = pd.read_csv("data-pre/test_preprocessed.csv")

y_train = train["stroke"].astype(int); X_train = train.drop(columns=["stroke"])
y_test  = test["stroke"].astype(int);  X_test  = test.drop(columns=["stroke"])

clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test), digits=4))
