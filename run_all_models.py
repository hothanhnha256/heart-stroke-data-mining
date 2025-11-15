"""
Script tổng hợp chạy tất cả models và tạo báo cáo so sánh
=========================================================

Chạy tất cả models (Logistic Regression, Random Forest, SVM, KNN) 
và tạo báo cáo chi tiết với visualizations.

Usage:
    python run_all_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve
)

class ModelRunner:
    """Chạy và đánh giá tất cả models"""
    
    def __init__(self, data_dir='data-pre'):
        """Initialize với dữ liệu đã preprocessing"""
        self.data_dir = data_dir
        self.results = {}
        self.load_data()
        
    def load_data(self):
        """Load train/test data"""
        print("Loading preprocessed data...")
        train = pd.read_csv(f"{self.data_dir}/train_preprocessed.csv")
        test = pd.read_csv(f"{self.data_dir}/test_preprocessed.csv")
        
        self.X_train = train.drop("stroke", axis=1)
        self.y_train = train["stroke"]
        self.X_test = test.drop("stroke", axis=1)
        self.y_test = test["stroke"]
        
        print(f"   Train: {len(self.X_train)} samples")
        print(f"   Test: {len(self.X_test)} samples")
        print(f"   Features: {self.X_train.shape[1]}\n")
        
    def evaluate_model(self, model, model_name):
        """Đánh giá một model và lưu kết quả"""
        print(f"Training {model_name}...")
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_prob) if y_prob is not None else None,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'y_pred': y_pred,
            'y_prob': y_prob,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print results
        print(f"   Accuracy:  {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall:    {results['recall']:.4f}")
        print(f"   F1-Score:  {results['f1_score']:.4f}")
        if results['roc_auc']:
            print(f"   ✓ ROC-AUC:   {results['roc_auc']:.4f}")
        print()
        
        self.results[model_name] = results
        return results
        
    def run_all_models(self):
        """Chạy tất cả models"""
        print("=" * 60)
        print("RUNNING ALL MODELS - STROKE PREDICTION")
        print("=" * 60)
        print()
        
        # 1. Logistic Regression
        log_reg = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        self.evaluate_model(log_reg, "Logistic Regression")
        
        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.evaluate_model(rf, "Random Forest")
        
        # 3. SVM with RBF kernel
        svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
        self.evaluate_model(svm, "SVM (RBF)")
        
        # 4. K-Nearest Neighbors
        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        )
        self.evaluate_model(knn, "KNN (k=5)")
        
        print("=" * 60)
        print("ALL MODELS COMPLETED")
        print("=" * 60)
        print()
        
    def create_comparison_table(self):
        """Tạo bảng so sánh models"""
        df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': f"{res['accuracy']:.4f}",
                'Precision': f"{res['precision']:.4f}",
                'Recall': f"{res['recall']:.4f}",
                'F1-Score': f"{res['f1_score']:.4f}",
                'ROC-AUC': f"{res['roc_auc']:.4f}" if res['roc_auc'] else 'N/A'
            }
            for name, res in self.results.items()
        ])
        
        print("\nMODEL COMPARISON TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        # Save to CSV
        df.to_csv('models/model_comparison_results.csv', index=False)
        print("\nSaved to: models/model_comparison_results.csv")

        return df
        
    def plot_roc_curves(self):
        """Vẽ ROC curves cho tất cả models"""
        plt.figure(figsize=(10, 8))
        
        for name, res in self.results.items():
            if res['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, res['y_prob'])
                auc = res['roc_auc']
                plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig('models/model_roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved to: models/model_roc_curves_comparison.png")
        plt.show()
        
    def plot_metrics_comparison(self):
        """Vẽ biểu đồ so sánh metrics"""
        metrics_data = []
        for name, res in self.results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1_score']
            })
        
        df = pd.DataFrame(metrics_data)
        df_melted = df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='Set2')
        plt.ylim([0, 1.0])
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        plt.legend(title='Metric', fontsize=10)
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig('models/model_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Metrics comparison saved to: models/model_metrics_comparison.png")
        plt.show()
        
    def plot_confusion_matrices(self):
        """Vẽ confusion matrices cho tất cả models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, res) in enumerate(self.results.items()):
            cm = np.array(res['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Stroke', 'Stroke'],
                       yticklabels=['No Stroke', 'Stroke'],
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_title(f'{name}\n(Acc={res["accuracy"]:.4f}, F1={res["f1_score"]:.4f})', 
                               fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('models/model_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved to: models/model_confusion_matrices.png")
        plt.show()
        
    def generate_report(self):
        """Tạo báo cáo text chi tiết"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BÁO CÁO TỔNG HỢP KẾT QUẢ MODELS - STROKE PREDICTION")
        report_lines.append("=" * 80)
        report_lines.append(f"\nNgày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Tổng số models: {len(self.results)}")
        report_lines.append(f"Test set size: {len(self.y_test)} samples")
        report_lines.append(f"Stroke cases: {self.y_test.sum()} ({self.y_test.mean()*100:.2f}%)")
        report_lines.append("\n" + "=" * 80)
        
        # Chi tiết từng model
        report_lines.append("\nCHI TIẾT TỪNG MODEL:")
        report_lines.append("-" * 80)
        
        for idx, (name, res) in enumerate(self.results.items(), 1):
            report_lines.append(f"\n{idx}. {name}")
            report_lines.append(f"   Accuracy:  {res['accuracy']:.4f}")
            report_lines.append(f"   Precision: {res['precision']:.4f}")
            report_lines.append(f"   Recall:    {res['recall']:.4f}")
            report_lines.append(f"   F1-Score:  {res['f1_score']:.4f}")
            if res['roc_auc']:
                report_lines.append(f"   ROC-AUC:   {res['roc_auc']:.4f}")
            
            cm = np.array(res['confusion_matrix'])
            report_lines.append(f"\n   Confusion Matrix:")
            report_lines.append(f"   TN={cm[0,0]:<4} FP={cm[0,1]:<4}")
            report_lines.append(f"   FN={cm[1,0]:<4} TP={cm[1,1]:<4}")
        
        # Ranking
        report_lines.append("\n" + "=" * 80)
        report_lines.append("RANKING MODELS:")
        report_lines.append("-" * 80)
        
        # By F1-Score
        sorted_by_f1 = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        report_lines.append("\nTheo F1-Score:")
        for idx, (name, res) in enumerate(sorted_by_f1, 1):
            report_lines.append(f"   {idx}. {name}: {res['f1_score']:.4f}")
        
        # By ROC-AUC
        sorted_by_auc = sorted(
            [(n, r) for n, r in self.results.items() if r['roc_auc']], 
            key=lambda x: x[1]['roc_auc'], 
            reverse=True
        )
        report_lines.append("\nTheo ROC-AUC:")
        for idx, (name, res) in enumerate(sorted_by_auc, 1):
            report_lines.append(f"   {idx}. {name}: {res['roc_auc']:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("KẾT LUẬN:")
        report_lines.append("-" * 80)
        
        best_f1_model = sorted_by_f1[0][0]
        best_f1_score = sorted_by_f1[0][1]['f1_score']
        
        report_lines.append(f"\n🏆 Model tốt nhất (F1-Score): {best_f1_model} ({best_f1_score:.4f})")
        
        if sorted_by_auc:
            best_auc_model = sorted_by_auc[0][0]
            best_auc_score = sorted_by_auc[0][1]['roc_auc']
            report_lines.append(f"🏆 Model tốt nhất (ROC-AUC): {best_auc_model} ({best_auc_score:.4f})")
        
        report_lines.append("\nLưu ý:")
        report_lines.append("- Dataset có class imbalance nghiêm trọng (95% vs 5%)")
        report_lines.append("- F1-Score và ROC-AUC là metrics quan trọng hơn Accuracy")
        report_lines.append("- Recall cao quan trọng cho medical diagnosis (giảm False Negatives)")
        
        report_lines.append("\n" + "=" * 80)
        
        # Save to file
        report_text = "\n".join(report_lines)
        with open('models/models_final_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n💾 Full report saved to: models/models_final_report.txt")

        # Save JSON results
        results_json = {
            name: {k: v for k, v in res.items() if k not in ['y_pred', 'y_prob']}
            for name, res in self.results.items()
        }

        with open('models/models_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        print("💾 JSON results saved to: models/models_results.json")


def main():
    """Main workflow"""
    # Initialize và chạy tất cả models
    runner = ModelRunner()
    runner.run_all_models()
    
    # Tạo visualizations và reports
    print("\nCreating visualizations and reports...\n")
    runner.create_comparison_table()
    runner.plot_roc_curves()
    runner.plot_metrics_comparison()
    runner.plot_confusion_matrices()
    runner.generate_report()
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED!")
    print("=" * 80)
    print("\nOutput files:")
    print("  models/models_final_report.txt - Detailed text report")
    print("  models/models_results.json - JSON results")
    print("  models/model_comparison_results.csv - Comparison table")
    print("  models/model_roc_curves_comparison.png - ROC curves")
    print("  models/model_metrics_comparison.png - Metrics bar chart")
    print("  models/model_confusion_matrices.png - Confusion matrices")
    print("\n")


if __name__ == "__main__":
    main()
