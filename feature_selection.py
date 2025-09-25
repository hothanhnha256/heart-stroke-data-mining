import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FeatureSelector:
    def __init__(self, data_path="data-raw/healthcare-dataset-stroke-data.csv"):
        self.data_path = data_path
        self.df = None
        self.X_encoded = None
        self.y = None
        self.feature_scores = {}
        
    def load_and_prepare_data(self):
        """Load và chuẩn bị dữ liệu cho feature selection"""
        print("Đang load và chuẩn bị dữ liệu...")
        
        self.df = pd.read_csv(self.data_path)
        
        # Remove ID column
        self.df = self.df.drop('id', axis=1)
        
        # Separate features and target
        self.y = self.df['stroke']
        X = self.df.drop('stroke', axis=1)
        
        # Encode categorical variables
        self.X_encoded = X.copy()
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            self.X_encoded[col] = le.fit_transform(self.X_encoded[col].astype(str))
            label_encoders[col] = le
        
        # Handle missing values
        numeric_cols = ['age', 'avg_glucose_level', 'bmi']
        for col in numeric_cols:
            self.X_encoded[col] = self.X_encoded[col].fillna(self.X_encoded[col].median())
        
        print(f"Dataset shape: {self.X_encoded.shape}")
        print(f"Features: {list(self.X_encoded.columns)}")
        
    def correlation_analysis(self):
        """Phân tích correlation với target"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Calculate correlations
        df_with_target = self.X_encoded.copy()
        df_with_target['stroke'] = self.y
        
        correlations = df_with_target.corr()['stroke'].abs().sort_values(ascending=False)
        correlations = correlations.drop('stroke')  # Remove self-correlation
        
        self.feature_scores['correlation'] = correlations
        
        print("Top features by correlation:")
        for feature, score in correlations.head(10).items():
            print(f"{feature}: {score:.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        correlations.plot(kind='bar', color='skyblue')
        plt.title('Feature Correlation với Target (Stroke)')
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature/feature_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlations
    
    def mutual_information_analysis(self):
        """Information Gain (Mutual Information) analysis"""
        print("\n=== MUTUAL INFORMATION ANALYSIS ===")
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(self.X_encoded, self.y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=self.X_encoded.columns).sort_values(ascending=False)
        
        self.feature_scores['mutual_info'] = mi_scores
        
        print("Top features by Mutual Information:")
        for feature, score in mi_scores.head(10).items():
            print(f"{feature}: {score:.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        mi_scores.plot(kind='bar', color='lightcoral')
        plt.title('Feature Mutual Information với Target')
        plt.xlabel('Features')
        plt.ylabel('Mutual Information Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature/feature_mutual_info_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return mi_scores
    
    def random_forest_importance(self):
        """Feature importance từ Random Forest"""
        print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(self.X_encoded, self.y)
        
        # Get feature importance
        rf_importance = pd.Series(rf.feature_importances_, 
                                index=self.X_encoded.columns).sort_values(ascending=False)
        
        self.feature_scores['rf_importance'] = rf_importance
        
        print("Top features by Random Forest importance:")
        for feature, score in rf_importance.head(10).items():
            print(f"{feature}: {score:.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        rf_importance.plot(kind='bar', color='lightgreen')
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature/feature_rf_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return rf_importance
    
    def statistical_tests(self):
        """Statistical tests: Chi-square cho categorical, ANOVA cho numeric"""
        print("\n=== STATISTICAL TESTS ===")
        
        # Separate numeric and categorical features
        numeric_cols = ['age', 'avg_glucose_level', 'bmi']
        categorical_cols = [col for col in self.X_encoded.columns if col not in numeric_cols]
        
        all_scores = {}
        
        # ANOVA F-test for numeric features
        if numeric_cols:
            f_scores, f_pvalues = f_classif(self.X_encoded[numeric_cols], self.y)
            for i, col in enumerate(numeric_cols):
                all_scores[col] = f_scores[i]
                print(f"ANOVA F-test {col}: score={f_scores[i]:.4f}, p-value={f_pvalues[i]:.4f}")
        
        # Chi-square test for categorical features  
        if categorical_cols:
            chi2_scores, chi2_pvalues = chi2(self.X_encoded[categorical_cols], self.y)
            for i, col in enumerate(categorical_cols):
                all_scores[col] = chi2_scores[i]
                print(f"Chi-square {col}: score={chi2_scores[i]:.4f}, p-value={chi2_pvalues[i]:.4f}")
        
        # Combine and sort scores
        statistical_scores = pd.Series(all_scores).sort_values(ascending=False)
        self.feature_scores['statistical'] = statistical_scores
        
        # Visualize
        plt.figure(figsize=(10, 6))
        statistical_scores.plot(kind='bar', color='gold')
        plt.title('Statistical Test Scores (ANOVA + Chi-square)')
        plt.xlabel('Features')
        plt.ylabel('Test Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature/feature_statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return statistical_scores
    
    def combined_ranking(self):
        """Kết hợp tất cả các phương pháp để tạo ranking tổng hợp"""
        print("\n=== COMBINED FEATURE RANKING ===")
        
        # Normalize all scores to 0-1 range
        normalized_scores = {}
        for method, scores in self.feature_scores.items():
            # Min-max normalization
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = pd.Series(1.0, index=scores.index)
            normalized_scores[method] = normalized
        
        # Create DataFrame with all normalized scores
        ranking_df = pd.DataFrame(normalized_scores)
        
        # Calculate combined score (average of all methods)
        ranking_df['combined_score'] = ranking_df.mean(axis=1)
        ranking_df = ranking_df.sort_values('combined_score', ascending=False)
        
        print("Combined Feature Ranking:")
        print(ranking_df.round(4))
        
        # Visualize combined ranking
        plt.figure(figsize=(12, 8))
        
        # Heatmap of all scores
        plt.subplot(2, 1, 1)
        sns.heatmap(ranking_df.drop('combined_score', axis=1).T, 
                   annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Normalized Scores từ các phương pháp Feature Selection')
        
        # Combined score bar plot
        plt.subplot(2, 1, 2)
        ranking_df['combined_score'].plot(kind='bar', color='purple')
        plt.title('Combined Feature Importance Score')
        plt.xlabel('Features')
        plt.ylabel('Combined Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature/feature_combined_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return ranking_df
    
    def select_top_features(self, top_k=10):
        """Chọn top K features quan trọng nhất"""
        print(f"\n=== TOP {top_k} FEATURES ===")
        
        # Run all analyses if not done yet
        if not self.feature_scores:
            self.correlation_analysis()
            self.mutual_information_analysis() 
            self.random_forest_importance()
            self.statistical_tests()
        
        ranking_df = self.combined_ranking()
        
        top_features = ranking_df.head(top_k).index.tolist()
        
        print(f"Top {top_k} features được chọn:")
        for i, feature in enumerate(top_features, 1):
            score = ranking_df.loc[feature, 'combined_score']
            print(f"{i:2d}. {feature}: {score:.4f}")
        
        # Save results
        results = {
            'top_features': top_features,
            'ranking_dataframe': ranking_df,
            'individual_scores': self.feature_scores
        }
        
        # Save to JSON for later use
        import json
        output_path = "feature/feature_selection_results.json"
        results_json = {
            'top_features': top_features,
            'feature_scores': {
                method: scores.to_dict() 
                for method, scores in self.feature_scores.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
        
        print(f"\nKết quả đã được lưu vào: {output_path}")
        
        return results

def main():
    """Main workflow cho feature selection"""
    print("=== FEATURE SELECTION CHO STROKE PREDICTION ===\n")
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Load và prepare data
    selector.load_and_prepare_data()
    
    # Run all feature selection methods
    print("Chạy các phương pháp feature selection...")
    
    selector.correlation_analysis()
    selector.mutual_information_analysis()
    selector.random_forest_importance() 
    selector.statistical_tests()
    
    # Get final results
    results = selector.select_top_features(top_k=8)  # Chọn top 8 features
    
    print("\n=== FEATURE SELECTION HOÀN THÀNH ===")
    print("Các file kết quả:")
    print("- feature_correlation_analysis.png")
    print("- feature_mutual_info_analysis.png")
    print("- feature_rf_importance_analysis.png")
    print("- feature_statistical_analysis.png")
    print("- feature_combined_ranking.png")
    print("- feature_selection_results.json")

if __name__ == "__main__":
    main()