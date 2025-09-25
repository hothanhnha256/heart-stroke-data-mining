import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load raw dataset"""
    data_path = Path("data-raw/healthcare-dataset-stroke-data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset không tìm thấy: {data_path}")
    return pd.read_csv(data_path)

def basic_info(df):
    """In thông tin cơ bản về dataset"""
    print("=== THÔNG TIN DATASET ===")
    print(f"Số dòng: {len(df):,}")
    print(f"Số cột: {len(df.columns)}")
    print(f"Tỷ lệ stroke (positive): {df['stroke'].mean():.4f}")
    print(f"Số lượng stroke cases: {df['stroke'].sum()}")
    
    print("\n=== KIỂU DỮ LIỆU ===")
    print(df.dtypes)
    
    print("\n=== GIÁ TRỊ THIẾU ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])

def target_analysis(df):
    """Phân tích biến target"""
    print("\n=== PHÂN TÍCH TARGET (STROKE) ===")
    target_counts = df['stroke'].value_counts()
    print(f"No Stroke (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
    print(f"Stroke (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
    
    # Visualize target distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    df['stroke'].value_counts().plot(kind='bar', ax=ax1, color=['lightblue', 'salmon'])
    ax1.set_title('Phân phối Target Variable')
    ax1.set_xlabel('Stroke')
    ax1.set_ylabel('Số lượng')
    ax1.tick_params(axis='x', rotation=0)
    
    # Pie chart
    ax2.pie(target_counts.values, labels=['No Stroke', 'Stroke'], autopct='%1.2f%%', 
            colors=['lightblue', 'salmon'])
    ax2.set_title('Tỷ lệ Stroke')
    
    plt.tight_layout()
    plt.savefig('eda/eda_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def numeric_analysis(df):
    """Phân tích các biến số"""
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    
    print("\n=== THỐNG KÊ MÔ TẢ - BIẾN SỐ ===")
    print(df[numeric_cols].describe())
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        # Histogram
        df[col].hist(bins=50, ax=axes[i], alpha=0.7, color='skyblue')
        axes[i].set_title(f'Phân phối {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Tần số')
        
        # Box plot by target
        df.boxplot(column=col, by='stroke', ax=axes[i+3])
        axes[i+3].set_title(f'{col} theo Stroke')
        axes[i+3].set_xlabel('Stroke')
        axes[i+3].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig('eda/eda_numeric_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def categorical_analysis(df):
    """Phân tích các biến categorical"""
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    print("\n=== PHÂN TÍCH BIẾN CATEGORICAL ===")
    for col in categorical_cols:
        print(f"\n{col.upper()}:")
        print(df[col].value_counts())
        
        # Stroke rate by category
        stroke_by_cat = df.groupby(col)['stroke'].agg(['count', 'sum', 'mean']).round(4)
        stroke_by_cat.columns = ['Total', 'Stroke_Count', 'Stroke_Rate']
        print(f"\nStroke rate by {col}:")
        print(stroke_by_cat)
    
    # Visualize categorical variables
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.ravel()
    
    for i, col in enumerate(categorical_cols):
        # Count plot with stroke hue
        sns.countplot(data=df, x=col, hue='stroke', ax=axes[i])
        axes[i].set_title(f'{col} Distribution by Stroke')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='Stroke', labels=['No', 'Yes'])
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('eda/eda_categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(df):
    """Phân tích correlation"""
    # Encode categorical variables for correlation
    df_encoded = df.copy()
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    for col in categorical_cols:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    
    # Handle missing values
    df_encoded = df_encoded.fillna(df_encoded.median())
    
    # Correlation matrix
    corr_matrix = df_encoded.corr()
    
    print("\n=== CORRELATION VỚI TARGET ===")
    target_corr = corr_matrix['stroke'].abs().sort_values(ascending=False)
    print(target_corr)
    
    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Ma trận Correlation')
    plt.tight_layout()
    plt.savefig('eda/eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def age_analysis(df):
    """Phân tích chi tiết về age - biến quan trọng"""
    print("\n=== PHÂN TÍCH CHI TIẾT: TUỔI ===")
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 100], 
                            labels=['<30', '30-50', '50-65', '65+'])
    
    age_stroke = df.groupby('age_group')['stroke'].agg(['count', 'sum', 'mean'])
    age_stroke.columns = ['Total', 'Stroke_Count', 'Stroke_Rate']
    print(age_stroke)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Age distribution by stroke
    df[df['stroke']==0]['age'].hist(alpha=0.7, bins=30, label='No Stroke', ax=ax1, color='lightblue')
    df[df['stroke']==1]['age'].hist(alpha=0.7, bins=30, label='Stroke', ax=ax1, color='salmon')
    ax1.set_xlabel('Tuổi')
    ax1.set_ylabel('Tần số')
    ax1.set_title('Phân phối Tuổi theo Stroke')
    ax1.legend()
    
    # Stroke rate by age group
    age_stroke['Stroke_Rate'].plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title('Tỷ lệ Stroke theo Nhóm tuổi')
    ax2.set_xlabel('Nhóm tuổi')
    ax2.set_ylabel('Tỷ lệ Stroke')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda/eda_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main EDA workflow"""
    print("Bắt đầu EDA cho Stroke Prediction Dataset...")
    
    # Load data
    df = load_data()
    
    # Basic information
    basic_info(df)
    
    # Target analysis
    target_analysis(df)
    
    # Numeric variables
    numeric_analysis(df)
    
    # Categorical variables
    categorical_analysis(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Detailed age analysis
    age_analysis(df)
    
    print("\n=== EDA HOÀN THÀNH ===")
    print("Các file hình ảnh đã được lưu:")
    print("- eda_target_distribution.png")
    print("- eda_numeric_analysis.png") 
    print("- eda_categorical_analysis.png")
    print("- eda_correlation_matrix.png")
    print("- eda_age_analysis.png")

if __name__ == "__main__":
    main()