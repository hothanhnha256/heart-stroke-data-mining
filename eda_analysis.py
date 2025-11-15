import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style - Data Storytelling Theme
plt.style.use('default')
sns.set_palette("husl")

# Custom color palette - Professional Theme
COLORS = {
    'stroke': '#e74c3c',      # Coral Red for stroke cases
    'healthy': '#1abc9c',     # Turquoise for healthy
    'bg': '#ecf0f1',         # Light gray background
    'primary': '#2c3e50',     # Navy blue primary
    'secondary': '#e74c3c',   # Coral red secondary
    'accent': '#3498db'       # Bright blue accent
}

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
    """Phân tích biến target - Data Storytelling Style"""
    print("\n=== PHÂN TÍCH TARGET (STROKE) ===")
    target_counts = df['stroke'].value_counts()
    print(f"No Stroke (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
    print(f"Stroke (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
    
    # Data storytelling visualization
    x = pd.DataFrame(df.groupby(['stroke'])['stroke'].count())
    healthy_pct = (x.stroke[0] / (x.stroke[0] + x.stroke[1])) * 100
    stroke_pct = (x.stroke[1] / (x.stroke[0] + x.stroke[1])) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Horizontal bars
    ax.barh([1], x.stroke[0], height=0.7, color=COLORS['healthy'])
    ax.barh([0], x.stroke[1], height=0.7, color=COLORS['stroke'])
    
    # Labels for bars
    plt.text(-250, 0.92, 'Healthy', 
             {'font': 'Serif', 'weight': 'bold', 'size': 16, 'color': COLORS['healthy']})
    plt.text(x.stroke[0] + 150, 0.92, f'{healthy_pct:.1f}%', 
             {'font': 'Serif', 'weight': 'bold', 'size': 16, 'color': COLORS['healthy']})
    
    plt.text(-250, -0.08, 'Stroke', 
             {'font': 'Serif', 'weight': 'bold', 'size': 16, 'color': COLORS['stroke']})
    plt.text(x.stroke[1] + 50, -0.08, f'{stroke_pct:.1f}%', 
             {'font': 'Serif', 'weight': 'bold', 'size': 16, 'color': COLORS['stroke']})
    
    # Background
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # Title and description
    plt.text(-250, 1.77, 'Phân Phối Tỷ Lệ Đột Quỵ Trong Dataset', 
             {'font': 'Serif', 'size': 22, 'weight': 'bold', 'color': 'black'})
    
    # Legend
    legend_y = 1.60
    plt.text(x.stroke[0] - 900, legend_y, 'Stroke', 
             {'font': 'Serif', 'weight': 'bold', 'size': 14, 'color': COLORS['stroke']})
    plt.text(x.stroke[0] - 600, legend_y, '|', 
             {'color': 'black', 'size': 14, 'weight': 'bold'})
    plt.text(x.stroke[0] - 550, legend_y, 'Healthy', 
             {'font': 'Serif', 'weight': 'bold', 'size': 14, 'color': COLORS['healthy']})
    
    # Description
    plt.text(-250, 1.42, 
             f'Dataset có sự mất cân bằng nghiêm trọng:\n'
             f'Chỉ {stroke_pct:.1f}% (249 người) bị đột quỵ trong tổng số {len(df):,} bệnh nhân.\n'
             f'Tỷ lệ imbalance: 1:{int(healthy_pct/stroke_pct)}', 
             {'font': 'Serif', 'size': 11, 'color': '#333333'})
    
    # Remove axes
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('eda/eda_target_distribution.png', dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()

def numeric_analysis(df):
    """Phân tích các biến số - Enhanced Style"""
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    
    print("\n=== THỐNG KÊ MÔ TẢ - BIẾN SỐ ===")
    print(df[numeric_cols].describe())
    
    # Enhanced distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    fig.patch.set_facecolor(COLORS['bg'])
    
    col_names = ['Tuổi (Age)', 'Glucose Trung Bình', 'BMI']
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.set_facecolor(COLORS['bg'])
        
        # Violin plot cho stroke vs non-stroke
        parts = ax.violinplot([df[df['stroke']==0][col].dropna(), 
                               df[df['stroke']==1][col].dropna()],
                              positions=[1, 2],
                              showmeans=True,
                              showmedians=True,
                              widths=0.7)
        
        # Color the violin plots
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS['primary'])
            pc.set_alpha(0.3)
        
        # Overlay with box plots
        bp = ax.boxplot([df[df['stroke']==0][col].dropna(), 
                        df[df['stroke']==1][col].dropna()],
                       positions=[1, 2],
                       widths=0.3,
                       patch_artist=True,
                       boxprops=dict(facecolor=COLORS['healthy'], alpha=0.6),
                       medianprops=dict(color=COLORS['stroke'], linewidth=2),
                       whiskerprops=dict(color=COLORS['primary']),
                       capprops=dict(color=COLORS['primary']))
        
        # Color second box differently
        bp['boxes'][1].set_facecolor(COLORS['stroke'])
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Healthy', 'Stroke'], fontsize=11, weight='bold')
        ax.set_ylabel(col_names[i], fontsize=12, weight='bold', color=COLORS['primary'])
        ax.set_title(f'{col_names[i]} Distribution', fontsize=13, weight='bold', 
                    color=COLORS['primary'], pad=10)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['primary'])
        ax.spines['bottom'].set_color(COLORS['primary'])
        ax.tick_params(colors=COLORS['primary'])
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS['primary'])
    
    plt.suptitle('Phân Tích Phân Phối Các Biến Số Theo Tình Trạng Đột Quỵ', 
                fontsize=16, weight='bold', color='black', y=1.02)
    plt.tight_layout()
    plt.savefig('eda/eda_numeric_analysis.png', dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()

def categorical_analysis(df):
    """Phân tích các biến categorical - Storytelling Style"""
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
    
    # Enhanced visualizations
    fig = plt.figure(figsize=(18, 12), dpi=100)
    fig.patch.set_facecolor(COLORS['bg'])
    
    col_names = {
        'gender': 'Giới Tính',
        'ever_married': 'Tình Trạng Hôn Nhân',
        'work_type': 'Loại Công Việc',
        'Residence_type': 'Nơi Cư Trú',
        'smoking_status': 'Tình Trạng Hút Thuốc'
    }
    
    for idx, col in enumerate(categorical_cols):
        ax = plt.subplot(2, 3, idx + 1)
        ax.set_facecolor(COLORS['bg'])
        
        # Calculate stroke rates
        stroke_data = df.groupby(col)['stroke'].agg(['sum', 'count'])
        stroke_data['rate'] = (stroke_data['sum'] / stroke_data['count']) * 100
        stroke_data = stroke_data.sort_values('rate', ascending=True)
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(stroke_data)), stroke_data['rate'], 
                      color=[COLORS['stroke'] if x > 5 else COLORS['healthy'] 
                            for x in stroke_data['rate']],
                      alpha=0.8, height=0.6)
        
        # Add value labels
        for i, (idx_val, row) in enumerate(stroke_data.iterrows()):
            ax.text(row['rate'] + 0.3, i, f"{row['rate']:.1f}%", 
                   va='center', fontsize=9, weight='bold',
                   color=COLORS['stroke'] if row['rate'] > 5 else COLORS['primary'])
        
        ax.set_yticks(range(len(stroke_data)))
        ax.set_yticklabels(stroke_data.index, fontsize=10, color=COLORS['primary'])
        ax.set_xlabel('Tỷ lệ Đột Quỵ (%)', fontsize=10, weight='bold', color=COLORS['primary'])
        ax.set_title(col_names[col], fontsize=12, weight='bold', 
                    color=COLORS['primary'], pad=10)
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['primary'])
        ax.spines['bottom'].set_color(COLORS['primary'])
        ax.tick_params(colors=COLORS['primary'])
        ax.grid(axis='x', alpha=0.3, linestyle='--', color=COLORS['primary'])
    
    # Remove the 6th subplot
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_facecolor(COLORS['bg'])
    
    # Add summary text in the empty space
    summary_text = (
        "Key Insights:\n\n"
        "- Người cao tuổi có nguy cơ cao hơn\n"
        "- Tình trạng hôn nhân liên quan đến tuổi\n"
        "- Self-employed có tỷ lệ cao nhất\n"
        "- Formerly smoked: nguy cơ tăng\n"
        "- Nơi cư trú: không ảnh hưởng nhiều"
    )
    ax6.text(0.1, 0.5, summary_text, fontsize=11, 
            color=COLORS['primary'], weight='bold',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['stroke']))
    
    plt.suptitle('Tỷ Lệ Đột Quỵ Theo Các Yếu Tố Phân Loại', 
                fontsize=16, weight='bold', color='black', y=0.98)
    plt.tight_layout()
    plt.savefig('eda/eda_categorical_analysis.png', dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()

def correlation_analysis(df):
    """Phân tích correlation - Modern Style"""
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
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 6), dpi=100)
    fig.patch.set_facecolor(COLORS['bg'])
    
    # 1. Heatmap
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor(COLORS['bg'])
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdPu', center=0, 
                square=True, fmt='.2f', ax=ax1,
                cbar_kws={'label': 'Correlation'},
                linewidths=0.5, linecolor='white')
    ax1.set_title('Ma Trận Correlation Toàn Bộ', fontsize=14, weight='bold', 
                 color=COLORS['primary'], pad=15)
    
    # 2. Bar chart of correlations with target
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor(COLORS['bg'])
    
    target_corr_sorted = corr_matrix['stroke'].drop('stroke').abs().sort_values(ascending=True)
    
    colors_bar = [COLORS['stroke'] if x > 0.1 else COLORS['healthy'] for x in target_corr_sorted]
    bars = ax2.barh(range(len(target_corr_sorted)), target_corr_sorted, 
                   color=colors_bar, alpha=0.8, height=0.6)
    
    # Add value labels
    for i, val in enumerate(target_corr_sorted):
        ax2.text(val + 0.005, i, f'{val:.3f}', 
                va='center', fontsize=9, weight='bold', color=COLORS['primary'])
    
    ax2.set_yticks(range(len(target_corr_sorted)))
    ax2.set_yticklabels(target_corr_sorted.index, fontsize=10, color=COLORS['primary'])
    ax2.set_xlabel('Absolute Correlation', fontsize=11, weight='bold', color=COLORS['primary'])
    ax2.set_title('Correlation Với Biến Target (Stroke)', fontsize=14, weight='bold', 
                 color=COLORS['primary'], pad=15)
    
    # Add threshold line
    ax2.axvline(x=0.1, color=COLORS['stroke'], linestyle='--', alpha=0.5, linewidth=2)
    ax2.text(0.1, len(target_corr_sorted)-0.5, '  Ngưỡng: 0.1', 
            fontsize=9, color=COLORS['stroke'], weight='bold')
    
    # Styling
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['primary'])
    ax2.spines['bottom'].set_color(COLORS['primary'])
    ax2.tick_params(colors=COLORS['primary'])
    ax2.grid(axis='x', alpha=0.3, linestyle='--', color=COLORS['primary'])
    
    plt.suptitle('Phân Tích Mối Tương Quan Giữa Các Biến', 
                fontsize=16, weight='bold', color='black', y=0.98)
    plt.tight_layout()
    plt.savefig('eda/eda_correlation_matrix.png', dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()

def age_analysis(df):
    """Phân tích chi tiết về age - Storytelling Style"""
    print("\n=== PHÂN TÍCH CHI TIẾT: TUỔI ===")
    
    # Age groups
    df_temp = df.copy()
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=[0, 30, 50, 65, 100], 
                            labels=['<30', '30-50', '50-65', '65+'])
    
    age_stroke = df_temp.groupby('age_group')['stroke'].agg(['count', 'sum', 'mean'])
    age_stroke.columns = ['Total', 'Stroke_Count', 'Stroke_Rate']
    print(age_stroke)
    
    # Create storytelling visualization
    fig = plt.figure(figsize=(16, 6), dpi=100)
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Left plot: Age distribution with KDE
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor(COLORS['bg'])
    
    # KDE plots
    df_temp[df_temp['stroke']==0]['age'].plot(kind='kde', ax=ax1, 
                                              color=COLORS['healthy'], 
                                              linewidth=3, alpha=0.7, 
                                              label='Healthy')
    df_temp[df_temp['stroke']==1]['age'].plot(kind='kde', ax=ax1, 
                                              color=COLORS['stroke'], 
                                              linewidth=3, alpha=0.7, 
                                              label='Stroke')
    
    # Fill under curves
    ax1.fill_between(df_temp[df_temp['stroke']==0]['age'].sort_values(),
                    0, 
                    df_temp[df_temp['stroke']==0]['age'].sort_values().map(
                        df_temp[df_temp['stroke']==0]['age'].value_counts(normalize=True).sort_index()
                    ).fillna(0),
                    alpha=0.2, color=COLORS['healthy'])
    
    ax1.set_xlabel('Tuổi (Age)', fontsize=12, weight='bold', color=COLORS['primary'])
    ax1.set_ylabel('Mật độ (Density)', fontsize=12, weight='bold', color=COLORS['primary'])
    ax1.set_title('Phân Phối Tuổi: Stroke vs Healthy', fontsize=14, weight='bold', 
                 color=COLORS['primary'], pad=15)
    ax1.legend(fontsize=11, frameon=True, facecolor='white', edgecolor=COLORS['primary'])
    ax1.grid(alpha=0.3, linestyle='--', color=COLORS['primary'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['primary'])
    ax1.spines['bottom'].set_color(COLORS['primary'])
    ax1.tick_params(colors=COLORS['primary'])
    
    # Right plot: Stroke rate by age group with annotations
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor(COLORS['bg'])
    
    age_stroke_pct = age_stroke['Stroke_Rate'] * 100
    colors_age = [COLORS['healthy'] if x < 5 else COLORS['stroke'] for x in age_stroke_pct]
    
    bars = ax2.bar(range(len(age_stroke_pct)), age_stroke_pct, 
                  color=colors_age, alpha=0.8, width=0.6,
                  edgecolor=COLORS['primary'], linewidth=2)
    
    # Add value labels and arrows
    for i, (idx, val) in enumerate(age_stroke_pct.items()):
        # Value on top of bar
        ax2.text(i, val + 0.8, f'{val:.1f}%', ha='center', va='bottom',
                fontsize=12, weight='bold', color=COLORS['primary'])
        
        # Sample size below
        ax2.text(i, -1.5, f'n={age_stroke.loc[idx, "Total"]}', ha='center', va='top',
                fontsize=9, color=COLORS['primary'], style='italic')
    
    # Add trend arrow
    if len(age_stroke_pct) > 1:
        ax2.annotate('', xy=(3, age_stroke_pct.iloc[3]), xytext=(0, age_stroke_pct.iloc[0]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['stroke'], 
                                  lw=2, linestyle='--', alpha=0.6))
        ax2.text(1.5, max(age_stroke_pct)/2, 'Xu hướng\ntăng', 
                fontsize=10, color=COLORS['stroke'], weight='bold',
                ha='center', style='italic')
    
    ax2.set_xticks(range(len(age_stroke_pct)))
    ax2.set_xticklabels(age_stroke_pct.index, fontsize=11, weight='bold', color=COLORS['primary'])
    ax2.set_ylabel('Tỷ Lệ Đột Quỵ (%)', fontsize=12, weight='bold', color=COLORS['primary'])
    ax2.set_xlabel('Nhóm Tuổi', fontsize=12, weight='bold', color=COLORS['primary'])
    ax2.set_title('Tỷ Lệ Đột Quỵ Tăng Theo Tuổi', fontsize=14, weight='bold', 
                 color=COLORS['primary'], pad=15)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['primary'])
    ax2.spines['bottom'].set_color(COLORS['primary'])
    ax2.tick_params(colors=COLORS['primary'])
    ax2.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS['primary'])
    ax2.set_ylim(-3, max(age_stroke_pct) * 1.15)
    
    plt.suptitle('Phân Tích Chi Tiết: Tuổi Là Yếu Tố Quan Trọng Nhất', 
                fontsize=16, weight='bold', color='black', y=0.98)
    plt.tight_layout()
    plt.savefig('eda/eda_age_analysis.png', dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
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