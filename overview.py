import pandas as pd
df = pd.read_csv('./data-raw/healthcare-dataset-stroke-data.csv', delimiter = ',', encoding = 'utf-8')
df.head(3).T
df.info()
# RangeIndex: 5110 entries, 0 to 5109
# Data columns (total 12 columns):
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   id                 5110 non-null   int64
#  1   gender             5110 non-null   object
#  2   age                5110 non-null   float64
#  3   hypertension       5110 non-null   int64
#  4   heart_disease      5110 non-null   int64
#  5   ever_married       5110 non-null   object
#  6   work_type          5110 non-null   object
#  7   Residence_type     5110 non-null   object
#  8   avg_glucose_level  5110 non-null   float64
#  9   bmi                4909 non-null   float64
#  10  smoking_status     5110 non-null   object
#  11  stroke             5110 non-null   int64
# dtypes: float64(3), int64(4), object(5)
# memory usage: 479.2+ KB


# Summary statistics

# Numerical columns summary
print(round (df.describe(exclude = 'object'), 2))
#              id      age  hypertension  heart_disease  avg_glucose_level      bmi   stroke
# count   5110.00  5110.00        5110.0        5110.00            5110.00  4909.00  5110.00
# mean   36517.83    43.23           0.1           0.05             106.15    28.89     0.05
# std    21161.72    22.61           0.3           0.23              45.28     7.85     0.22
# min       67.00     0.08           0.0           0.00              55.12    10.30     0.00
# 25%    17741.25    25.00           0.0           0.00              77.24    23.50     0.00
# 50%    36932.00    45.00           0.0           0.00              91.88    28.10     0.00
# 75%    54682.00    61.00           0.0           0.00             114.09    33.10     0.00
# max    72940.00    82.00           1.0           1.00             271.74    97.60     1.00

# Categorical columns summary
print(round (df.describe(exclude = ['float', 'int64']),2))
#         gender ever_married work_type Residence_type smoking_status
# count     5110         5110      5110           5110           5110
# unique       3            2         5              2              4
# top     Female          Yes   Private          Urban   never smoked
# freq      2994         3353      2925           2596           1892

# initial insight from overview of data
# 1. The dataset contains 5110 entries with 12 columns, including demographic and health-related features.
# 2. Data from bmi column has 201 missing values (5110 - 4909).
# 3. Both categorical and numerical features are present:
#     - Categorical : gender, ever_married, work_type, Residence_type, smoking_status
#     - Binary : hypertension, heart_disease, stroke
#     - Numerical : age, avg_glucose_level, bmi

# Percentage of stroke cases
stroke_rate = df['stroke'].mean() * 100
print(f"Percentage of stroke cases: {stroke_rate:.2f}%")
