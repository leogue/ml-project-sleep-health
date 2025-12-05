import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/raw_dataset.csv")
df.head()

# Drop Unnecessary Columns
df = df.drop('Person ID', axis=1)

# Split Blood Pressure
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df = df.drop('Blood Pressure', axis=1)

# Calculate Pulse Pressure
df['Pulse Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']

# Handle Missing Values
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')


# Label Encoding for ordinal/binary variables
categorical_cols = ['Gender']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# One-Hot Encoding for nominal variables
df = pd.get_dummies(df, columns=['Occupation'], prefix='Occupation', dtype=int)

# Encoding for Sleep Disorder (Target)
sleep_disorder_mapping = {
    'None': 0,
    'Insomnia': 1,
    'Sleep Apnea': 2
}
df['Sleep Disorder'] = df['Sleep Disorder'].map(sleep_disorder_mapping)

# Ordinal Encoding for BMI Category
bmi_mapping = {
    'Normal': 0,
    'Normal Weight': 0,
    'Overweight': 1,
    'Obese': 2
}
df['BMI Category'] = df['BMI Category'].map(bmi_mapping)

# Normalization
numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                  'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 
                  'Diastolic_BP', 'Pulse Pressure']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save a processed dataset and a validation dataset (95/05 split)
train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, stratify=df['Sleep Disorder'])
train_df.to_csv("../data/processed_dataset.csv", index=False)
val_df.to_csv("../data/validation_dataset.csv", index=False)

print(df.head())

