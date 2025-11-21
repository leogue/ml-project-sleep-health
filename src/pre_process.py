import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("../data/raw_dataset.csv")
df.head()



# pre process

df = df.drop('Person ID', axis=1)

# Split Blood Pressure
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df = df.drop('Blood Pressure', axis=1)

# Calculate Pulse Pressure
df['Pulse Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']

# Handle Missing Values
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# Categorical Encoding
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"{col} mapping:")
    for i, item in enumerate(le.classes_):
        print(f"  {i}: {item}")

# Normalization
numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                  'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 
                  'Diastolic_BP', 'Pulse Pressure']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save processed dataset
df.to_csv("../data/processed_dataset.csv", index=False)

print("Preprocessing complete. Saved to ../data/processed_dataset.csv")
print(df.head())

