
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
try:
    df = pd.read_csv('Students Social Media Addiction.csv')
except FileNotFoundError:
    print("Dataset not found.")
    exit()

# Preprocessing
label_encoders = {}
categorical_columns = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 
                       'Affects_Academic_Performance', 'Relationship_Status']

df_encoded = df.copy()
for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

if 'Student_ID' in df_encoded.columns:
    df_encoded = df_encoded.drop('Student_ID', axis=1)

# Model Building
X = df_encoded.drop('Addicted_Score', axis=1)
y = df_encoded['Addicted_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")
print("-" * 30)
