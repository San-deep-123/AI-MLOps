import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Ensure output directories exist
os.makedirs('artifacts', exist_ok=True)

# 1. Load the original training data
df = pd.read_csv('data/employee.csv')

X = df.drop(columns=['empid', 'name', 'eligible_for_promotion'])
y = df['eligible_for_promotion'].astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define Preprocessing Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['department', 'techstack']),
        ('num', StandardScaler(), ['income_lpa', 'experience'])
    ],
    remainder='passthrough'
)

# 3. Create and Fit Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
print(f"Training Complete. Validation Accuracy: {pipeline.score(X_test, y_test)*100:.2f}%")

# Save model binary asset
with open('artifacts/promotion_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Saved model file to artifacts/model.pkl")
