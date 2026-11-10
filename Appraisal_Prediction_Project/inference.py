import pandas as pd
import pickle

# 1. Load the trained model pipeline asset
with open('artifacts/promotion_model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# 2. Read the unseen dataset
unseen_df = pd.read_csv('data/inference.csv')

# 3. Drop text identifiers to isolate the feature columns
X_unseen = unseen_df.drop(columns=['empid', 'name', 'eligible_for_promotion'])

# 4. Generate predictions and confidence probabilities
predictions = model_pipeline.predict(X_unseen)
probabilities = model_pipeline.predict_proba(X_unseen)[:, 1]

# 5. Append results back to the dataframe for comparison
unseen_df['Model_Prediction_Raw'] = predictions
unseen_df['Model_Prediction_Text'] = unseen_df['Model_Prediction_Raw'].map({1: 'Y', 0: 'N'})
unseen_df['Confidence_Score'] = (probabilities * 100).round(2)

print("\n=== BATCH INFERENCE PERFORMANCE REPORT ===")
print(f"Total Profiles Evaluated: {len(unseen_df)}\n")
print("=== VISUAL PREDICTION SNAPSHOT ===")
print(unseen_df[['empid', 'name', 'eligible_for_promotion', 'Model_Prediction_Text', 'Confidence_Score']].to_string(index=False))
