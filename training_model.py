import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_csv(r"C:\Users\akhia\Downloads\laptop_cleaned.csv")

# Select features and target
X = df[['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys', 'ScreenResolution', 'Memory', 'Ram']]
y = df['Price']

# Preprocessing
categorical_features = ['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys', 'ScreenResolution', 'Memory', 'Ram']
preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and save
model_pipeline.fit(X_train, y_train)

# Save the pipeline
with open("model_pipeline.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

