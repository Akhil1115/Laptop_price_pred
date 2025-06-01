import pickle

with open("model_pipeline.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

print("Model loaded successfully.")
