import joblib

def save_model(model, filename='models/insurance_model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='models/insurance_model.pkl'):
    return joblib.load(filename)
