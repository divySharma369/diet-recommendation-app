
import streamlit as st
import torch
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1️⃣ Define model
class DietRecommendationModel(nn.Module):
    def __init__(self, input_dim=11, hidden1=128, hidden2=64, output_dim=4, dropout_rate=0.3):
        super(DietRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 2️⃣ Load model safely
model = DietRecommendationModel(input_dim=11, output_dim=4)
try:
    model.load_state_dict(torch.load("diet_recommendation_model.pth", map_location=torch.device("cpu")))
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.warning(f"⚠️ Model not found or mismatched. Using random weights. ({e})")
model.eval()

# 3️⃣ Dummy scaler (so app never crashes)
dummy_scaler = StandardScaler()
dummy_scaler.mean_ = np.zeros(6)
dummy_scaler.scale_ = np.ones(6)
dummy_scaler.n_features_in_ = 6
dummy_scaler.feature_names_in_ = np.array(['Age','Height_cm','Weight_kg','BMI','Stress_Level','Sleep_Hours'])

# 4️⃣ Prediction helper
def predict_diet(input_features):
    with torch.no_grad():
        inputs = torch.tensor(input_features, dtype=torch.float32)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 5️⃣ Smart diet recommendation text
def get_diet_advice(diet_label, bmi, stress, sleep, smoker, exercise):
    if diet_label == "Excellent":
        return "You're maintaining an excellent diet! Keep your balance of proteins, fruits, and whole grains."
    elif diet_label == "Good":
        tips = []
        if bmi > 27: tips.append("reduce sugary drinks and processed snacks")
        if sleep < 6: tips.append("improve your sleep schedule for better recovery")
        if stress > 7: tips.append("add meditation or yoga to reduce stress")
        if not tips:
            return "A good diet overall — just ensure consistent hydration and moderate exercise."
        return "Your diet is good, but try to " + ", ".join(tips) + "."
    elif diet_label == "Average":
        advice = [
            "increase intake of fresh fruits and vegetables",
            "avoid skipping breakfast",
            "focus on home-cooked meals instead of fast food",
            "reduce salt and oily foods",
        ]
        return "Your diet is average. You could " + ", ".join(np.random.choice(advice, 2, replace=False)) + "."
    else:  # Poor
        poor_tips = [
            "avoid junk food and sugary drinks immediately",
            "increase daily water intake to at least 2.5L",
            "include green leafy vegetables and pulses in meals",
            "try a consistent meal schedule — 3 meals and 2 small snacks daily",
            "add light exercise like brisk walking or cycling 30 mins/day",
        ]
        return "Your diet needs improvement. You should " + ", ".join(np.random.choice(poor_tips, 3, replace=False)) + "."

# 6️⃣ Streamlit UI
st.title("🍎 AI-Powered Diet Recommendation System")
st.caption("Analyze your lifestyle and get personalized diet advice")

age = st.number_input("Age", 18, 100, 30)
height = st.number_input("Height (cm)", 120, 220, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
bmi = weight / ((height/100)**2)
smoker = st.selectbox("Smoker", ["Yes", "No"])
exercise = st.selectbox("Exercise Frequency", ["1-2 times/week", "3-5 times/week", "Daily"])
alcohol = st.selectbox("Alcohol Consumption", ["Low", "Moderate", "High"])
chronic = st.selectbox("Chronic Disease", ["Yes", "No"])
stress = st.number_input("Stress Level (1-10)", 1, 10, 5)
sleep = st.number_input("Sleep Hours", 1, 12, 7)

num_features = pd.DataFrame({
    "Age":[age],
    "Height_cm":[height],
    "Weight_kg":[weight],
    "BMI":[bmi],
    "Stress_Level":[stress],
    "Sleep_Hours":[sleep]
})

num_scaled = dummy_scaler.transform(num_features)

cat_features = pd.DataFrame({
    "Gender": [0],
    "Smoker": [1 if smoker=="Yes" else 0],
    "Exercise_Freq": [1 if exercise in ["3-5 times/week", "Daily"] else 0],
    "Alcohol_Consumption": [1 if alcohol in ["Moderate", "High"] else 0],
    "Chronic_Disease": [1 if chronic=="Yes" else 0]
})

final_input = pd.concat(
    [pd.DataFrame(num_scaled, columns=num_features.columns), cat_features], axis=1
).values

if st.button("Predict Diet Quality"):
    pred_idx = predict_diet(final_input)
    diet_map = {0: "Poor", 1: "Good", 2: "Excellent", 3: "Average"}
    diet_label = diet_map.get(pred_idx, "Unknown")
    st.subheader(f"🍽️ Recommended Diet Quality: {diet_label}")
    st.info(get_diet_advice(diet_label, bmi, stress, sleep, smoker, exercise))
