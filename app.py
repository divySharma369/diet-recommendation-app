
import streamlit as st
import torch
import pandas as pd
import torch.nn as nn
import joblib  # ✅ For loading the saved scaler

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

# 2️⃣ Load trained model and scaler
model = DietRecommendationModel(input_dim=11, output_dim=4)
model.load_state_dict(torch.load("diet_recommendation_model.pth", map_location=torch.device('cpu')))
model.eval()

# ✅ Load the saved scaler
scaler = joblib.load("scaler.pkl")

# 3️⃣ Prediction function
def predict_diet(input_features):
    with torch.no_grad():
        inputs = torch.tensor(input_features, dtype=torch.float32)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 4️⃣ Streamlit UI
st.title("🍎 Diet Recommendation System")

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

input_df = pd.DataFrame({
    "Age":[age], "Height_cm":[height], "Weight_kg":[weight], "BMI":[bmi],
    "Stress_Level":[stress], "Sleep_Hours":[sleep],
    "Smoker_Yes":[1 if smoker=="Yes" else 0],
    "Exercise_Freq_3-5 times/week":[1 if exercise=="3-5 times/week" else 0],
    "Exercise_Freq_Daily":[1 if exercise=="Daily" else 0],
    "Alcohol_Consumption_Moderate":[1 if alcohol=="Moderate" else 0],
    "Alcohol_Consumption_High":[1 if alcohol=="High" else 0],
    "Chronic_Disease":[1 if chronic=="Yes" else 0]
})

# ✅ Apply same scaler used during training
input_scaled = scaler.transform(input_df)

if st.button("Predict Diet Quality"):
    pred_idx = predict_diet(input_scaled)
    diet_map = {0: "Poor", 1: "Good", 2: "Excellent", 3: "Average"}
    st.success(f"Recommended Diet Quality: {diet_map[pred_idx]}")

