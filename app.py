import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Page config (UI only)
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model & scaler with proper file path handling
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")

# If model files don't exist, train them
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    df = pd.read_csv("Churn_Modelling.csv")
    
    # Drop unnecessary columns
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
    
    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)
    
    # Encode target (Exited = Churn)
    df["Exited"] = df["Exited"].astype(int)
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    le_dict = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Split data
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)


# ---------- HEADER ----------
st.markdown(
    """
    <h1 style="text-align:center;">📊 Customer Churn Prediction</h1>
    <p style="text-align:center;color:gray;">
    Predict whether a customer is likely to churn based on their profile
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "Fill in customer details to predict churn using a trained ML model."
    )
    st.markdown("---")
    st.caption("🔍 Model: Classification")
    st.caption("📈 Output: Churn Probability")

# ---------- INPUT SECTION ----------
st.subheader("🧾 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    CreditScore = st.number_input("Credit Score", 300, 850, 600)
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    
with col2:
    Age = st.slider("Age", 18, 100, 40)
    Tenure = st.slider("Tenure (Years)", 0, 10, 5)
    NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])

with col3:
    Balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    EstimatedSalary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
   

# ---------- ENCODING ----------
geography_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 1, "Female": 0}

data = {
    "CreditScore": CreditScore,
    "Geography": geography_map[Geography],
    "Gender": gender_map[Gender],
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary,
}

input_df = pd.DataFrame([data])

st.markdown("---")

# ---------- PREDICTION ----------
center_col = st.columns([1, 2, 1])[1]

with center_col:
    if st.button("🔮 Predict Churn", use_container_width=True):
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("### 📌 Prediction Result")

        if prediction == 1:
            st.error(
                f"❌ **Customer is likely to CHURN**  \n"
                f"**Probability:** {probability[1]:.2%}"
            )
        else:
            st.success(
                f"✅ **Customer is NOT likely to churn**  \n"
                f"**Probability:** {probability[0]:.2%}"
            )
