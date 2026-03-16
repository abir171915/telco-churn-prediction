import streamlit as st
import requests

# ── Page config ──
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Header ──
st.title("📡 Telco Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they will churn.")
st.divider()

# ── Input Form ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")
    senior_citizen   = st.selectbox("Senior Citizen",      ["No", "Yes"])
    partner          = st.selectbox("Has Partner",         ["No", "Yes"])
    dependents       = st.selectbox("Has Dependents",      ["No", "Yes"])
    tenure           = st.slider("Tenure (months)",        0, 72, 12)
    paperless        = st.selectbox("Paperless Billing",   ["No", "Yes"])

with col2:
    st.subheader("💳 Service Info")
    internet         = st.selectbox("Internet Service",    ["DSL", "Fiber optic", "No"])
    contract         = st.selectbox("Contract Type",       ["Month-to-month", "One year", "Two year"])
    payment          = st.selectbox("Payment Method",      ["Electronic check", "Mailed check",
                                                            "Bank transfer (automatic)",
                                                            "Credit card (automatic)"])
    monthly_charges  = st.slider("Monthly Charges ($)",   18.0, 120.0, 65.0)
    total_charges    = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

st.divider()

col3, col4, col5 = st.columns(3)
with col3:
    online_security  = st.selectbox("Online Security",    ["No", "Yes"])
with col4:
    online_backup    = st.selectbox("Online Backup",      ["No", "Yes"])
with col5:
    device_protect   = st.selectbox("Device Protection",  ["No", "Yes"])

col6, col7 = st.columns(2)
with col6:
    tech_support     = st.selectbox("Tech Support",       ["No", "Yes"])

st.divider()

# ── Helper — Yes/No to 0/1 ──
def encode(val): return 1 if val == "Yes" else 0

# ── Predict button ──
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    # Build payload matching your API
    payload = {
        "SeniorCitizen":                        encode(senior_citizen),
        "Partner":                              encode(partner),
        "Dependents":                           encode(dependents),
        "tenure":                               tenure,
        "OnlineSecurity":                       encode(online_security),
        "OnlineBackup":                         encode(online_backup),
        "DeviceProtection":                     encode(device_protect),
        "TechSupport":                          encode(tech_support),
        "PaperlessBilling":                     encode(paperless),
        "MonthlyCharges":                       monthly_charges,
        "TotalCharges":                         total_charges,
        "InternetService_DSL":                  1 if internet == "DSL" else 0,
        "InternetService_Fiber_optic":          1 if internet == "Fiber optic" else 0,
        "InternetService_No":                   1 if internet == "No" else 0,
        "Contract_Month_to_month":              1 if contract == "Month-to-month" else 0,
        "Contract_One_year":                    1 if contract == "One year" else 0,
        "Contract_Two_year":                    1 if contract == "Two year" else 0,
        "PaymentMethod_Bank_transfer":          1 if payment == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit_card":            1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic_check":       1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed_check":           1 if payment == "Mailed check" else 0,
    }

    # ── Call FastAPI ──
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result   = response.json()

        # ── Display Result ──
        st.divider()

        if result["prediction"] == "WILL CHURN":
            st.error("🔴 This customer is likely to CHURN")
        else:
            st.success("🟢 This customer is likely to STAY")

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability",  result["churn_probability"])
        m2.metric("Stay Probability",   result["stay_probability"])
        m3.metric("Risk Level",         result["risk_level"].split("—")[0].strip())

        # Action box
        st.info(f"💡 Recommended Action: **{result['risk_level']}**")

        # Churn probability bar
        churn_pct = float(result["churn_probability"].replace("%", ""))
        st.subheader("Churn Probability")
        st.progress(int(churn_pct))

    except Exception as e:
        st.error(f"❌ Could not connect to API. Make sure FastAPI is running.\nError: {e}")