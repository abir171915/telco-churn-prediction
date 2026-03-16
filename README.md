# 📡 Telco Customer Churn Prediction

> An end-to-end machine learning project that identifies customers likely to cancel their subscription — enabling targeted retention offers before they leave.

---

## 🧩 The Problem

Telecom companies lose millions every year to customer churn. The challenge is not just knowing **how many** customers leave — it's knowing **who** is about to leave **before** they do, so the business can intervene in time.

Most companies only find out a customer has churned after they've already left. By then it's too late.

This project solves that.

---

## 💡 The Solution

A machine learning system that:
1. Analyses a customer's profile and usage patterns
2. Predicts their churn probability in real time
3. Assigns a risk tier (High / Medium / Low)
4. Recommends a specific retention action for each tier

---

## 💰 Business Impact

| Metric | Value |
|--------|-------|
| Dataset size | 7,043 customers |
| Churn rate | 26.5% (~1,869/month) |
| Model recall | 76% (catches 1,421 churners) |
| Avg revenue per customer | $64.76/month |
| Estimated revenue saved | ~$441,000/year |
| Cost of retention offers | ~$170,000/year |
| **Net value of model** | **~$271,000/year** ✅ |

### How It Works in Production

```
Every month
     ↓
Feed all customers into model
     ↓
Model scores each customer
     ↓
High risk   (>70%)   →  Personal call + 30% discount offer
Medium risk (40-70%) →  Automated email with loyalty reward
Low risk    (<40%)   →  No action — save the budget
```

---

## 📊 Dataset

- **Source** — IBM Telco Customer Churn Dataset
- **Size** — 7,043 customers · 21 features
- **Target** — Churn (Yes / No)

### Key Features

| Feature | Type | Importance |
|---------|------|------------|
| tenure | Numerical | 🔴 Very High |
| Contract | Categorical | 🔴 Very High |
| MonthlyCharges | Numerical | 🟡 High |
| InternetService | Categorical | 🟡 High |
| PaymentMethod | Categorical | 🟡 High |
| OnlineSecurity | Categorical | 🟢 Medium |
| TechSupport | Categorical | 🟢 Medium |
| SeniorCitizen | Binary | 🟢 Medium |

---

## 🔍 Key EDA Findings

- **Contract type is the #1 predictor** — month-to-month customers churn at 42.7% vs 2.8% for 2-year contracts
- **First year is critical** — 47.7% of customers in their first 12 months churn
- **Electronic check = red flag** — 45.3% churn rate vs ~16% for auto-pay methods
- **Fiber optic customers churn most** — 41.9% vs 19% for DSL
- **Churned customers pay more** — avg $74.44/month vs $61.27 for retained customers

---

## 🤖 Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 76% | 0.55 | 0.55 | 0.55 |
| Random Forest (default) | 78% | 0.62 | 0.54 | 0.58 |
| XGBoost (tuned) | 73% | 0.49 | 0.80 | 0.61 |
| **Random Forest (tuned)** | **75%** | **0.53** | **0.76** | **0.62** ✅ |

### Why Random Forest (tuned) wins

For churn prediction **recall matters more than accuracy**. Missing a churner costs the business $64/month permanently. A false alarm costs ~$10 for an unnecessary offer.

```
Recall 0.76 → catches 76 out of every 100 real churners
             missing only 24 — acceptable business risk
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| API | FastAPI |
| UI | Streamlit |
| Experiment tracking | MLflow |
| Containerisation | Docker |
| Language | Python 3.9+ |

---

## 📁 Project Structure

```
telco-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── churn_eda_and_training.ipynb    ← full EDA + model training
│
├── model/
│   ├── churn_model.pkl                 ← trained Random Forest
│   └── scaler.pkl                      ← fitted StandardScaler
│
├── app/
│   ├── main.py                         ← FastAPI backend
│   └── streamlit_app.py                ← Streamlit frontend
│
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (or use saved pkl files)
```bash
jupyter notebook notebooks/churn_eda_and_training.ipynb
```

### 4. Run the API
```bash
uvicorn app.main:app --reload
```
API runs at → `http://127.0.0.1:8000`
API docs at → `http://127.0.0.1:8000/docs`

### 5. Run the UI (new terminal)
```bash
streamlit run app/streamlit_app.py
```
UI runs at → `http://localhost:8501`

---

## 🔗 API Reference

### `GET /`
Health check
```json
{
  "message": "Churn Prediction API is running",
  "status": "healthy"
}
```

### `POST /predict`
Predict churn for a customer

**Request body:**
```json
{
  "tenure": 2,
  "MonthlyCharges": 85.0,
  "TotalCharges": 170.0,
  "Contract_Month_to_month": 1,
  "InternetService_Fiber_optic": 1,
  "PaymentMethod_Electronic_check": 1,
  ...
}
```

**Response:**
```json
{
  "prediction": "WILL CHURN",
  "churn_probability": "87.0%",
  "stay_probability": "13.0%",
  "risk_level": "HIGH — call immediately with retention offer"
}
```

---

## 📈 ML Pipeline

```
Raw CSV Data
     ↓
Data Cleaning          remove blanks, fix TotalCharges dtype
     ↓
Feature Engineering    label encode, one-hot encode, drop weak features
     ↓
Train/Test Split       80/20, stratified on target
     ↓
Feature Scaling        StandardScaler on numerical columns
     ↓
Model Training         Logistic Regression → Random Forest → XGBoost
     ↓
Hyperparameter Tuning  GridSearchCV, 5-fold CV, optimising F1
     ↓
Model Evaluation       accuracy, precision, recall, F1, confusion matrix
     ↓
Model Saved            pickle → churn_model.pkl + scaler.pkl
     ↓
FastAPI                model served as REST API
     ↓
Streamlit              interactive UI consuming the API
```

---

## 🧠 What I Learned

- Why **recall matters more than accuracy** for imbalanced classification problems
- How **class imbalance** affects model predictions and how to handle it with `class_weight='balanced'`
- The difference between **label encoding vs one-hot encoding** and when to use each
- Why you **fit the scaler on training data only** to prevent data leakage
- How to wrap an ML model into a **production REST API** with FastAPI
- How to build an **interactive ML app** with Streamlit
- End-to-end **MLOps practices** — from notebook to deployed application

---

## 👤 Author

**Md Abir Hossain**
100 Days of Machine Learning — Day 6 Project

---

## 📄 License

MIT License — feel free to use and adapt this project.