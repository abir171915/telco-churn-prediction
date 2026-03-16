# 📡 Telco Customer Churn Prediction

End-to-end machine learning project predicting customer churn
for a telecom company using Random Forest with MLOps practices.

## 🔍 Problem
Predicting which customers are likely to cancel their subscription
so the business can intervene with retention offers.

## 📊 Dataset
- 7,043 customers · 21 features
- Source: IBM Telco Customer Churn dataset

## 🛠️ Tech Stack
- Python, Pandas, Scikit-learn
- FastAPI — model serving
- Streamlit — interactive UI
- MLflow — experiment tracking
- Docker — containerisation

## 📈 Results
| Model                  | Accuracy | F1-Churn |
|------------------------|----------|----------|
| Logistic Regression    |   76%    |   0.55   |
| Random Forest (default)|   78%    |   0.58   |
| Random Forest (tuned)  |   75%    |   0.62   |
| XGBoost (tuned)        |   73%    |   0.61   |

## 🚀 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload

# Run UI
streamlit run app/streamlit_app.py
```

## 📁 Project Structure
```
├── data/
├── notebooks/
├── model/
├── app/
│   ├── main.py           # FastAPI
│   └── streamlit_app.py  # Streamlit UI
└── requirements.txt
```