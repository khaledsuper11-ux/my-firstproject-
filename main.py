import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Prediction")
st.write("Predict whether a customer will subscribe to a term deposit.")

@st.cache_data
def load_data():
    return pd.read_csv("bank-additional-full.csv.xls", sep=";")

df = load_data()

X = df.drop("y", axis=1)
y = df["y"].map({"no": 0, "yes": 1})

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

st.sidebar.header("Customer Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

job = st.sidebar.selectbox(
    "Job",
    sorted(df["job"].dropna().unique())
)

marital = st.sidebar.selectbox(
    "Marital Status",
    sorted(df["marital"].dropna().unique())
)

education = st.sidebar.selectbox(
    "Education",
    sorted(df["education"].dropna().unique())
)

default = st.sidebar.selectbox(
    "Credit Default",
    sorted(df["default"].dropna().unique())
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    sorted(df["housing"].dropna().unique())
)

loan = st.sidebar.selectbox(
    "Personal Loan",
    sorted(df["loan"].dropna().unique())
)

contact = st.sidebar.selectbox(
    "Contact Type",
    sorted(df["contact"].dropna().unique())
)

month = st.sidebar.selectbox(
    "Last Contact Month",
    sorted(df["month"].dropna().unique())
)

day_of_week = st.sidebar.selectbox(
    "Last Contact Day",
    sorted(df["day_of_week"].dropna().unique())
)

duration = st.sidebar.number_input(
    "Call Duration",
    min_value=0,
    value=300
)

campaign = st.sidebar.number_input(
    "Number of Contacts",
    min_value=1,
    value=1
)

pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    min_value=0,
    value=999
)

previous = st.sidebar.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.sidebar.selectbox(
    "Previous Campaign Result",
    sorted(df["poutcome"].dropna().unique())
)

emp_var_rate = st.sidebar.number_input(
    "Employment Variation Rate",
    value=1.0
)

cons_price_idx = st.sidebar.number_input(
    "Consumer Price Index",
    value=93.0
)

cons_conf_idx = st.sidebar.number_input(
    "Consumer Confidence Index",
    value=-40.0
)

euribor3m = st.sidebar.number_input(
    "Euribor 3 Month Rate",
    value=4.0
)

nr_employed = st.sidebar.number_input(
    "Number of Employees",
    value=5000.0
)

customer = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "month": [month],
    "day_of_week": [day_of_week],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome],
    "emp.var.rate": [emp_var_rate],
    "cons.price.idx": [cons_price_idx],
    "cons.conf.idx": [cons_conf_idx],
    "euribor3m": [euribor3m],
    "nr.employed": [nr_employed]
})

st.subheader("Customer Data")
st.dataframe(customer, use_container_width=True)

if st.button("🔮 Predict", use_container_width=True):

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ The customer is likely to subscribe to a term deposit.")
    else:
        st.error("❌ The customer is unlikely to subscribe to a term deposit.")

    st.metric(
        "Subscription Probability",
        f"{probability * 100:.2f}%"
    )

st.divider()

st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of Customers", len(df))

with col2:
    st.metric("Number of Features", len(df.columns) - 1)

with col3:
    st.metric(
        "Positive Subscriptions",
        f"{(df['y'] == 'yes').sum():,}"
    )

st.subheader("🎯 Target Distribution")

target_counts = df["y"].value_counts()

st.bar_chart(target_counts)