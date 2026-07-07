import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import feature_engineering 

#Page Configurations
st.set_page_config(page_title = "Customer Churn Prediction Website",
                   page_icon = "https://thumbs.dreamstime.com/b/group-user-icons-some-individuals-leaving-highlighting-customer-churn-several-stand-together-two-others-walk-away-396264448.jpg?w=768",
                   menu_items = {"Get help": "mailto:oguzhanaydin16@ku.edu.tr",
                                 "About": "For more information\n" + "https://github.com/oguzhanydn"})

#Page Editing
st.title("Customer Churn Prediction Model Project")

st.markdown("A telecommunication company wants to decide whether a customer will stop remaining a customer or not by looking at the various features of the customer.")

st.image("https://thumbs.dreamstime.com/b/customer-churn-shown-using-text-customer-churn-shown-using-text-312935596.jpg?w=992")

st.markdown("In today's competitive telecom industry, **retaining existing customers** is more critical than ever.")
st.markdown("Our client wants to identify customers who are likely to churn before they leave, and take proactive action to retain them.")
st.markdown("*Let's predict who's at risk!*")

#Data Preview
st.header("Data Dictionary")

st.subheader("Data Attributes")

st.markdown("- **gender**: Whether the customer is a male or a female")
st.markdown("- **SeniorCitizen**: Whether the customer is a senior citizen or not (1, 0)")
st.markdown("- **Partner**: Whether the customer has a partner or not (Yes, No)")
st.markdown("- **Dependents**: Whether the customer has dependents or not (Yes, No)")
st.markdown("- **tenure**: Number of months the customer has stayed with the company")
st.markdown("- **PhoneService**: Whether the customer has a phone service or not (Yes, No)")
st.markdown("- **MultipleLines**: Whether the customer has multiple lines or not (Yes, No)")
st.markdown("- **InternetService**: Customer’s internet service provider (DSL, Fiber optic, No)")
st.markdown("- **OnlineSecurity**: Whether the customer has online security or not (Yes, No)")
st.markdown("- **OnlineBackup**: Whether the customer has online backup or not (Yes, No)")
st.markdown("- **DeviceProtection**: Whether the customer has device protection or not (Yes, No)")
st.markdown("- **TechSupport**: Whether the customer has tech support or not (Yes, No)")
st.markdown("- **StreamingTV**: Whether the customer has streaming TV or not (Yes, No)")
st.markdown("- **StreamingMovies**: Whether the customer has streaming movies or not (Yes, No)")
st.markdown("- **Contract**: The contract term of the customer (Month-to-month, One year, Two year)")
st.markdown("- **PaperlessBilling**: Whether the customer has paperless billing or not (Yes, No)")
st.markdown("- **PaymentMethod**: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))")
st.markdown("- **MonthlyCharges**: The amount charged to the customer monthly")
st.markdown("- **TotalCharges**: The total amount charged to the customer")
st.markdown("- **Churn**: Whether the customer churned or not (Yes or No)")

#Reading the dataframe with pandas
df = joblib.load("X_train.pkl")

#Adding a sample table
st.dataframe(df.sample(5), width="content")

#Sidebar Part
st.sidebar.markdown("**Choose** the features below to see the result!")

#Getting Feature Inputs
gender = st.sidebar.selectbox("gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("SeniorCitizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.number_input("tenure", min_value=0, max_value=100, format="%d")
PhoneService = st.sidebar.selectbox("PhoneService", ["No", "Yes"])
if PhoneService == "No":
    MultipleLines = st.sidebar.selectbox("MultipleLines",["No"],disabled=True)
else:
    MultipleLines = st.sidebar.selectbox("MultipleLines",["No", "Yes"])
InternetService = st.sidebar.selectbox("InternetService",["No", "DSL", "Fiber optic"])
if InternetService == "No":
    OnlineSecurity = st.sidebar.selectbox("OnlineSecurity",["No"],disabled=True)
    OnlineBackup = st.sidebar.selectbox("OnlineBackup",["No"],disabled=True)
    DeviceProtection = st.sidebar.selectbox("DeviceProtection",["No"],disabled=True)
    TechSupport = st.sidebar.selectbox("TechSupport",["No"],disabled=True)
    StreamingTV = st.sidebar.selectbox("StreamingTV",["No"],disabled=True)
    StreamingMovies = st.sidebar.selectbox("StreamingMovies",["No"],disabled=True)
else:
    OnlineSecurity = st.sidebar.selectbox("OnlineSecurity",["No", "Yes"])
    OnlineBackup = st.sidebar.selectbox("OnlineBackup",["No", "Yes"])
    DeviceProtection = st.sidebar.selectbox("DeviceProtection",["No", "Yes"])
    TechSupport = st.sidebar.selectbox("TechSupport",["No", "Yes"])
    StreamingTV = st.sidebar.selectbox("StreamingTV",["No", "Yes"])
    StreamingMovies = st.sidebar.selectbox("StreamingMovies",["No", "Yes"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("PaperlessBilling", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.sidebar.number_input("MonthlyCharges", min_value=10, max_value=130, format="%.2f")
TotalCharges = st.sidebar.number_input("TotalCharges", min_value=0, max_value=10000, format="%.2f")

#Getting the Trained Final Model
xgb_model = joblib.load("streamlit_predict_pipeline.pkl")

input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
})

#Results
st.header("Results")

if st.sidebar.button("Submit"):

    st.info("You can find your results below.")

    #Getting the Prediction
    pred = int(xgb_model.predict(input_df)[0])
    pred_probability = np.round(xgb_model.predict_proba(input_df), 2)

    #Result table
    results_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
        "Prediction": [pred],
        "Will Not Churn Probability": [pred_probability[0, 0]],
        "Will Churn Probability": [pred_probability[0, 1]]
    })
    results_df["Prediction"] = results_df["Prediction"].map({0: "Not Churn",1: "Churn"})
    
    st.dataframe(results_df, use_container_width=True)

    if pred == 0:
        st.image("https://thumbs.dreamstime.com/b/cartoon-illustration-happy-smiling-boss-manager-businessma-stick-man-businessman-clerk-politician-posing-thumbs-up-97511262.jpg?w=768")
    
    else:
        st.image("https://thumbs.dreamstime.com/b/vector-cartoon-stick-figure-drawing-conceptual-illustration-happy-smiling-man-employee-worker-customer-leaving-sad-depressed-187552886.jpg")
    
    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    
    st.subheader("Why did the model make this prediction?")

    X_shap = xgb_model[:-1].transform(input_df)

    model = xgb_model.named_steps["model"]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_shap)

    st.subheader("Prediction Explanation")
    st.markdown("The figure below shows how each feature affected the prediction. " \
    "Features pushing the prediction towards churn are shown in red, while features reducing churn risk are shown in blue.")

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)
    plt.close(fig)

    #Writing Explanation
    shap_df = pd.DataFrame({"feature": X_shap.columns,
                            "value": X_shap.iloc[0],
                            "shap": shap_values.values[0]})

    shap_df["abs_shap"] = shap_df["shap"].abs()

    top3 = shap_df.sort_values("abs_shap", ascending=False).head(3)

    def explain_feature(feature, value, original_df):

        original = original_df.iloc[0]

        if feature == "MonthlyCharges":
            return f"the monthly charge being **{original['MonthlyCharges']:.2f}**"

        elif feature == "tenure":
            return f"the customer staying for **{int(original['tenure'])} months**"

        elif feature == "TotalCharges":
            return f"the total charge being **{original['TotalCharges']:.2f}**"

        elif feature == "TotalServices":
            return f"the customer subscribing to **{int(value)} services**"

        elif feature == "SeniorCitizen":
            return ("the customer being a **senior citizen**"
                    if original["SeniorCitizen"] == 1
                    else
                    "the customer not being a **senior citizen**")

        elif feature == "gender":
            return ("the customer being **Male**"
                    if original["gender"] == "Male"
                    else
                    "the customer being **Female**")

        elif feature == "Partner":
            return ("the customer having a **partner**"
                    if original["Partner"] == "Yes"
                    else
                    "the customer not having a **partner**")

        elif feature == "Dependents":
            return ("the customer having **dependents**"
                    if original["Dependents"] == "Yes"
                    else
                    "the customer not having **dependents**")

        elif feature == "PhoneService":
            return ("the customer having **phone service**"
                    if original["PhoneService"] == "Yes"
                    else
                    "the customer not having **phone service**")

        elif feature == "MultipleLines":
            return ("the customer having **multiple phone lines**"
                    if original["MultipleLines"] == "Yes"
                    else
                    "the customer not having **multiple phone lines**")

        elif feature in ["OnlineSecurity",
                         "OnlineBackup",
                         "DeviceProtection",
                         "TechSupport",
                         "StreamingTV",
                         "StreamingMovies",
                         "PaperlessBilling"]:

            return f"{feature} being **{original[feature]}**"

        elif feature.startswith("Contract"):
            return f"contract type being **{original['Contract']}**"

        elif feature.startswith("InternetService"):
            return f"internet service being **{original['InternetService']}**"

        elif feature.startswith("PaymentMethod"):
            return f"payment method being **{original['PaymentMethod']}**"

        else:
            return f"{feature} being **{value}**"


    st.subheader("Prediction Explanation")
    st.markdown("The prediction was mainly driven by the following factors:")

    for i, (_, row) in enumerate(top3.iterrows()):

        explanation = explain_feature(row["feature"],
                                      row["value"],
                                      input_df)

        if row["shap"] > 0:

            if i == 0:
                st.write(f"🔴 The most important reason for this prediction is {explanation}, which increased the likelihood of churn.")
            else:
                st.write(f"🔴 Another important factor is {explanation}, which also increased the likelihood of churn.")

        else:

            if i == 0:
                st.write(f"🔵 The strongest factor reducing churn risk is {explanation}.")
            else:
                st.write(f"🔵 Another factor reducing churn risk is {explanation}.")

else:
    st.markdown("Please click the *Submit Button*!")
