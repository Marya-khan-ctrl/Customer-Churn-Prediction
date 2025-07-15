# Customer-Churn-Prediction
This app uses machine learning to predict if a customer will leave (churn) or stay, based on details like tenure, charges, and services. It uses a Random Forest model with a Streamlit frontend, helping businesses identify at-risk customers early and improve retention through proactive actions.


Objective
The objective of this project is to develop a machine learning model that predicts whether a telecom customer is likely to churn (leave the service) or stay. The goal is to help businesses proactively identify customers who are at risk of leaving so they can take steps to retain them.

Methodology
Dataset: The Telco Customer Churn dataset was used for training and evaluation.
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Data Preprocessing:

Dropped unnecessary columns (e.g., customer ID).

Handled missing values in the TotalCharges column.

Converted categorical variables using one-hot encoding.

Scaled numerical features using StandardScaler.

Feature Selection:

Input features include tenure, contract type, monthly charges, total charges, and various service-related fields.

Target variable is Churn, mapped as 1 (Yes) and 0 (No).

Model:

Used a Random Forest Classifier within a scikit-learn pipeline.

Hyperparameters were tuned using GridSearchCV for optimal accuracy.

Frontend:

A web interface was created using Streamlit.

Users can input customer details manually and receive a real-time churn prediction.

Model Deployment:

The trained model was saved using joblib for reuse and deployment.

Key Results
The model achieved an accuracy of approximately 79% on the test dataset.

It performs well in identifying non-churning customers and reasonably well in detecting churners.

The Streamlit interface makes the tool accessible to non-technical users.

This solution can assist telecom companies in reducing churn by allowing them to take early action for customer retention.
