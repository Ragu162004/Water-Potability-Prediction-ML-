import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

st.set_page_config(page_title="Water Potability Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df.dropna()

df = load_data()
X = df.drop("Potability", axis=1)
y = df["Potability"]
feature_names = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    trained_models[name] = model
    results.append((name, acc, y_pred))

results.sort(key=lambda x: x[1])

st.sidebar.header("Water Sample Input")
def user_input():
    return pd.DataFrame({
        "ph": [st.sidebar.slider("pH", 0.0, 14.0, 7.0)],
        "Hardness": [st.sidebar.slider("Hardness", 50.0, 300.0, 150.0)],
        "Solids": [st.sidebar.slider("Solids", 500.0, 50000.0, 10000.0)],
        "Chloramines": [st.sidebar.slider("Chloramines", 0.0, 15.0, 7.5)],
        "Sulfate": [st.sidebar.slider("Sulfate", 100.0, 500.0, 300.0)],
        "Conductivity": [st.sidebar.slider("Conductivity", 100.0, 1000.0, 500.0)],
        "Organic_carbon": [st.sidebar.slider("Organic Carbon", 0.0, 30.0, 15.0)],
        "Trihalomethanes": [st.sidebar.slider("Trihalomethanes", 0.0, 120.0, 60.0)],
        "Turbidity": [st.sidebar.slider("Turbidity", 0.0, 10.0, 5.0)],
    })

input_df = user_input()
scaled_input = scaler.transform(input_df)

st.title("Water Potability Prediction")
st.subheader("Input Sample Data")
st.write(input_df)

st.subheader("Model Predictions (from least to most accurate)")
cols = st.columns(3)
for i, (name, acc, _) in enumerate(results):
    model = trained_models[name]
    prediction = model.predict(scaled_input)[0]
    label = "Potable" if prediction == 1 else "Not Potable"
    with cols[i % 3]:
        st.success(f"**{name}: {label}**")

st.subheader("Model Evaluation Metrics (Least to Most Accurate)")
for name, acc, y_pred in results:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label=f"{name} Accuracy", value=f"{acc*100:.2f}%")
    with col2:
        st.markdown("**Confusion Matrix**")
        st.table(pd.DataFrame(confusion_matrix(y_test, y_pred),
                              index=["Actual 0", "Actual 1"],
                              columns=["Predicted 0", "Predicted 1"]))
        st.markdown("**Classification Report**")
        st.code(classification_report(y_test, y_pred), language='text')

st.subheader("Feature Importance (Tree-Based Models)")
if st.button("Show Feature Importance"):
    for name in ["Decision Tree", "Random Forest", "XGBoost"]:
        model = trained_models[name]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = feature_names[sorted_idx]
        sorted_importance = importances[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=sorted_importance, y=sorted_features, ax=ax)
        ax.set_title(f"{name} - Feature Importance")
        st.pyplot(fig)

if "show_tree" not in st.session_state:
    st.session_state.show_tree = False

if st.button("Toggle Decision Tree Diagram"):
    st.session_state.show_tree = not st.session_state.show_tree

if st.session_state.show_tree:
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_tree(trained_models["Decision Tree"], filled=True, feature_names=feature_names, class_names=["Not Potable", "Potable"], rounded=True)
    st.pyplot(fig)
