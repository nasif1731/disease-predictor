import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Disease Category Predictor", layout="centered")

@st.cache_data
def load_data():
    from imblearn.over_sampling import SMOTE

    # Load raw data
    df = pd.read_csv("/content/drive/My Drive/disease_features.csv")
    df['Combined'] = df['Risk Factors'].fillna('') + ' ' + df['Symptoms'].fillna('') + ' ' + df['Signs'].fillna('')

    # TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['Combined'].astype(str))

    # Original labels
    categories = np.array([
        "Cardiovascular", "Endocrine", "Neurological", "Cardiovascular", "Respiratory",
        "Cardiovascular", "Cardiovascular", "Respiratory", "Endocrine", "Neurological",
        "Gastrointestinal", "Gastrointestinal", "Cardiovascular", "Cardiovascular",
        "Cardiovascular", "Neurological", "Neurological", "Gastrointestinal", "Endocrine",
        "Infectious", "Cardiovascular", "Neurological", "Endocrine", "Infectious", "Gastrointestinal"
    ])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(categories)

    # Apply SMOTE to balance categories
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_bal, y_bal = smote.fit_resample(X, y)

    return df, X_bal, y_bal, tfidf, label_encoder, categories

df, X, y, tfidf, label_encoder, categories = load_data()

st.title("🧠 Disease Category Prediction")
st.markdown("Enter patient **risk factors, symptoms, and signs** below to predict a possible disease category.")

user_input = st.text_area("🩺 Input symptoms (e.g., chest pain, fever, headache):")

k = st.selectbox("🔢 Select value of k", [3, 5, 7])
metric = st.selectbox("📏 Choose Distance Metric", ["euclidean", "manhattan", "cosine"])

if st.button("🔍 Predict Disease Category"):
    if not user_input.strip():
        st.warning("Please enter at least one symptom or sign.")
    else:
        vec = tfidf.transform([user_input])
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X, y)
        pred = knn.predict(vec)
        predicted_label = label_encoder.inverse_transform(pred)[0]
        st.success(f"✅ **Predicted Disease Category:** {predicted_label}")

        distances, indices = knn.kneighbors(vec)
        st.subheader("🔎 Top Similar Cases from Dataset:")
        for i in indices[0]:
          if i < len(df):
            st.markdown(
                f"• **Category:** {categories[y[i]]}<br>"
                f"• **Symptoms:** {df.iloc[i]['Symptoms']}<br>"
                f"• **Signs:** {df.iloc[i]['Signs']}<br>"
                f"• **Risk Factors:** {df.iloc[i]['Risk Factors']}",
                unsafe_allow_html=True
            )
          else:
            st.markdown(
                f"• **Category:** {label_encoder.inverse_transform([y[i]])[0]}<br>"
                f"• *Synthetic case – no real patient data available*",
                unsafe_allow_html=True
            )