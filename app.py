
import streamlit as st
import pandas as pd
import joblib
import datetime
from sklearn.metrics import classification_report, accuracy_score

# Load model dan data
model = joblib.load("model_sentiment.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder_sentiment.pkl")
dataset = pd.read_csv("data_cleaned.csv", on_bad_lines="skip")

st.set_page_config(page_title="Sentimen Mobil Listrik", layout="wide")

# Sidebar
st.sidebar.title("Sentimen App")
halaman = st.sidebar.radio("Navigasi", ["Klasifikasi", "Data Komentar", "Evaluasi Model"])

if halaman == "Klasifikasi":
    st.markdown("<h1 style='text-align: center; color: #F3B27C;'>Klasifikasi Sentimen Mobil Listrik</h1>", unsafe_allow_html=True)
    komentar = st.text_area("Komentar Anda:", placeholder="Masukkan komentar anda...")
    if st.button("Klasifikasi", use_container_width=True):
        if komentar:
            komentar_vector = tfidf.transform([komentar])
            pred = model.predict(komentar_vector)
            label = le.inverse_transform(pred)[0]
            st.success(f"Hasil klasifikasi: {label.capitalize()}")
            new_data = pd.DataFrame([{
                "id_komentar": "-",
                "nama_akun": "-",
                "tanggal": datetime.datetime.now(),
                "text_cleaning": komentar,
                "sentimen": label.lower()
            }])
            new_data.to_csv("data_cleaned.csv", mode="a", header=False, index=False)
            st.toast("Komentar berhasil ditambahkan ke dataset!", icon="✅")
        else:
            st.warning("Silakan masukkan komentar terlebih dahulu.")

elif halaman == "Data Komentar":
    st.markdown("## Statistik Sentimen Komentar")

    counts = dataset["sentimen"].value_counts()
    st.bar_chart(counts)

    sentimen_filter = st.selectbox("Filter Sentimen", ["Semua", "positif", "negatif", "netral"])
    if sentimen_filter != "Semua":
        data_filtered = dataset[dataset["sentimen"] == sentimen_filter]
    else:
        data_filtered = dataset

    st.markdown("### Data Komentar")
    st.dataframe(data_filtered.reset_index(drop=True))

    st.download_button("Unduh Dataset", data=dataset.to_csv(index=False), file_name="data_cleaned.csv")

elif halaman == "Evaluasi Model":
    st.markdown("## Evaluasi Model Sentimen")
    uploaded_file = st.file_uploader("Unggah file CSV untuk evaluasi", type=["csv"])
    if uploaded_file:
        df_eval = pd.read_csv(uploaded_file)
        if {"text_cleaning", "sentimen"}.issubset(df_eval.columns):
            df_eval = df_eval.dropna(subset=["text_cleaning", "sentimen"])
            df_eval["sentimen"] = df_eval["sentimen"].astype(str).str.strip().str.lower()
            valid_labels = set(le.classes_)
            df_eval = df_eval[df_eval["sentimen"].isin(valid_labels)]
            if df_eval.empty:
                st.error("Semua label tidak dikenali model.")
            else:
                X = tfidf.transform(df_eval["text_cleaning"].astype(str))
                y_true = le.transform(df_eval["sentimen"])
                y_pred = model.predict(X)
                acc = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, target_names=le.classes_)
                st.markdown(f"**Akurasi Model:** {acc:.4f}")
                st.text("Laporan Klasifikasi:")
                st.code(report, language="text")
        else:
            st.error("File harus memiliki kolom 'text_cleaning' dan 'sentimen'.")
