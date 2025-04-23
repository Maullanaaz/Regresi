import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Automobile.csv")

st.title("Analisis Mobil dan Regresi MPG")

# --- FILTER NON-NUMERIK ---
origin_options = st.multiselect("Pilih asal mobil (origin):", df["origin"].unique())
year_options = st.multiselect("Pilih tahun model:", sorted(df["model_year"].unique()))

filtered_df = df.copy()

if origin_options:
    filtered_df = filtered_df[filtered_df["origin"].isin(origin_options)]

if year_options:
    filtered_df = filtered_df[filtered_df["model_year"].isin(year_options)]

st.subheader("Data Terfilter")
st.dataframe(filtered_df)

# --- REGRESI LINIER BERGANDA ---
st.subheader("Regresi Linear: Prediksi MPG berdasarkan Beberapa Fitur")

# Fitur yang digunakan
features = ["horsepower", "weight", "displacement", "acceleration"]

# Drop nilai yang hilang
reg_df = filtered_df.dropna(subset=features + ["mpg"])

if not reg_df.empty:
    X = reg_df[features]
    y = reg_df["mpg"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Tampilkan koefisien tiap fitur
    st.write("**Koefisien Regresi (pengaruh masing-masing fitur):**")
    coef_df = pd.DataFrame({
        "Fitur": features,
        "Koefisien": model.coef_
    })
    st.dataframe(coef_df)

    st.write(f"**Intercept:** {model.intercept_:.2f}")
    st.write(f"**R² Score (akurasi model):** {model.score(X, y):.2f}")

    # Visualisasi: MPG Aktual vs Prediksi
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
    ax.set_xlabel("MPG Aktual")
    ax.set_ylabel("MPG Prediksi")
    ax.set_title("Aktual vs Prediksi MPG")
    st.pyplot(fig)
else:
    st.warning("Data tidak cukup untuk menjalankan regresi.")
