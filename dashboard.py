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

# --- REGRESI LINIER SEDERHANA ---
st.subheader("Regresi Linear: Prediksi MPG berdasarkan Horsepower")

# Drop missing values
reg_df = filtered_df.dropna(subset=["horsepower", "mpg"])

if not reg_df.empty:
    X = reg_df[["horsepower"]]
    y = reg_df["mpg"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    st.write(f"Koefisien regresi: {model.coef_[0]:.2f}")
    st.write(f"Intercept: {model.intercept_:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Data Aktual", alpha=0.5)
    ax.plot(X, y_pred, color="red", label="Garis Regresi")
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("MPG")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Data tidak cukup untuk menjalankan regresi.")
