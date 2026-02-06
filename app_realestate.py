import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


st.set_page_config(page_title="Gayrimenkul Talep Tahmini", layout="wide")

st.title("🏠 Gayrimenkul Talep Tahmini Dashboard (TÜİK + Google Trends)")
st.write("Bu dashboard, TÜİK konut satış verileri ve Google Trends sinyalleriyle talep tahmini yapar.")

# -------------------------
# Kullanıcı Ayarları
# -------------------------
st.sidebar.header("⚙️ Ayarlar")

file_path = st.sidebar.text_input("Veri dosyası (Excel)", value="merged_tuik_trends.xlsx")

target = st.sidebar.selectbox(
    "Tahmin edilecek değişken",
    ["Toplam", "mortgage_sales", "first_hand", "second_hand"],
    index=0
)

trend_cols = st.sidebar.multiselect(
    "Google Trends değişkenleri",
    ["satılık daire", "kiralık daire", "konut kredisi", "ev fiyatları"],
    default=["satılık daire", "kiralık daire", "konut kredisi", "ev fiyatları"]
)

lags = st.sidebar.multiselect(
    "Lag değerleri",
    [1, 2, 3, 6, 12],
    default=[1, 2, 3, 6, 12]
)

split_ratio = st.sidebar.slider("Train oranı", 0.6, 0.95, 0.8, 0.05)

n_estimators = st.sidebar.slider("n_estimators", 200, 1200, 600, 100)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 10, 2, 1)

run_btn = st.sidebar.button("🚀 Modeli Çalıştır")


# -------------------------
# Fonksiyonlar
# -------------------------
def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(" ", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_lags(df, cols, lags):
    for c in cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
    return df


# -------------------------
# Model Çalıştır
# -------------------------
if run_btn:
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Excel okunamadı: {e}")
        st.stop()

    if "Date" not in df.columns:
        st.error("Excel dosyasında 'Date' sütunu yok.")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required_cols = [target] + trend_cols
    df = clean_numeric(df, required_cols)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Dosyada eksik sütunlar var: {missing}")
        st.stop()

    # Lag üret
    df_lagged = df.copy()
    df_lagged = make_lags(df_lagged, cols=[target] + trend_cols, lags=lags)
    df_lagged = df_lagged.dropna().reset_index(drop=True)

    feature_cols = [c for c in df_lagged.columns if "_lag" in c]

    # Train-test split
    split = int(len(df_lagged) * split_ratio)
    train = df_lagged.iloc[:split]
    test = df_lagged.iloc[split:]

    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    # Model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds_test)

    # Test tahmin tablosu
    pred_df = test[["Date", target]].copy()
    pred_df["prediction"] = preds_test
    pred_df["error"] = pred_df["prediction"] - pred_df[target]

    # Gelecek ay tahmini (modeli tüm veriyle eğit)
    model.fit(df_lagged[feature_cols], df_lagged[target])
    last_date = df_lagged["Date"].max()
    next_date = last_date + pd.offsets.MonthBegin(1)

    X_next = df_lagged.iloc[-1][feature_cols].values.reshape(1, -1)
    forecast_next = float(model.predict(X_next)[0])

    # -------------------------
    # Sonuçları Göster
    # -------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Gözlem Sayısı", f"{len(df_lagged):,}")
    col2.metric("Test MAE", f"{mae:,.0f}")
    col3.metric("Gelecek Ay Tahmini", f"{forecast_next:,.0f}")

    st.write(f"**Son gözlem ayı:** {last_date.date()}  \n**Tahmin edilen ay:** {next_date.date()}")

    # -------------------------
    # Grafik
    # -------------------------
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df_lagged["Date"], df_lagged[target], label="Gerçek", linewidth=2)

    # Test tahmini kesikli çizgi
    ax.plot(pred_df["Date"], pred_df["prediction"], linestyle="--", label="Tahmin (Test)", linewidth=2)

    # Gelecek ay tahmini kesikli çizgi
    last_real = float(df_lagged[target].iloc[-1])
    ax.plot([last_date, next_date], [last_real, forecast_next], linestyle="--", linewidth=2, label="Gelecek Ay Tahmini")
    ax.scatter(next_date, forecast_next, s=80, label="Forecast Noktası")

    ax.set_title(f"{target} Tahmini (Gerçek vs Tahmin)")
    ax.set_xlabel("Tarih")
    ax.set_ylabel(target)
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # -------------------------
    # Tahmin Tablosu
    # -------------------------
    st.subheader("📌 Test Dönemi Tahminleri")
    st.dataframe(pred_df, use_container_width=True)

    # Excel çıktısı
    out_file = "predictions_dashboard.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        pred_df.to_excel(writer, sheet_name="test_predictions", index=False)
        pd.DataFrame({"Date": [next_date], "forecast_next": [forecast_next]}).to_excel(
            writer, sheet_name="next_month_forecast", index=False
        )

    with open(out_file, "rb") as f:
        st.download_button(
            "📥 Tahminleri Excel olarak indir",
            data=f,
            file_name=out_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Sol menüden ayarları yapıp **Modeli Çalıştır** butonuna bas.")

