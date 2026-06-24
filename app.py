import streamlit as st
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ✅ PAGE CONFIG
st.set_page_config(
    page_title="Rental Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ✅ CUSTOM UI STYLE (Power BI look)
st.markdown("""
<style>
.kpi-card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
}
.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #00d4aa;
}
.kpi-label {
    font-size: 14px;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

# ✅ LOAD MODEL
try:
    model = joblib.load("model.pkl")
except:
    model = None

# ✅ LOAD DATA (DEPLOYMENT VERSION)
def load_data():
    try:
        df = pd.read_json("dags/processed_rentals.json")
        return df
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return pd.DataFrame()

df = load_data()

# ✅ HANDLE EMPTY DATA
if df.empty:
    st.warning("⚠️ No data available. Run your Airflow pipeline.")
    st.stop()

# ✅ HEADER
st.title("🏠 Rental Intelligence Dashboard")
st.markdown("### Analytics + AI Insights Platform")

# =======================
# ✅ FILTERS SIDEBAR
# =======================
st.sidebar.header("🔎 Filters")

city_filter = st.sidebar.multiselect(
    "City",
    df["city"].unique(),
    default=df["city"].unique()
)

bedroom_filter = st.sidebar.multiselect(
    "Bedrooms",
    sorted(df["bedrooms"].unique()),
    default=sorted(df["bedrooms"].unique())
)

price_range = st.sidebar.slider(
    "Price Range",
    int(df.price.min()),
    int(df.price.max()),
    (int(df.price.min()), int(df.price.max()))
)

# ✅ APPLY FILTER
filtered_df = df[
    (df["city"].isin(city_filter)) &
    (df["bedrooms"].isin(bedroom_filter)) &
    (df["price"].between(price_range[0], price_range[1]))
]

# =======================
# ✅ KPI CARDS
# =======================
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

kpis = [
    ("Total Listings", len(filtered_df)),
    ("Avg Price", f"${filtered_df['price'].mean():.0f}"),
    ("Price/Room", f"${filtered_df['price_per_room'].mean():.0f}"),
    ("Max Price", f"${filtered_df['price'].max():.0f}")
]

for col, (label, value) in zip([col1, col2, col3, col4], kpis):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# =======================
# ✅ CHARTS
# =======================
st.markdown("## 📈 Analytics")

c1, c2 = st.columns(2)

# Histogram
with c1:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["price"], bins=10, color="#00d4aa")
    st.pyplot(fig)

# Bedrooms
with c2:
    st.subheader("Bedrooms Distribution")
    fig2, ax2 = plt.subplots()
    filtered_df["bedrooms"].value_counts().plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

# =======================
# ✅ CITY ANALYSIS
# =======================
st.markdown("## 🌍 Listings by City")
st.bar_chart(filtered_df["city"].value_counts())

# =======================
# ✅ SCATTER
# =======================
st.markdown("## 💰 Price Efficiency")

fig3, ax3 = plt.subplots()
ax3.scatter(filtered_df["bedrooms"], filtered_df["price_per_room"], color="red")
ax3.set_xlabel("Bedrooms")
ax3.set_ylabel("Price per Room")
st.pyplot(fig3)

# =======================
# ✅ ML PREDICTION (FIXED ✅)
# =======================
st.markdown("## 🤖 AI Price Prediction")

if model is not None:

    col1, col2 = st.columns(2)

    with col1:
        bed_input = st.slider("Bedrooms", 1, 5, 2)

    with col2:
        room_input = st.slider("Price per Room", 500, 2000, 800)

    prediction = model.predict([[bed_input, room_input]])

    # ✅ FIXED LINE (IMPORTANT)
    st.success(f"💰 Estimated Rent: ${prediction[0]:.0f}")

else:
    st.info("Run model.py to enable predictions")

# =======================
# ✅ TABLE
# =======================
st.markdown("## 📄 Data Table")
st.dataframe(filtered_df)

# =======================
# ✅ EXPORT DATA
# =======================
st.markdown("## 📤 Export Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download as CSV",
    data=csv,
    file_name="rental_data.csv",
    mime="text/csv"
)
