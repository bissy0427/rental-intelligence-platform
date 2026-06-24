import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# =============================
# ✅ PAGE CONFIG
# =============================
st.set_page_config(layout="wide")

# =============================
# ✅ CUSTOM STYLE (MODERN UI)
# =============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.kpi-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    transition: transform 0.3s ease;
}
.kpi-card:hover {
    transform: scale(1.05);
}
.kpi-value {
    font-size: 30px;
    color: #00ffcc;
}
</style>
""", unsafe_allow_html=True)

# =============================
# ✅ LOAD DATA
# =============================
def load_data():
    try:
        return pd.read_json("dags/processed_rentals.json")
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data available")
    st.stop()

# =============================
# ✅ LOAD MODEL
# =============================
try:
    model = joblib.load("model.pkl")
except:
    model = None

# =============================
# ✅ SIDEBAR FILTERS
# =============================
st.sidebar.header("🔎 Filters")

city = st.sidebar.multiselect("City", df.city.unique(), default=df.city.unique())
bed = st.sidebar.multiselect("Bedrooms", df.bedrooms.unique(), default=df.bedrooms.unique())

price_min, price_max = st.sidebar.slider(
    "Price Range",
    int(df.price.min()),
    int(df.price.max()),
    (int(df.price.min()), int(df.price.max()))
)

filtered_df = df[
    (df.city.isin(city)) &
    (df.bedrooms.isin(bed)) &
    (df.price.between(price_min, price_max))
]

# =============================
# ✅ TABS (SaaS UI)
# =============================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI", "📄 Data"])

# =============================
# ✅ DASHBOARD
# =============================
with tab1:

    st.title("🏠 Rental Intelligence")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{len(filtered_df)}</div>
        <div>Total Listings</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">${filtered_df.price.mean():.0f}</div>
        <div>Avg Price</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">${filtered_df.price.max():.0f}</div>
        <div>Max Price</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Price Distribution")

    fig, ax = plt.subplots()
    ax.hist(filtered_df.price, color="#00ffcc")
    st.pyplot(fig)

# =============================
# ✅ AI TAB
# =============================
with tab2:

    st.header("🤖 AI Prediction")

    if model:
        b = st.slider("Bedrooms", 1, 5, 2)
        p = st.slider("Price per Room", 500, 2000, 800)

        prediction = model.predict([[b, p]])
        st.success(f"Estimated Rent: ${prediction:.0f}")

# =============================
# ✅ DATA TAB
# =============================
with tab3:

    st.dataframe(filtered_df)

    st.download_button(
        "Download CSV",
        filtered_df.to_csv(index=False),
        "data.csv"
    )