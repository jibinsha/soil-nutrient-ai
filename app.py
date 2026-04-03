import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import ee

# ===============================
# INIT GEE (MANDATORY 🔥)
# ===============================
import json
import streamlit as st

# ===============================
# GEE INIT (CLOUD SAFE 🔥)
# ===============================
service_account_info = json.loads(st.secrets["EARTHENGINE_TOKEN"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=json.dumps(service_account_info)
)

ee.Initialize(credentials)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Kerala Soil AI", layout="wide")

# ===============================
# DARK UI
# ===============================
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

.title {
    font-size: 34px;
    font-weight: bold;
    color: #2ecc71;
}

.subtitle {
    color: #bbbbbb;
}

.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 15px;
    border-left: 4px solid #2ecc71;
}

.metric-card h2 {
    color: #2ecc71;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODELS
# ===============================
model_N = joblib.load("model_N.pkl")
model_P = joblib.load("model_P.pkl")
model_K = joblib.load("model_K.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# HEADER
# ===============================
st.markdown('<div class="title">🌱 Kerala Soil Nutrient AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI + Satellite Based Soil Prediction</div>', unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Controls")
st.sidebar.info("Click on map to predict soil nutrients")

# ===============================
# KERALA BOUNDS
# ===============================
KERALA_BOUNDS = {
    "lat_min": 8.2,
    "lat_max": 12.8,
    "lon_min": 74.8,
    "lon_max": 77.5
}

# ===============================
# MAP
# ===============================
m = folium.Map(location=[10.5, 76.2], zoom_start=7)

folium.Rectangle(
    bounds=[
        [KERALA_BOUNDS["lat_min"], KERALA_BOUNDS["lon_min"]],
        [KERALA_BOUNDS["lat_max"], KERALA_BOUNDS["lon_max"]]
    ],
    color="green",
    fill=True,
    fill_opacity=0.1
).add_to(m)

# ===============================
# GEE FUNCTION
# ===============================
@st.cache_data(show_spinner=False)
def fetch_satellite_data(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])

        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(point) \
            .filterDate("2023-01-01", "2023-12-31") \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .median()

        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')

        rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate("2023-01-01", "2023-12-31") \
            .sum()

        temp = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate("2023-01-01", "2023-12-31") \
            .mean()

        maxTemp = temp.select('temperature_2m_max').subtract(273.15)
        minTemp = temp.select('temperature_2m_min').subtract(273.15)

        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem)

        def safe(img, band, scale):
            val = img.reduceRegion(ee.Reducer.mean(), point, scale).get(band)
            return val.getInfo() if val else None

        return {
            "NDVI": safe(ndvi, 'NDVI', 30),
            "Rainfall": safe(rainfall, 'precipitation', 5000),
            "MaxTemp": safe(maxTemp, 'temperature_2m_max', 1000),
            "MinTemp": safe(minTemp, 'temperature_2m_min', 1000),
            "Elevation": safe(dem, 'elevation', 30),
            "Slope": safe(slope, 'slope', 30)
        }

    except:
        return None

# ===============================
# LAYOUT
# ===============================
col1, col2 = st.columns([2,1])

with col1:
    map_data = st_folium(m, height=500, width=700)

# ===============================
# CLICK EVENT
# ===============================
if map_data and map_data["last_clicked"]:

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.info(f"📍 Location: {lat:.4f}, {lon:.4f}")

    if not (KERALA_BOUNDS["lat_min"] <= lat <= KERALA_BOUNDS["lat_max"] and
            KERALA_BOUNDS["lon_min"] <= lon <= KERALA_BOUNDS["lon_max"]):

        st.error("❌ Outside Kerala")

    else:
        with st.spinner("📡 Fetching satellite data..."):
            data = fetch_satellite_data(lat, lon)

        # fallback
        if data is None or data["NDVI"] is None:
            Elevation, Slope = 50, 5
            Rainfall = 200
            MinTemp, MaxTemp = 24, 32
            NDVI = 0.5
        else:
            Elevation = data["Elevation"] or 50
            Slope = data["Slope"] or 5
            Rainfall = data["Rainfall"] or 200
            MinTemp = data["MinTemp"] or 24
            MaxTemp = data["MaxTemp"] or 32
            NDVI = data["NDVI"] or 0.5

        Humidity = 75

        # ===============================
        # 📡 SATELLITE DATA DISPLAY
        # ===============================
        st.subheader("📡 Satellite Data")

        c1, c2, c3 = st.columns(3)
        c1.metric("NDVI 🌿", f"{NDVI:.2f}")
        c2.metric("Rainfall 🌧", f"{Rainfall:.2f}")
        c3.metric("Elevation ⛰", f"{Elevation:.2f}")

        c4, c5 = st.columns(2)
        c4.metric("Max Temp 🌡", f"{MaxTemp:.2f} °C")
        c5.metric("Min Temp 🌡", f"{MinTemp:.2f} °C")

        # ===============================
        # MODEL INPUT
        # ===============================
        Temp_Range = MaxTemp - MinTemp
        Rain_NDVI = Rainfall * NDVI
        Slope_Elev = Slope * Elevation

        input_data = pd.DataFrame([{
            'Elevation': Elevation,
            'Slope': Slope,
            'Rainfall': Rainfall,
            'MinTemp': MinTemp,
            'MaxTemp': MaxTemp,
            'Humidity': Humidity,
            'NDVI': NDVI,
            'Temp_Range': Temp_Range,
            'Rain_NDVI': Rain_NDVI,
            'Slope_Elev': Slope_Elev
        }])

        for col in scaler.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[scaler.feature_names_in_]
        input_scaled = scaler.transform(input_data)

        N = model_N.predict(input_scaled)[0]
        P = np.expm1(model_P.predict(input_scaled)[0])
        K = np.expm1(model_K.predict(input_scaled)[0])

        # ===============================
        # RESULTS UI
        # ===============================
        with col2:
            st.subheader("🌱 Results")

            st.markdown(f"""
            <div class="metric-card">
                <b>🌿 Nitrogen</b>
                <h2>{N:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <b>🌾 Phosphorus</b>
                <h2>{P:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <b>🌳 Potassium</b>
                <h2>{K:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 AI + Remote Sensing | Precision Agriculture System")
