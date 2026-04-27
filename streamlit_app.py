# Mumbai Local Train Crowd Predictor
import streamlit as st
import pickle
import json 
import numpy as np
import pandas as pd
import os
st.set_page_config(
    page_title="Mumbai Local Train Crowd Predictor",
    page_icon="🚂",
    layout="wide"
)
# Load model 
@st.cache_resource
def load_model():
    model      = pickle.load(open("models/crowd_model.pkl", "rb"))
    scaler     = pickle.load(open("models/scaler.pkl",      "rb"))
    le_station = pickle.load(open("models/le_station.pkl",  "rb"))
    le_line    = pickle.load(open("models/le_line.pkl",     "rb"))
    le_crowd   = pickle.load(open("models/le_crowd.pkl",    "rb"))
    meta       = json.load(open("models/metadata.json"))
    return model, scaler, le_station, le_line, le_crowd, meta
model, scaler, le_station, le_line, le_crowd, meta = load_model()
MAJOR_HUBS = {
    "Churchgate", "CSMT", "Dadar", "Andheri", "Borivali",
    "Thane", "Kurla", "Bandra", "Virar", "Kalyan",
    "Panvel", "Vashi", "Lower Parel", "Ghatkopar"
}
STATION_LINE_MAP = {}
for s in ["Churchgate","Marine Lines","Charni Road","Grant Road","Mumbai Central",
          "Mahalaxmi","Lower Parel","Elphinstone Road","Dadar","Matunga Road",
          "Mahim","Bandra","Khar Road","Santacruz","Vile Parle","Andheri",
          "Jogeshwari","Goregaon","Malad","Kandivali","Borivali","Dahisar",
          "Mira Road","Bhayandar","Naigaon","Vasai Road","Nala Sopara","Virar"]:
    STATION_LINE_MAP[s] = "Western"
for s in ["CSMT","Sandhurst Road","Masjid","Bhaykhala","Chinchpokli","Currey Road",
          "Parel","Dadar","Matunga","Sion","Kurla","Vidyavihar","Ghatkopar",
          "Vikhroli","Kanjurmarg","Bhandup","Nahur","Mulund","Thane","Kalwa",
          "Mumbra","Diva","Kopar","Dombivli","Thakurli","Kalyan","Vithalwadi",
          "Ulhasnagar","Ambivali","Badlapur"]:
    STATION_LINE_MAP[s] = "Central"
for s in ["CSMT","Masjid","Sandhurst Road","Dockyard Road","Reay Road",
          "Cotton Green","Sewri","Wadala Road","GTB Nagar","Chunabhatti","Kurla",
          "Tilak Nagar","Chembur","Govandi","Mankhurd","Vashi","Sanpada",
          "Juinagar","Nerul","Seawoods","Belapur","Kharghar","Mansarovar",
          "Khandeshwar","Panvel"]:
    STATION_LINE_MAP[s] = "Harbour"
ALL_STATIONS = sorted(set(STATION_LINE_MAP.keys()))
MONTHS = {
    "January":1,"February":2,"March":3,"April":4,
    "May":5,"June":6,"July (Monsoon)":7,"August (Monsoon)":8,
    "September":9,"October":10,"November":11,"December":12
}
DAYS = {
    "Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
    "Friday":4,"Saturday":5,"Sunday":6
}
CROWD_COLORS = {
    "Low":      "#2ecc71",
    "Moderate": "#f1c40f",
    "High":     "#e67e22",
    "Very High":"#e74c3c"
}
CROWD_EMOJI = {
    "Low": "🟢", "Moderate": "🟡",
    "High": "🟠", "Very High": "🔴"
}
# Helper: build features
def build_features(station, hour, day_of_week, month, is_holiday=0):
    line       = STATION_LINE_MAP.get(station, "Western")
    is_weekend = int(day_of_week >= 5)
    is_peak    = int(hour in range(7,10) or hour in range(17,20))
    is_monsoon = int(month in [6,7,8,9])
    is_major   = int(station in MAJOR_HUBS)
    station_enc = le_station.transform([station])[0]
    line_enc    = le_line.transform([line])[0]
    feat = np.array([[
        station_enc, line_enc,
        hour,
        np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),day_of_week,
        np.sin(2*np.pi*day_of_week/7), np.cos(2*np.pi*day_of_week/7),month,
        np.sin(2*np.pi*month/12),      np.cos(2*np.pi*month/12),
        is_major, is_holiday, is_weekend, is_peak, is_monsoon
    ]])
    return scaler.transform(feat), line
# Helper: predict single
def predict_crowd(station, hour, day_of_week, month, is_holiday=0):
    feat, line   = build_features(station, hour, day_of_week, month, is_holiday)
    pred_enc     = model.predict(feat)[0]
    proba        = model.predict_proba(feat)[0]
    crowd_level  = le_crowd.inverse_transform([pred_enc])[0]
    confidence   = round(float(max(proba)) * 100, 1)
    prob_dict    = {
        cls: round(float(p)*100, 1)
        for cls, p in zip(le_crowd.classes_, proba)
    }
    return crowd_level, confidence, prob_dict, line
# Helper: predict all 24 hours
def predict_all_hours(station, day_of_week, month):
    results = []
    for hour in range(24):
        feat, _ = build_features(station, hour, day_of_week, month)
        pred    = model.predict(feat)[0]
        proba   = model.predict_proba(feat)[0]
        level   = le_crowd.inverse_transform([pred])[0]
        conf    = round(float(max(proba))*100, 1)
        results.append({
            "Hour": hour,
            "Time": f"{hour:02d}:00",
            "Crowd Level": level,
            "Confidence %": conf
        })
    return pd.DataFrame(results)
# Helper: advice text 
def get_advice(level, hour):
    if level == "Low":
        return "✅ Great time to travel! Trains are comfortable and spacious."
    elif level == "Moderate":
        return "🟡 Decent time to travel. Some crowding but manageable."
    elif level == "High":
        if 7 <= hour <= 10:
            return "⚠️ Morning peak hours! Consider travelling before 7am or after 10:30am."
        elif 17 <= hour <= 20:
            return "⚠️ Evening peak hours! Consider waiting until after 8:30pm if possible."
        return "⚠️ Crowded. Be prepared for packed compartments."
    return "🔴 Extremely crowded! Strongly advised to delay travel by 1–2 hours."
# UI STARTS HERE
# Header 
st.title("🚂 Mumbai Local Train Crowd Predictor")
st.caption("Western · Central · Harbour Lines")
#  Model stats row 
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",meta["model_name"])
c2.metric("Accuracy",f"{meta['accuracy']*100:.1f}%")
c3.metric("F1 Score",f"{meta['f1_score']:.4f}")
c4.metric("Stations",len(ALL_STATIONS))
st.divider()
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Crowd",
    "🌡️ Full Day Heatmap",
    "⏰ Best Travel Time"
])
# TAB 1 — SINGLE PREDICTION
with tab1:
    st.subheader("Predict crowd level for a station")
    col1, col2 = st.columns([1, 1])
    with col1:
        station    = st.selectbox("Select Station", ALL_STATIONS, index=ALL_STATIONS.index("Dadar"))
        hour       = st.slider("Hour of Day", 0, 23, 8,
                               help="0 = 12 midnight, 8 = 8am, 17 = 5pm")
        day_name   = st.selectbox("Day of Week", list(DAYS.keys()))
        month_name = st.selectbox("Month", list(MONTHS.keys()), index=3)
        is_holiday = st.checkbox("Public Holiday?")
        predict_btn = st.button("🔮 Predict Crowd Level", use_container_width=True, type="primary")
    with col2:
        if predict_btn:
            dow   = DAYS[day_name]
            month = MONTHS[month_name]
            hol   = 1 if is_holiday else 0
            level, conf, probs, line = predict_crowd(station, hour, dow, month, hol)
            color = CROWD_COLORS[level]
            emoji = CROWD_EMOJI[level]
            # Result card using st components
            st.markdown(f"### {emoji} {level} Crowd")
            st.caption(f"{station} · {line} Line · {day_name}, {hour:02d}:00 – {(hour+1)%24:02d}:00")
            st.info(get_advice(level, hour))
            st.caption(f"Model confidence: **{conf}%**")
            st.markdown("**Probability breakdown:**")
            # Probability bars using Streamlit progress bars
            for cls in ["Low", "Moderate", "High"]:
                pct = probs.get(cls, 0.0)
                st.write(f"{CROWD_EMOJI.get(cls,'⚪')} **{cls}** — {pct}%")
                st.progress(int(pct))
        else:
            st.info("👈 Fill in the details on the left and click **Predict Crowd Level**")
# TAB 2 — FULL DAY HEATMAP
with tab2:
    st.subheader("Full day crowd heatmap (24 hours)")
    st.caption("See crowd levels for every hour of the day at once")
    col1, col2, col3 = st.columns(3)
    with col1:
        hm_station = st.selectbox("Station", ALL_STATIONS,index=ALL_STATIONS.index("Dadar"), key="hm_s")
    with col2:
        hm_day = st.selectbox("Day of Week", list(DAYS.keys()), key="hm_d")
    with col3:
        hm_month= st.selectbox("Month", list(MONTHS.keys()),index=3, key="hm_m")
    hm_btn = st.button("🌡️ Generate Heatmap", use_container_width=True, type="primary")
    if hm_btn:
        dow   = DAYS[hm_day]
        month = MONTHS[hm_month]
        df    = predict_all_hours(hm_station, dow, month)
#  Summary table 
        st.markdown("**Hourly breakdown table:**")
        st.dataframe(df, use_container_width=True,)
#  Peak hours summary
        st.markdown("**Summary:**")
        peak_hours = df[df["Crowd Level"].isin(["High","Very High"])]
        low_hours  = df[df["Crowd Level"] == "Low"]
        col_a, col_b = st.columns(2)
        with col_a:
            if not peak_hours.empty:
                times = ", ".join(peak_hours["Time"].tolist())
                st.error(f"🔴 **Avoid:** {times}")
            else:
                st.success("No highly crowded hours today!")
        with col_b:
            if not low_hours.empty:
                times = ", ".join(low_hours["Time"].head(5).tolist())
                st.success(f"🟢 **Best times:** {times}...")
# TAB 3 — BEST TRAVEL TIME
with tab3:
    st.subheader("Find the best time to travel")
    st.caption("Automatically finds the least crowded hours for your journey")
    col1, col2, col3 = st.columns(3)
    with col1:
        bt_station = st.selectbox("Station", ALL_STATIONS,index=ALL_STATIONS.index("Dadar"), key="bt_s")
    with col2:
        bt_day= st.selectbox("Day of Week", list(DAYS.keys()), key="bt_d")
    with col3:
        bt_month = st.selectbox("Month", list(MONTHS.keys()),index=3, key="bt_m")
    bt_btn = st.button("⏰ Find Best Times", use_container_width=True, type="primary")
    if bt_btn:
        dow   = DAYS[bt_day]
        month = MONTHS[bt_month]
        df    = predict_all_hours(bt_station, dow, month)
# Practical hours only (5am–11pm)
        df_practical = df[(df["Hour"] >= 5) & (df["Hour"] <= 23)].copy()
        order = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
        df_practical["rank"] = df_practical["Crowd Level"].map(order)
        df_sorted = df_practical.sort_values("rank")
        best  = df_sorted.head(3)
        worst = df_sorted.tail(3).sort_values("rank", ascending=False)
# Recommendation banner
        best_row = best.iloc[0]
        st.success(
            f"✅ **Best time to travel from {bt_station} on {bt_day}:** "
            f"{best_row['Time']} — {best_row['Crowd Level']} crowd"
        )
        col_best, col_worst = st.columns(2)
        with col_best:
            st.markdown("### 🟢 Best times to travel")
            for _, row in best.iterrows():
                emoji = CROWD_EMOJI[row["Crowd Level"]]
                st.success(f"{emoji} **{row['Time']}** — {row['Crowd Level']} crowd")
        with col_worst:
            st.markdown("### 🔴 Times to avoid")
            for _, row in worst.iterrows():
                emoji = CROWD_EMOJI[row["Crowd Level"]]
                st.error(f"{emoji} **{row['Time']}** — {row['Crowd Level']} crowd")
        # Line chart of crowd levels across the day
        st.markdown("---")
        st.markdown("**Crowd level trend across the day:**")
        df_chart= df_practical.copy()
        df_chart["rank"] = df_chart["Crowd Level"].map(order)
#  Footer 
st.divider()
st.caption(
    "Mumbai Local Train Crowd Predictor · "
    "Built with Python + Streamlit + scikit-learn · "
)
