🚂 Mumbai Local Train Crowd Predictor

A Machine Learning project that predicts crowd levels in Mumbai local trains based on time, station, and day.

Built using Python, Streamlit, and Scikit-learn.

📌 Project Overview

Mumbai local trains are often overcrowded, especially during peak hours.
This project helps users predict crowd levels in advance and choose better travel times.

The system uses a trained ML model and provides an interactive web interface.

🎯 Features
🔮 Predict crowd level (Low / Moderate / High)
⏰ Find best time to travel
🌡️ View full-day crowd distribution
📊 Model accuracy and performance display
🖥️ Simple and clean Streamlit UI
🧠 Machine Learning
Algorithm: Random Forest Classifier
Feature Engineering:
Time (hour, day, month)
Station & railway line
Peak hours, weekends, holidays
Cyclical encoding (sin & cos)
Evaluation Metrics:
Accuracy
F1 Score
📂 Project Structure
mumbai_crowd_predictor/
│
├── data/
│   └── mumbai_crowd_data.csv
│
├── models/
│   ├── crowd_model.pkl
│   ├── scaler.pkl
│   ├── le_station.pkl
│   ├── le_line.pkl
│   ├── le_crowd.pkl
│   └── metadata.json
│
├── train_model.py
├── streamlit_app.py
└── README.md
