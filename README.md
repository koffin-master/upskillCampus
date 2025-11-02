
Mining Process – Silica Concentration Prediction

Project Overview:
This project focuses on predicting the percentage of Silica (% Silica Concentrate) in iron ore produced in a flotation mining process. The model uses real industrial time-series data to forecast silica impurity levels ahead of time, allowing engineers to take corrective actions to improve quality and reduce waste in production.

Objective:
	•	Predict silica impurity in iron ore concentrate.
	•	Provide early warning for quality engineers.
	•	Test prediction performance with and without Iron Concentrate feature.
	•	Build a complete machine learning time-series forecasting pipeline.

Dataset:
Real industrial flotation plant dataset
Time period: March 2017 – September 2017
Size: ~737,000 rows
Mixed sampling rates: 20-second sensor readings + hourly lab results

Due to dataset size, it is not included in this repository.

Dataset Download Link:
Download from Google Drive and place the file in the data folder:
[https://drive.google.com/file/d/YOUR_FILE_ID/view](https://drive.google.com/file/d/1rCozT2HOnUhkNuTD1LAud-VDQjV-NnlU/view?usp=sharing)

After downloading, create a folder named data and place:
MiningProcess_Flotation_Plant_Database.csv

Machine Learning Approach:
	•	Cleaned industrial numeric data with comma formats.
	•	Converted timestamps to datetime and resampled to 1-minute intervals.
	•	Engineered lag features (1, 5, 10, 30, 60 minutes).
	•	Computed rolling mean and standard deviation.
	•	Added time-based features (hour, minute, weekday).
	•	Trained Random Forest model and evaluated persistence baseline.
	•	Forecasted silica up to 24 hours ahead.
	•	Compared results with and without Iron Concentrate feature.

Key Results:
	•	Accurate short-term prediction for silica concentration.
	•	Prediction error increases for higher forecast horizon (expected).
	•	Silica can still be predicted without Iron Concentrate, but accuracy improves when it is included.
	•	Outperformed persistence baseline in multiple horizons.

Project Structure:
Mining_Silica_Prediction
├─ data/ (place dataset here)
├─ src/ (Silica.py main script)
└─ README.txt

Requirements:
Install dependencies:
pip install pandas numpy scikit-learn matplotlib

How to Run:
	1.	Download dataset from Google Drive and store in data folder.
	2.	Run the script:
python src/Silica.py

Features:
	•	Real industrial time-series dataset
	•	Data cleaning and preprocessing pipeline
	•	Multi-step forecasting
	•	Feature comparison with and without iron concentration
	•	Performance comparison against baseline

Future Improvements:
	•	Use advanced models (LSTM, GRU, Transformers)
	•	Create dashboard (Streamlit or Plotly)
	•	Add model explainability (SHAP)
	•	Real-time deployment pipeline
