from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load and clean the dataset
df = pd.read_excel('./Dataset ML.xlsx').dropna()

# Calculate Power (P = V * I)
df['Power'] = df['V_mp(V)'] * df['I_mp(A)']

# Define features (Temperature, Irradiance) and targets (Voltage, Current, Power)
X = df[['Temperature ( C )', 'Irradiance (W/m^2)']]
y = df[['V_mp(V)', 'I_mp(A)', 'Power']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# MultiOutput Regressor for Decision Tree
dt_model = MultiOutputRegressor(DecisionTreeRegressor())
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_r2 = r2_score(y_test, dt_pred, multioutput='uniform_average')
dt_mse = mean_squared_error(y_test, dt_pred, multioutput='uniform_average')

# MultiOutput Regressor for Random Forest
rf_model = MultiOutputRegressor(RandomForestRegressor())
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred, multioutput='uniform_average')
rf_mse = mean_squared_error(y_test, rf_pred, multioutput='uniform_average')

# MultiOutput Regressor for Gradient Boosting
gb_model = MultiOutputRegressor(GradientBoostingRegressor())
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_r2 = r2_score(y_test, gb_pred, multioutput='uniform_average')
gb_mse = mean_squared_error(y_test, gb_pred, multioutput='uniform_average')

@app.route('/')
def home():
    # Stats for all models
    model_stats = {
        'dt_r2': dt_r2,
        'dt_mse': dt_mse,
        'rf_r2': rf_r2,
        'rf_mse': rf_mse,
        'gb_r2': gb_r2,
        'gb_mse': gb_mse
    }

    dataset_stats = {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'columns': df.columns.tolist(),
        'sample_data': df.head(5).to_dict(orient='records'),
        'desc_stats': df.describe().to_dict()  # Statistical data of dataset
    }

    return render_template('index.html', model_stats=model_stats, dataset_stats=dataset_stats)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    temperature = float(request.form['temperature'])
    irradiance = float(request.form['irradiance'])

    # Preprocess input and predict using the chosen model
    user_input = np.array([[temperature, irradiance]])
    user_input_scaled = scaler.transform(user_input)
    prediction = dt_model.predict(user_input_scaled)

    # Extract predictions for voltage, current, and power
    v_mp_pred, i_mp_pred, power_pred = prediction[0]

    return render_template('predict.html', v_mp_pred=v_mp_pred, i_mp_pred=i_mp_pred, power_pred=power_pred)

if __name__ == '__main__':
    app.run(debug=True, port=53838)
