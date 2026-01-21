from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/wine_cultivar_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

FEATURE_NAMES = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input from form
    alcohol = float(request.form['alcohol'])
    malic_acid = float(request.form['malic_acid'])
    ash = float(request.form['ash'])
    alcalinity = float(request.form['alcalinity_of_ash'])
    magnesium = float(request.form['magnesium'])
    total_phenols = float(request.form['total_phenols'])

    # Create DataFrame
    input_data = pd.DataFrame([[alcohol, malic_acid, ash, alcalinity, magnesium, total_phenols]],
                              columns=FEATURE_NAMES)
    
    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    result = f"Cultivar {prediction + 1}"  # display as 1,2,3

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
