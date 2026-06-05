
import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Optional NLP imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------
# 1. ML-Modell laden
# ------------------------------------------------

model = joblib.load("best_car_price_model.joblib")

REFERENCE_YEAR = 2020

# Source for fuel and CO2 factors:
# Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3:
# "Overview of GHG factors for different fuel types (DESNZ 2023)".
# GOV.UK/DESNZ publishes the official UK greenhouse gas conversion factors annually.
# These values are used only as additional fuel/CO2 features for this student project.

fuel_reference = {
    "Petrol": {
        "fuel_unit": "litre",
        "kg_co2e_per_unit": 2.10,
        "kg_co2e_per_kwh": 0.22,
        "kwh_per_unit": 9.545455
    },
    "Diesel": {
        "fuel_unit": "litre",
        "kg_co2e_per_unit": 2.51,
        "kg_co2e_per_kwh": 0.24,
        "kwh_per_unit": 10.45833
    },
    "CNG": {
        "fuel_unit": "kg",
        "kg_co2e_per_unit": 2.5625,
        "kg_co2e_per_kwh": 0.18,
        "kwh_per_unit": 14.23611
    },
    "LPG": {
        "fuel_unit": "litre",
        "kg_co2e_per_unit": 1.56,
        "kg_co2e_per_kwh": 0.21,
        "kwh_per_unit": 7.428571
    },
    "Electric": {
        "fuel_unit": "kWh",
        "kg_co2e_per_unit": 0.212,
        "kg_co2e_per_kwh": 0.19,
        "kwh_per_unit": 1.0
    }
}

# ------------------------------------------------
# 2. Sprachmodell lazy laden
# ------------------------------------------------

tokenizer = None
language_model = None

def load_language_model():
    global tokenizer, language_model

    if tokenizer is None or language_model is None:
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        language_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, language_model

# ------------------------------------------------
# 3. Faktoren analysieren
# ------------------------------------------------

def analyse_car_factors(row, predicted_price):
    factors_up = []
    factors_down = []

    age = row.get("Car_Age")
    km = row.get("Kilometers_Driven")
    power = row.get("Power_num")
    transmission = row.get("Transmission")
    owner = row.get("Owner_Type")
    brand = row.get("Brand")
    fuel = row.get("Fuel_Type")

    if age <= 5:
        factors_up.append("the car is relatively new")
    elif age >= 10:
        factors_down.append("the car is older")

    if km <= 40000:
        factors_up.append("the mileage is low")
    elif km >= 90000:
        factors_down.append("the mileage is high")

    if power >= 120:
        factors_up.append("the engine power is strong")
    elif power <= 75:
        factors_down.append("the engine power is modest")

    if transmission == "Automatic":
        factors_up.append("automatic transmission can increase market value")

    if owner == "First":
        factors_up.append("it is a first-owner car")
    else:
        factors_down.append("it had previous owners")

    if brand in ["Mercedes-Benz", "BMW", "Audi", "Porsche", "Jaguar", "Land"]:
        factors_up.append("the brand is positioned as premium")

    if fuel == "Diesel":
        factors_up.append("diesel cars can be valued higher for long-distance use")
    elif fuel in ["CNG", "LPG"]:
        factors_down.append("alternative fuel types may have a smaller market")

    if len(factors_up) == 0:
        factors_up.append("some features support the predicted value")

    if len(factors_down) == 0:
        factors_down.append("there are no major negative factors visible in the input data")

    return factors_up, factors_down

# ------------------------------------------------
# 4. Fallback-Erklärung
# ------------------------------------------------

def build_fallback_explanation(row, predicted_price):
    factors_up, factors_down = analyse_car_factors(row, predicted_price)

    text = f"""Main reason:
The model uses structured car data such as brand, age, mileage, fuel type, transmission, engine size, power and estimated CO2 information to estimate the price.

Factors increasing the price:
"""

    for factor in factors_up:
        text += f"- {factor}\n"

    text += "\nFactors decreasing the price:\n"

    for factor in factors_down:
        text += f"- {factor}\n"

    text += """
Short buying advice:
The prediction should be seen as an estimated market value. The real price can still change depending on condition, service history, accident history, equipment and local demand.
"""

    return text.strip()

# ------------------------------------------------
# 5. NLP-Erklärung generieren
# ------------------------------------------------

def generate_explanation(row, predicted_price):
    fallback = build_fallback_explanation(row, predicted_price)

    try:
        tok, lm = load_language_model()

        factors_up, factors_down = analyse_car_factors(row, predicted_price)

        prompt = f"""
Write a structured explanation for this used car price prediction.

Use exactly this structure:
Main reason:
Factors increasing the price:
Factors decreasing the price:
Short buying advice:

Car information:
Brand: {row.get("Brand")}
Year: {row.get("Year")}
Car age: {row.get("Car_Age")}
Kilometers driven: {row.get("Kilometers_Driven")}
Fuel type: {row.get("Fuel_Type")}
Transmission: {row.get("Transmission")}
Owner type: {row.get("Owner_Type")}
Mileage: {row.get("Mileage_num")}
Engine size: {row.get("Engine_num")}
Power: {row.get("Power_num")}
Estimated CO2 per km: {row.get("estimated_kg_co2e_per_km")}
Predicted price: {predicted_price:.2f}
Positive factors: {", ".join(factors_up)}
Negative factors: {", ".join(factors_down)}
"""

        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = lm.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=4,
            early_stopping=True
        )

        generated = tok.decode(outputs[0], skip_special_tokens=True).strip()

        if len(generated.split()) < 25:
            return fallback

        return generated

    except Exception:
        return fallback


# ------------------------------------------------
# 5B. Prediction uncertainty from Random Forest trees
# ------------------------------------------------

def predict_with_uncertainty(input_data):
    """
    Returns:
    predicted_price, lower_range, upper_range

    The final model is a Random Forest inside a scikit-learn Pipeline.
    We use the predictions from the individual trees to estimate a rough
    uncertainty range. This is not a formal confidence interval, but it
    gives users a more honest range instead of only one exact-looking number.
    """
    predicted_price = float(model.predict(input_data)[0])

    try:
        preprocessor = model.named_steps["preprocessor"]
        rf_model = model.named_steps["model"]

        transformed_input = preprocessor.transform(input_data)

        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()

        tree_predictions = np.array([
            tree.predict(transformed_input)[0]
            for tree in rf_model.estimators_
        ])

        lower_price = float(np.percentile(tree_predictions, 10))
        upper_price = float(np.percentile(tree_predictions, 90))

        lower_price = max(0.0, lower_price)
        upper_price = max(lower_price, upper_price)

        return predicted_price, lower_price, upper_price

    except Exception:
        # fallback if the model is changed later
        margin = max(0.5, predicted_price * 0.15)
        return predicted_price, max(0.0, predicted_price - margin), predicted_price + margin

# ------------------------------------------------
# 6. Prediction-Funktion für App
# ------------------------------------------------

def predict_price(
    brand,
    location,
    year,
    kilometers_driven,
    fuel_type,
    transmission,
    owner_type,
    seats,
    mileage,
    engine,
    power
):
    fuel_data = fuel_reference[fuel_type]

    car_age = REFERENCE_YEAR - year

    if mileage > 0:
        estimated_kg_co2e_per_km = fuel_data["kg_co2e_per_unit"] / mileage
    else:
        estimated_kg_co2e_per_km = 0.123529

    input_data = pd.DataFrame([{
        "Location": location,
        "Year": year,
        "Kilometers_Driven": kilometers_driven,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Seats": seats,
        "Mileage_num": mileage,
        "Engine_num": engine,
        "Power_num": power,
        "Brand": brand,
        "Car_Age": car_age,
        "kg_co2e_per_unit": fuel_data["kg_co2e_per_unit"],
        "kg_co2e_per_kwh": fuel_data["kg_co2e_per_kwh"],
        "kwh_per_unit": fuel_data["kwh_per_unit"],
        "estimated_kg_co2e_per_km": estimated_kg_co2e_per_km
    }])

    predicted_price, lower_price, upper_price = predict_with_uncertainty(input_data)

    explanation = generate_explanation(input_data.iloc[0], predicted_price)

    result_text = (
        f"Predicted price: approx. {predicted_price:.2f} Lakh ₹\n"
        f"Estimated uncertainty range: {lower_price:.2f} - {upper_price:.2f} Lakh ₹\n\n"
        f"{explanation}\n\n"
        "Transparency note: If the language model output is too short or unclear, "
        "the app uses a rule-based fallback explanation based on the car factors."
    )

    return result_text

# ------------------------------------------------
# 7. Gradio UI
# ------------------------------------------------

brands = [
    "Maruti", "Hyundai", "Honda", "Toyota", "Mercedes-Benz", "BMW",
    "Audi", "Volkswagen", "Ford", "Renault", "Mahindra", "Tata",
    "Porsche", "Jaguar", "Land"
]

locations = [
    "Mumbai", "Pune", "Chennai", "Coimbatore", "Hyderabad",
    "Jaipur", "Kochi", "Kolkata", "Delhi", "Bangalore", "Ahmedabad"
]

demo = gr.Interface(
    fn=predict_price,
    flagging_mode="never",
    inputs=[
        gr.Dropdown(brands, value="Hyundai", label="Brand"),
        gr.Dropdown(locations, value="Mumbai", label="Location"),
        gr.Slider(1998, 2019, value=2015, step=1, label="Year"),
        gr.Number(value=50000, label="Kilometers driven"),
        gr.Dropdown(["Petrol", "Diesel", "CNG", "LPG", "Electric"], value="Diesel", label="Fuel type"),
        gr.Dropdown(["Manual", "Automatic"], value="Manual", label="Transmission"),
        gr.Dropdown(["First", "Second", "Third", "Fourth & Above"], value="First", label="Owner type"),
        gr.Slider(2, 10, value=5, step=1, label="Seats"),
        gr.Number(value=18.0, label="Mileage"),
        gr.Number(value=1200, label="Engine size"),
        gr.Number(value=90.0, label="Power")
    ],
    outputs=gr.Textbox(label="Prediction and explanation", lines=26),
    title="Used Car Price Estimator with NLP Explanation",
    description="This app combines ML Numeric Data and NLP. A Random Forest model predicts the used car price in Lakh ₹ and estimates a rough uncertainty range. A language model explains the prediction; if the generated explanation is too short, the app uses a transparent rule-based fallback explanation. Note: the Brand feature is simplified from the first word of the car name in the dataset."
)

if __name__ == "__main__":
    demo.launch()
