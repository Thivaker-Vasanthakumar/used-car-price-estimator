from pathlib import Path

# -----------------------------
# Patch app.py
# -----------------------------
app_path = Path("app.py")
text = app_path.read_text(encoding="utf-8")

source_comment = '''# Source for fuel and CO2 factors:
# Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3:
# "Overview of GHG factors for different fuel types (DESNZ 2023)".
# GOV.UK/DESNZ publishes the official UK greenhouse gas conversion factors annually.
# These values are used only as additional fuel/CO2 features for this student project.
'''

if source_comment not in text:
    text = text.replace("fuel_reference = {\n", source_comment + "\nfuel_reference = {\n")

uncertainty_function = r'''
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
'''

if "def predict_with_uncertainty(" not in text:
    text = text.replace(
        "# ------------------------------------------------\n# 6. Prediction-Funktion für App\n# ------------------------------------------------",
        uncertainty_function + "\n# ------------------------------------------------\n# 6. Prediction-Funktion für App\n# ------------------------------------------------"
    )

# Remove duplicate price line inside fallback explanation
text = text.replace(
    '    text = f"""The predicted used car price is {predicted_price:.2f}.\n\nMain reason:',
    '    text = f"""Main reason:'
)

old_prediction_block = '''    predicted_price = model.predict(input_data)[0]

    explanation = generate_explanation(input_data.iloc[0], predicted_price)

    result_text = f"Predicted price: {predicted_price:.2f}\\n\\n{explanation}"

    return result_text'''

new_prediction_block = '''    predicted_price, lower_price, upper_price = predict_with_uncertainty(input_data)

    explanation = generate_explanation(input_data.iloc[0], predicted_price)

    result_text = (
        f"Predicted price: approx. {predicted_price:.2f} Lakh ₹\\n"
        f"Estimated uncertainty range: {lower_price:.2f} - {upper_price:.2f} Lakh ₹\\n\\n"
        f"{explanation}\\n\\n"
        "Transparency note: If the language model output is too short or unclear, "
        "the app uses a rule-based fallback explanation based on the car factors."
    )

    return result_text'''

if old_prediction_block in text:
    text = text.replace(old_prediction_block, new_prediction_block)

# Update description
old_desc = 'description="This app combines ML Numeric Data and NLP. A Random Forest model predicts the used car price, and a language model explains the prediction."'
new_desc = 'description="This app combines ML Numeric Data and NLP. A Random Forest model predicts the used car price in Lakh ₹ and estimates a rough uncertainty range. A language model explains the prediction; if the generated explanation is too short, the app uses a transparent rule-based fallback explanation. Note: the Brand feature is simplified from the first word of the car name in the dataset."'

text = text.replace(old_desc, new_desc)

app_path.write_text(text, encoding="utf-8")


# -----------------------------
# Patch README.md
# -----------------------------
readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")

readme = readme.replace(
    "The application combines two AI blocks:",
    "The application combines two AI blocks. Prices are shown in Lakh ₹, which is the price unit used in the original dataset:"
)

readme = readme.replace(
    "1. predicted used car price\n2. NLP explanation of the prediction",
    "1. predicted used car price in Lakh ₹\n2. estimated uncertainty range based on the individual Random Forest trees\n3. NLP explanation of the prediction"
)

readme = readme.replace(
    "A second data source was added as a fuel and CO2 reference table. It contains fuel-related information for Petrol, Diesel, CNG, LPG and Electric.",
    "A second data source was added as a fuel and CO2 reference table. It contains fuel-related information for Petrol, Diesel, CNG, LPG and Electric. The values are based on the Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3, which uses DESNZ 2023 greenhouse gas conversion factors."
)

readme = readme.replace(
    "The structured prompt was selected for the final application because it gives clearer and more useful explanations.",
    "The structured prompt was selected for the final application because it gives clearer and more useful explanations. If the language model returns an output that is too short or unclear, the app uses a transparent rule-based fallback explanation based on the car factors."
)

readme = readme.replace(
    "The application uses a saved machine learning model for inference. The training process is separated from the deployed app.",
    "The application uses a saved machine learning model for inference. The training process is separated from the deployed app. The Brand feature is simplified from the first word of the car name in the dataset; for example, Land Rover appears as Land in the learned Brand feature."
)

readme_path.write_text(readme, encoding="utf-8")


# -----------------------------
# Patch documentation.md
# -----------------------------
doc_path = Path("documentation.md")
doc = doc_path.read_text(encoding="utf-8")

doc = doc.replace(
    "| 2 | Fuel / CO2 reference table | Small structured reference table | 5 fuel types | Added fuel and CO2 features through `Fuel_Type` |",
    "| 2 | Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3: Overview of GHG factors for different fuel types (DESNZ 2023), based on GOV.UK/DESNZ greenhouse gas conversion factors | Small structured reference table | 5 fuel types | Added fuel and CO2 features through `Fuel_Type` |"
)

doc = doc.replace(
    "* Feature engineering and selection: Created `Mileage_num`, `Engine_num`, `Power_num`, `Brand`, `Car_Age`, `kg_co2e_per_unit`, `kg_co2e_per_kwh`, `kwh_per_unit` and `estimated_kg_co2e_per_km`.",
    "* Feature engineering and selection: Created `Mileage_num`, `Engine_num`, `Power_num`, `Brand`, `Car_Age`, `kg_co2e_per_unit`, `kg_co2e_per_kwh`, `kwh_per_unit` and `estimated_kg_co2e_per_km`. The fuel and CO2 values come from Energy Saving Trust / DESNZ 2023 factors and are joined through `Fuel_Type`."
)

doc = doc.replace(
    "* Results: The structured prompt was selected because it gives clearer explanations than the simple prompt. A fallback explanation was added because the small free model sometimes produced very short answers.",
    "* Results: The structured prompt was selected because it gives clearer explanations than the simple prompt. A fallback explanation was added because the small free model sometimes produced very short answers. The app explicitly tells the user when this rule-based fallback logic may be used."
)

doc = doc.replace(
    "* Main user flow: The user enters car attributes in the Gradio interface, clicks Submit, receives a predicted used car price and a natural-language explanation.",
    "* Main user flow: The user enters car attributes in the Gradio interface, clicks Submit, receives a predicted used car price in Lakh ₹, a rough uncertainty range estimated from the Random Forest trees, and a natural-language explanation."
)

doc = doc.replace(
    "* Reproducibility notes: The deployed app pins `scikit-learn==1.6.1` because the saved model was trained with this version. Training and inference are separated: the notebook trains and saves the model, while the app only loads the saved model and predicts.",
    "* Reproducibility notes: The deployed app pins `scikit-learn==1.6.1` because the saved model was trained with this version. Training and inference are separated: the notebook trains and saves the model, while the app only loads the saved model and predicts. The uncertainty range is calculated during inference from the existing Random Forest trees and does not require retraining. The Brand feature is simplified from the first word of the car name in the dataset; for example, Land Rover appears as Land in the learned feature."
)

doc = doc.replace(
    "* Ethics, bias, or fairness analysis: The dataset represents a sample of used car listings and may not cover all markets equally.",
    "* Ethics, bias, or fairness analysis: The dataset represents a sample of used car listings and may not cover all markets equally. The output is shown as an estimate with an uncertainty range instead of a single exact-looking value."
)

doc_path.write_text(doc, encoding="utf-8")

print("Patch completed: app.py, README.md and documentation.md were updated.")
