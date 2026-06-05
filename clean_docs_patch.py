from pathlib import Path

documentation = """# AI Applications Project Documentation Template

Use this template to document your project concisely and completely. Fill in all required fields. Keep answers short and precise.

## Documentation Hint

When possible, the corresponding code location is referenced directly in the description.

## Project Metadata

* Project title: Used Car Price Estimator with NLP Explanation
* Student: Thivaker Vasanthakumar
* GitHub repository URL: https://github.com/Thivaker-Vasanthakumar/used-car-price-estimator
* Deployment URL: https://huggingface.co/spaces/vasanthi8134/used-car-price-estimator
* Submission date: 05.06.2026

### Mandatory Setup Checks

* At least 2 blocks selected: Yes
* Multiple and different data sources used: Yes
* Deployment URL provided: Yes
* Required GitHub users added to repository (`jasminh`, `bkuehnis`): Yes

## Selected AI Blocks

* ML Numeric Data
* NLP
* Computer Vision: N/A

Primary blocks used for core solution:

* Primary block 1: ML Numeric Data
* Primary block 2: NLP

The project combines a numeric machine learning model for used car price prediction with an NLP component that explains the prediction in natural language.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

* Problem statement: Used car buyers often have difficulty understanding whether a listed price is reasonable and which car attributes influence the price.
* Goal: Build a web application that predicts a used car price from structured car data and explains the prediction in clear natural language.
* Success criteria: The app predicts a price, generates a readable explanation, is deployed online, and documents data sources, preprocessing, model comparison, NLP prompt comparison and evaluation.

### 1.2 Integration Logic

* How the selected blocks interact: The ML model first predicts the used car price. The predicted price and the user input are then passed to the NLP component, which generates an explanation.
* Data and output flow between blocks: User input -> preprocessing/features -> Random Forest prediction -> predicted price + car features -> NLP explanation -> Gradio output.

Evidence: See `app.py`, especially the functions `predict_price`, `predict_with_uncertainty`, `analyse_car_factors`, `generate_explanation` and `build_fallback_explanation`.

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | https://raw.githubusercontent.com/sagnikghoshcr7/Car-Price-Prediction/master/data/dataset.csv | CSV table | 6019 rows, 13 original columns | Main used car dataset with features and target variable `Price` |
| 2 | Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3: Overview of GHG factors for different fuel types (DESNZ 2023), based on GOV.UK/DESNZ greenhouse gas conversion factors | Small structured reference table | 5 fuel types | Added fuel and CO2 features through `Fuel_Type` |

#### 2A.2 Preprocessing and Features

* Exploratory data analysis (EDA): The original dataset contains 6019 car records and 13 columns. The initial analysis checked dataset shape, data types, missing values, descriptive statistics and category frequencies.
* EDA key finding 1: The price distribution is right-skewed. Most cars are in a lower to mid price range, while very expensive premium cars are rare.
* EDA key finding 2: Diesel and petrol are the dominant fuel types. Manual transmission is more common than automatic transmission, and first-owner cars are the largest owner group.
* EDA key finding 3: Scatter plots show that newer cars tend to have higher prices, while older cars usually have lower prices. Expensive premium cars appear as outliers and later also show higher prediction errors.
* Cleaning steps: Removed unnecessary index columns, removed `New_Price` because it had too many missing values, removed extreme kilometer outliers, and handled missing values.
* Preprocessing steps: Converted text-based numeric columns such as `Mileage`, `Engine` and `Power` into numeric columns.
* Feature engineering and selection: Created `Mileage_num`, `Engine_num`, `Power_num`, `Brand`, `Car_Age`, `kg_co2e_per_unit`, `kg_co2e_per_kwh`, `kwh_per_unit` and `estimated_kg_co2e_per_km`.
* Car age calculation: `Car_Age` was calculated using reference year 2020, because the newest cars in the dataset are from 2019.
* Fuel and CO2 features: The fuel and CO2 values come from Energy Saving Trust / DESNZ 2023 factors and are joined through `Fuel_Type`.

Source: Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3 (https://fleetdecarbonisationtoolkit.energysavingtrust.org.uk/t/decarbonisation-strategy/emissions-calculated/car-van-ghg-kwh-calculations-2/), based on GOV.UK/DESNZ greenhouse gas conversion factors 2023 (https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2023).

Evidence: See notebook sections `SCHRITT 1: DATEN LADEN + ERSTER ÜBERBLICK`, `SCHRITT 3: DATEN PUTZEN + FEATURES BAUEN` and `SCHRITT 4: ZWEITE DATENQUELLE EINBAUEN`.

#### 2A.3 Model Selection

* Models tested: Linear Regression and Random Forest Regressor.
* Why these models were chosen: Linear Regression was used as a simple baseline. Random Forest was used because it can model non-linear relationships between car features and price.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
|---|---|---|---|---|---|
| 1 | Create baseline model | Basic preprocessing and train-test split | Linear Regression | RMSE | Baseline result |
| 2 | Improve prediction quality | Add Random Forest model | Random Forest Regressor | RMSE | Strong improvement over baseline |
| 3 | Select final model | Save best model with joblib | Random Forest Regressor | RMSE | Final model selected |

#### 2A.5 Evaluation and Error Analysis

* Metrics used: MAE, RMSE and R2.
* Final results: Random Forest achieved MAE 1.40, RMSE 2.95 and R2 0.92. Linear Regression achieved MAE 2.90, RMSE 4.83 and R2 0.80.
* Error patterns and likely causes: The largest prediction errors occur mostly for expensive premium cars such as BMW, Mercedes-Benz, Porsche, Audi and Jaguar. This is likely because the dataset does not include important factors such as exact equipment, service history, accident history, condition and local demand.
* Additional analysis: The deployed app shows a rough uncertainty range calculated from the individual Random Forest tree predictions. This avoids presenting the model output as one exact-looking value.

Evidence: See notebook section `SCHRITT 5: MACHINE LEARNING MODELLE TRAINIEREN` and `app.py`, function `predict_with_uncertainty`.

#### 2A.6 Integration with Other Block(s)

* Inputs received from other block(s): User input from the Gradio app.
* Outputs provided to other block(s): Predicted used car price and uncertainty range, which are used by the NLP block to generate and contextualize the explanation.

Evidence: See `app.py`, function `predict_price`.

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | User input from Gradio app | Structured text/numeric input | One car at a time | Used to build the explanation prompt |
| 2 | ML model output | Numeric prediction | One predicted price per input | Used as central information in the explanation |
| 3 | Prompt comparison output | CSV | 3 sample explanations | Used to compare simple and structured prompt variants |

#### 2B.2 Preprocessing and Prompt Design

* Text preprocessing: The structured car input is converted into a text summary containing brand, year, car age, kilometers, fuel type, transmission, owner type, mileage, engine size, power, CO2 estimate and predicted price.
* Prompt design or retrieval setup: Two prompt variants were tested: a simple explanation prompt and a structured explanation prompt.
* The structured prompt asks for main reason, factors increasing price, factors decreasing price and short buying advice.

Evidence: See notebook section `SCHRITT 6B: BESSERE NLP-ERKLÄRUNG + FALLBACK`.

#### 2B.3 Approach Selection

* Approach used: Transformer-based text generation with `google/flan-t5-small`, prompt engineering and fallback explanation logic.
* Alternatives considered: OpenAI API was considered as a possible later upgrade for smoother explanations, but the final solution uses a free local Hugging Face model to avoid API-key dependency.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
|---|---|---|---|---|---|
| 1 | Test NLP generation | Basic prompt with car data and predicted price | `google/flan-t5-small`, simple prompt | Qualitative readability | Output was too short |
| 2 | Improve explanation quality | Added structured prompt | `google/flan-t5-small`, structured prompt | Completeness and clarity | More useful explanation |
| 3 | Stabilize final app output | Added rule-based factor analysis and fallback explanation | Structured prompt + fallback | Reliability | App always returns a clear explanation |

#### 2B.5 Evaluation and Error Analysis

* Evaluation strategy: Qualitative comparison of prompt outputs using sample cars. The outputs were checked for readability, completeness and whether they refer to relevant car features.
* Results: The structured prompt was selected because it gives clearer explanations than the simple prompt.
* Error patterns and likely causes: The free language model sometimes returns short or incomplete text.
* Mitigation: A rule-based fallback explanation was added because the small free model sometimes produced very short answers. The app explicitly tells the user when this fallback logic may be used.

Evidence: See `nlp_prompt_comparison_improved_step6b.csv` and `app.py`, functions `generate_explanation` and `build_fallback_explanation`.

#### 2B.6 Integration with Other Block(s)

* Inputs received from other block(s): Predicted price from the ML model and the same car features used for prediction.
* Outputs provided to other block(s): Natural-language explanation shown in the Gradio app.
* The NLP component enhances interpretation and user interaction by explaining the numeric model output in understandable language.

Evidence: See `app.py`, function `predict_price`.

---

### 2C. Computer Vision

N/A. Computer Vision was not selected. The project uses ML Numeric Data and NLP as the two required AI blocks.

---

## 3. Deployment

* Deployment URL: https://huggingface.co/spaces/vasanthi8134/used-car-price-estimator
* Main user flow: The user enters car attributes in the Gradio interface, clicks Submit, receives a predicted used car price in Lakh ₹, a rough uncertainty range estimated from the Random Forest trees, and a natural-language explanation.
* Screenshot or short demo: See `screenshots/screenshot_01_hyundai_prediction.png` and `screenshots/screenshot_02_bmw_prediction.png`.

The deployed app uses `app.py`, `requirements.txt` and the saved model file `best_car_price_model.joblib`.

---

## 4. Execution Instructions

* Environment setup: Python environment with packages from `requirements.txt`.
* Data setup: The raw car dataset is loaded from the public CSV URL in the notebook. The fuel / CO2 reference table is created in the notebook and joined through `Fuel_Type`.
* Training command(s): Run the notebook from the data loading section through the model training section. The final model is saved as `best_car_price_model.joblib`.
* Inference/run command(s): Install dependencies with `pip install -r requirements.txt` and run the app with `python app.py`.
* Reproducibility notes: The deployed app pins `scikit-learn==1.6.1` because the saved model was trained with this version.
* Training and inference are separated: the notebook trains and saves the model, while the app only loads the saved model and predicts.
* The uncertainty range is calculated during inference from the existing Random Forest trees and does not require retraining.
* The Brand feature is simplified from the first word of the car name in the dataset; for example, Land Rover appears as Land in the learned feature.

---

## 5. Optional Bonus Evidence

* Ethics, bias, or fairness analysis: The dataset represents a sample of used car listings and may not cover all markets equally.
* The output is shown as an estimate with an uncertainty range instead of a single exact-looking value.
* The model should not be interpreted as an exact market valuation.
* Important real-world factors such as accident history, service records, vehicle condition and local demand are not included.
* The app includes a short buying advice section to communicate that the prediction is only an estimate.

## Footer

End of project documentation.
"""

Path("documentation.md").write_text(documentation, encoding="utf-8")


readme = """# Used Car Price Estimator with NLP Explanation

## Project Overview

This project is an AI application that estimates the price of a used car and explains the prediction in natural language.

The application combines two AI blocks. Prices are shown in Lakh ₹, which is the price unit used in the original dataset:

1. ML Numeric Data: A machine learning model predicts the used car price from structured car data.
2. NLP: A language model generates a human-readable explanation of the predicted price.

The goal is to help users understand not only the estimated price, but also which factors influence it.

## Live Demo

Hugging Face Space:

https://huggingface.co/spaces/vasanthi8134/used-car-price-estimator

## AI Blocks Used

### Block 1: ML Numeric Data

The machine learning part uses structured car data such as brand, location, year, kilometers driven, fuel type, transmission, owner type, mileage, engine size, power and estimated CO2 information.

The target variable is `Price`.

Two models were trained and compared:

- Linear Regression
- Random Forest Regressor

The final model is the Random Forest model because it achieved better results.

### Block 2: NLP

The NLP part explains the prediction in natural language. The predicted price from the ML model is used as input for the NLP explanation.

The explanation includes:

- main reason for the price
- factors increasing the price
- factors decreasing the price
- short buying advice

Two prompt variants were compared:

- simple prompt
- structured prompt

The structured prompt was selected for the final application because it gives clearer and more useful explanations. If the language model returns an output that is too short or unclear, the app uses a transparent rule-based fallback explanation based on the car factors.

## Data Sources

### Data Source 1: Used Car Dataset

Used car dataset from Kaggle / public GitHub copy.

URL used in the project:

https://raw.githubusercontent.com/sagnikghoshcr7/Car-Price-Prediction/master/data/dataset.csv

The dataset contains around 6,000 used car records with structured features and the target variable `Price`.

### Data Source 2: Fuel / CO2 Reference Table

A second data source was added as a fuel and CO2 reference table. It contains fuel-related information for Petrol, Diesel, CNG, LPG and Electric.

The values are based on the Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3, which uses DESNZ 2023 greenhouse gas conversion factors.

Source links:

- Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3: https://fleetdecarbonisationtoolkit.energysavingtrust.org.uk/t/decarbonisation-strategy/emissions-calculated/car-van-ghg-kwh-calculations-2/
- GOV.UK DESNZ greenhouse gas conversion factors 2023: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2023

This table was joined with the car dataset using the `Fuel_Type` column. It was used to create additional fuel and CO2-related features.

## EDA Summary

The initial EDA checked dataset shape, data types, missing values, descriptive statistics and category frequencies. The price distribution is right-skewed, meaning most cars are in a lower to mid price range while expensive premium cars are rare. Diesel and petrol are the dominant fuel types, manual transmission is more common than automatic transmission, and newer cars tend to have higher prices.

## Model Performance

The following models were compared:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Random Forest | 1.40 | 2.95 | 0.92 |
| Linear Regression | 2.90 | 4.83 | 0.80 |

The Random Forest model performed better and was selected as the final model.

## Error Analysis

The model performs better for cheaper and mid-range cars. The largest errors occur mainly for expensive premium cars such as BMW, Mercedes-Benz, Porsche, Audi and Jaguar.

This makes sense because the dataset does not include all important premium-car factors, such as exact equipment level, service history, accident history, condition and local demand.

## Application

The app was built with Gradio and deployed on Hugging Face Spaces.

The user enters car information, and the app returns:

1. predicted used car price in Lakh ₹
2. estimated uncertainty range based on the individual Random Forest trees
3. NLP explanation of the prediction

## Files in this Repository

- `notebook.ipynb`: complete development notebook
- `app.py`: Gradio application
- `requirements.txt`: required Python packages
- `best_car_price_model.joblib`: saved machine learning model
- `model_features.json`: model feature information
- `documentation.md`: project documentation using the required template
- `screenshots/`: screenshots of the running app

## How to Run Locally

1. Clone the repository.
2. Install the requirements with: `pip install -r requirements.txt`
3. Run the app with: `python app.py`
4. Open the local Gradio URL in the browser.

## Notes

The project does not use the Zurich apartment dataset or the dog breed image dataset that were used during the semester.

The application uses a saved machine learning model for inference. The training process is separated from the deployed app.

The Brand feature is simplified from the first word of the car name in the dataset; for example, Land Rover appears as Land in the learned Brand feature.
"""

Path("README.md").write_text(readme, encoding="utf-8")

print("Clean documentation.md and README.md were rewritten successfully.")
