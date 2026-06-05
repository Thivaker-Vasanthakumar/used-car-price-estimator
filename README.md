# Used Car Price Estimator with NLP Explanation

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

The target variable is Price.

Two models were trained and compared:

- Linear Regression
- Random Forest Regressor

The final model is the Random Forest model because it achieved better results.

### Block 2: NLP

The NLP part explains the prediction in natural language.

The predicted price from the ML model is used as input for the NLP explanation. The explanation includes:

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

The dataset contains around 6,000 used car records with structured features and the target variable Price.

### Data Source 2: Fuel / CO2 Reference Table

A second data source was added as a fuel and CO2 reference table. It contains fuel-related information for Petrol, Diesel, CNG, LPG and Electric. The values are based on the Energy Saving Trust Fleet Decarbonisation Toolkit, Table 3, which uses DESNZ 2023 greenhouse gas conversion factors.

This table was joined with the car dataset using the Fuel_Type column. It was used to create additional fuel and CO2-related features.

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

- notebook.ipynb: complete development notebook
- app.py: Gradio application
- requirements.txt: required Python packages
- best_car_price_model.joblib: saved machine learning model
- model_features.json: model feature information
- documentation.md: project documentation using the required template
- screenshots/: screenshots of the running app

## How to Run Locally

1. Clone the repository.
2. Install the requirements with: pip install -r requirements.txt
3. Run the app with: python app.py
4. Open the local Gradio URL in the browser.

## Notes

The project does not use the Zurich apartment dataset or the dog breed image dataset that were used during the semester.

The application uses a saved machine learning model for inference. The training process is separated from the deployed app. The Brand feature is simplified from the first word of the car name in the dataset; for example, Land Rover appears as Land in the learned Brand feature.