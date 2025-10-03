# Housing Prices Prediction (Kaggle Competition)
## 📌 Project Overview
This repository contains my solution for the Housing Prices Competition for Kaggle Learn Users.
The goal of this competition is to predict house sale prices based on various property features. It is a classic regression problem, widely used as an introductory project for machine learning practice.
 
## 📂 Project Structure
├── data/              # Dataset folder (not uploaded to GitHub)
├── eda_outputs/       # EDA report, plot images
├── pipelines/         # Python scripts (EDA, data cleaning, modeling, RF LOG VS RAW, diagnostic, predict)
├── rf_clean_outputs
├── rf_compare_outputs
└── README.md          # Project documentation
 
## 📊 Dataset
The dataset is provided by Kaggle. Note: Due to Kaggle’s rules, the dataset files (train.csv, test.csv) are not included in this repository.
You can download the dataset directly from the competition page:
👉 House Prices Dataset
Main files:
•	train.csv: Training data with features and sale prices
•	test.csv: Test data with features (target values not provided)
•	sample_submission.csv: Example submission file
 
## 🔧 How to Run
### 1. Clone the repository
git clone https://github.com/<your-username>/housing-prices-kaggle.git
cd housing-prices-kaggle
### 2. Install dependencies
pip install -r requirements.txt
### 3. Download the dataset
Place the train.csv and test.csv files inside the data/ folder.
### 4. Run Python scripts
#### a. prepares the clean data for train.csv
python pipelines/clean_for_random_forest.py 
--input data/train.csv 
--outdir rf_clean_outputs 
--target SalePrice

#### b. generates the models
python pipelines/train_rf_from_clean.py \
  --x rf_clean_outputs/X_clean.csv \
  --y rf_clean_outputs/y.csv \
  --outdir rf_model_outputs \
  --test-size 0.2 \
  --random-state 42 \
  --n-estimators 1200

#### c. compares 2 models (log vs raw)
python pipelines/compare_rf_raw_vs_log.py \
  --x rf_clean_outputs/X_clean.csv \
  --y rf_clean_outputs/y.csv \
  --outdir rf_compare_outputs \
  --test-size 0.2 \
  --random-state 42 \
  --n-estimators 1200

#### d. prepares the clean data for test.csv
python pipelines/clean_for_random_forest.py \
  --input data/train.csv \
  --outdir rf_clean_outputs \
  --save_pipeline rf_clean_outputs/cleaner.joblib

Saved:
 - rf_clean_outputs/feature_names.txt
 - rf_clean_outputs/X_clean.csv
 - rf_clean_outputs/y.csv
Cleaner saved to: rf_clean_outputs/cleaner.joblib

python pipelines/clean_for_random_forest.py \
  --input data/test.csv \
  --outdir rf_clean_outputs \
  --load_pipeline rf_clean_outputs/cleaner.joblib \
  --no-save-y \
  --output-name X_submit_clean.csv

Saved:
 - rf_clean_outputs/feature_names.txt
 - rf_clean_outputs/X_submit_clean.csv

#### e. predict the sales prices
python pipelines/predict_rf_from_clean.py \
  --model rf_model_outputs_raw/model.joblib \
  --x-submit rf_clean_outputs/X_submit_clean.csv \
  --test data/test.csv \
  --feat-names rf_clean_outputs/feature_names.txt \
  --out data/submission.csv

 
## 🧠 Methods & Models
•	Data Cleaning: handling missing values, rare category group, No garage: GarageYrBlt = 0, LotFrontage: filled with median of Neighborhood, Set cap for GrLivArea, 
•	Feature Engineering: categorical encoding, scaling, feature selection (TotalSF, AgeSinceBuilt, AgeSinceRemod
•	Modeling Approaches:
o	Random Forest with raw 
o	Random Forest with log
•	Evaluation Metric: Root Mean Squared Error (RMSE)
 
## 📈 Results
Best submission on Kaggle achieved:
•	RMSE: 
 
## 📜 License
This project is for learning and research purposes only. Please follow the Kaggle Terms of Service.

