# Credit Card Transaction Data Pipeline

This project provides a Python-based pipeline for processing large credit card transaction datasets (e.g., 2GB CSV files). It includes three main components:
- **Data Preprocessing**: Loads the raw data in chunks, applies cleaning and transformations, and handles large files efficiently with multiprocessing.
- **Feature Engineering**: Takes the preprocessed data and generates new features for analysis or modeling (e.g., fraud detection), using threading for parallel processing.
- **Modeling**: Trains and evaluates multiple machine learning models on the featured data, saving them for future use.

The pipeline is designed for scalability, memory efficiency, and simplicity. It processes data like transaction IDs, timestamps, amounts, merchant details, and fraud labels, transforming them into a clean, feature-rich format.

## Dependencies
Dataset used: https://www.kaggle.com/datasets/ismetsemedov/transactions

> pip install -r req.txt

## Data Preprocessing Details
The `Data_preprocess.py` script loads a large CSV (e.g., with columns like transaction_id, timestamp, amount, merchant_category, velocity_last_hour, is_fraud) in chunks to avoid memory issues. Each chunk is split into 6 sub-chunks and processed in parallel using multiprocessing for speed. Here's what happens transformation by transformation, explained simply (like preparing raw ingredients into a basic meal):

1. **Drop Unwanted Columns**: Remove identifiers and non-useful data like 'transaction_id', 'customer_id', 'card_number', 'merchant', 'device_fingerprint', and 'ip_address'. Why? These are unique labels that don't help analysis and bloat the file. Example: A row with "transaction_id: TX_a0ad2a2a" loses that column.

2. **Handle Timestamp**: Convert 'timestamp' (e.g., "2024-09-30 00:00:01+00:00") to datetime, extract 'transaction_hour' (0-23) and 'weekend_transaction' (1 if Saturday/Sunday, 0 otherwise). Invalid dates get defaults (-1 for hour, 0 for weekend). Drop the original timestamp. Why? Makes time data usable for patterns (e.g., fraud at night). Example: Midnight timestamp becomes transaction_hour=0, weekend_transaction=0.

3. **Parse Velocity Column**: The 'velocity_last_hour' string (e.g., "{'num_transactions': 1197, ...}") is parsed into new columns like 'vel_num_transactions', 'vel_total_amount'. Uses fast `ast.literal_eval` with error handling (defaults to 0). Drop the original. Why? Turns text into numbers for analysis (e.g., high velocity might indicate fraud). Example: Extracts 1197 as 'vel_num_transactions'.

4. **Convert Booleans**: Change True/False in columns like 'card_present', 'weekend_transaction', 'high_risk_merchant', 'is_fraud' to 1/0. Why? Numbers are better for math/models. Example: "is_fraud: True" becomes 1.

5. **Encode Categoricals**: Convert text columns (e.g., 'merchant_category' from "Restaurant" to a number like 5) using LabelEncoder. Fits on the first chunk for consistency, then applies to all. Why? Text is inefficient; numbers speed up processing. Example: "currency: GBP" becomes 4.

6. **Fill Missing Values**: Replace blanks (NaN) with 0. Why? Prevents errors in later steps. Example: Missing 'amount' becomes 0.

Output: A preprocessed CSV with cleaned, numeric data—smaller and ready for features.

## Feature Engineering Details
The `Feature_engineer.py` script takes the preprocessed CSV and adds new features in chunks with threading for speed. It processes in parallel but fits scalers sequentially for consistency. Here's each transformation explained simply (like adding spices to the prepped meal):

1. **Time-Based Features**: Add 'is_night_time' (1 if 'transaction_hour' is 0-5, else 0). Why? Flags odd-hour transactions for fraud patterns. Example: Hour 3 becomes 1.

2. **Aggregate Velocity Features**: Add 'avg_amount_per_tx_last_hour' ('vel_total_amount' / 'vel_num_transactions', avoiding divide-by-zero) and 'high_velocity_flag' (1 if 'vel_num_transactions' > 50, else 0). Why? Highlights spending speed (e.g., rapid transactions as risk). Example: Total 10,000 with 100 tx becomes avg=100.

3. **Bin Numeric Features**: Add 'amount_bin' (0: 0-100 low, 1: 100-1000 medium, 2: 1000+ high) and 'distance_from_home_bin' (0: 0-10 close, 1: 10-100 medium, 2: 100+ far). Why? Groups values for easier patterns (e.g., high amounts linked to fraud). Example: Amount 294.87 becomes 1 (medium).

4. **Interaction Features**: Add 'amount_vs_velocity' ('amount' * 'vel_num_transactions') and 'merchant_risk_combo' ('merchant_category' * 'high_risk_merchant'). Why? Captures combined risks (e.g., big amount with high velocity). Example: Amount 500 * 10 tx = 5000.

5. **Normalize/Scale Numeric Features**: Scale 'amount' and 'vel_total_amount' to 0-1 range using MinMaxScaler (fitted on first chunk). Why? Makes numbers comparable for models. Example: Amount 294.87 in a 0-1,000,000 range becomes ~0.000295.

Output: A featured CSV with the original preprocessed data plus these new columns—ready for modeling or analysis.

## Modeling Details
The `Modeling.py` script trains and evaluates multiple machine learning models on the featured data for fraud detection. It loads the featured CSV, splits into train/test sets, trains each model, computes performance metrics, and saves the models for later use. Here's each step explained simply (like cooking the meal with the prepped ingredients):

1. **Load Featured Data**: Reads the 'featured_transactions.csv' into a DataFrame. Why? To access the cleaned and engineered features for training. Example: Loads rows with new columns like 'is_night_time' and 'amount_bin'.

2. **Split Data**: Separates features (X) from the label ('is_fraud' as y), then splits into train (70%) and test (30%) sets with stratification to maintain fraud/non-fraud balance. Why? To train on one part and evaluate fairly on unseen data. Example: If you have 10M rows, ~7M for training, ~3M for testing.

3. **Train Models**: Fits four models (Random Forest, LightGBM, XGBoost) on the train set, handling imbalance with class weights. Why? To compare which works best for numeric features in fraud detection. Example: Random Forest builds 100 decision trees to learn patterns.

4. **Evaluate Models**: Predicts on the test set and computes metrics like accuracy, F1-score, and ROC-AUC. Why? To measure how well each model detects fraud (e.g., high AUC means good at ranking risky transactions). Example: Outputs a report showing 97% accuracy for Random Forest.

5. **Save Models**: Stores each trained model as a .joblib file in the 'models/' directory. Why? For reuse in production or further testing without retraining. Example: Saves 'fraud_random_forest.joblib' for quick loading later.

Output: Trained model files in 'models/' and printed evaluation reports—ready for deployment or further tuning.
