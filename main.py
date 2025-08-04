import os
from src.Data_preprocess import TransactionDataLoader
from src.Feature_engineer import TransactionFeatureEngineer
from src.Modeling import train_and_save_models  # Import the new modeling module

def main():
    # File paths and params
    raw_file_path = 'data/synthetic_fraud_data.csv'
    preprocessed_path = 'data/preprocessed_transactions.csv'
    featured_path = 'data/featured_transactions.csv'
    chunk_size = 1000000
    n_threads = 6
    models_dir = 'models'

    # Step 1: Preprocessing
    print("\n========== DATA PROCESSING ==========")
    loader = TransactionDataLoader(file_path=raw_file_path, chunk_size=chunk_size, n_threads=n_threads)
    preprocessed_df = loader.load_and_preprocess()
    preprocessed_df.to_csv(preprocessed_path, index=False)
    print(f"\nDEBUG: Preprocessing complete. Saved to {preprocessed_path}")

    # Step 2: Feature engineering
    print("\n========== FEATURE ENGINEERING ==========")
    engineer = TransactionFeatureEngineer(
        input_path=preprocessed_path,
        output_path=featured_path,
        chunk_size=chunk_size,
        num_threads=n_threads
    )
    featured_df = engineer.engineer_features()
    print("\nDEBUG: First 5 rows of featured data:")
    print(featured_df.head())

    # Step 3: Modeling and model saving
    print("\n========== MODELING ==========")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    train_and_save_models(featured_path, models_dir)

if __name__ == "__main__":
    main()
