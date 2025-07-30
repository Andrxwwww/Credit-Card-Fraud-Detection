from src.Data_preprocess import TransactionDataLoader  # Assuming your loader file
from src.Feature_engineer import TransactionFeatureEngineer

def main():
    # Step 1: Load and preprocess (if not already done)
    file_path = 'data/synthetic_fraud_data.csv'  # Your original raw CSV
    preprocessed_path = 'preprocessed_transactions.csv'
    output_path = 'featured_transactions.csv'
    chunk_size = 1000000
    n_threads=6
    
    loader = TransactionDataLoader(file_path=file_path, chunk_size=chunk_size , n_threads=n_threads)
    preprocessed_data = loader.load_and_preprocess()
    preprocessed_data.to_csv(preprocessed_path, index=False)
    print(f"\nDEBUG: Preprocessing complete. Saved to {preprocessed_path}")
    
    # Step 2: Feature engineering
    engineer = TransactionFeatureEngineer(input_path=preprocessed_path , output_path=output_path , num_threads=n_threads)
    featured_data = engineer.engineer_features()
    
    # Display first few rows for verification
    print("\nDEBUG: First 5 rows of featured data:")
    print(featured_data.head())

if __name__ == "__main__":
    main()
