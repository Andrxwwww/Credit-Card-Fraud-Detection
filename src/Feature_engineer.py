import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

class TransactionFeatureEngineer:
    def __init__(self, input_path, output_path='featured_transactions.csv', chunk_size=1000000, num_threads=6):
        """
        Initialize the feature engineer.
        - input_path: Path to preprocessed CSV (e.g., 'preprocessed_transactions.csv').
        - output_path: Where to save the featured CSV.
        - chunk_size: Rows per chunk for memory efficiency.
        - num_threads: Fixed number of threads (e.g., 4) for parallel processing.
        """
        print(f"DEBUG: Initializing TransactionFeatureEngineer with input: {input_path}, chunk_size: {chunk_size}, num_threads: {num_threads}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        self.input_path = input_path
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.num_threads = num_threads
        self.scaler = None  # For scaling consistency across chunks

    def engineer_features(self):
        """
        Load preprocessed data in chunks, apply feature engineering in parallel using threads,
        combine, and save.
        Returns: DataFrame with new features.
        """
        print("DEBUG: Starting feature engineering (parallel chunk mode)...")
        start_time = time.time()
        
        # Read in chunks and collect them in a list (sequential read, parallel processing)
        chunks = pd.read_csv(self.input_path, chunksize=self.chunk_size)
        chunk_list = list(chunks)  # Load chunks into memory (fine for most cases; if too big, we can stream)
        print(f"DEBUG: Read {len(chunk_list)} chunks")
        
        # Process chunks in parallel with threads
        processed_chunks = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            print(f"DEBUG: Starting thread pool with {self.num_threads} workers")
            future_to_chunk = {executor.submit(self._engineer_chunk, chunk, idx): idx for idx, chunk in enumerate(chunk_list)}
            for future in as_completed(future_to_chunk):
                idx = future_to_chunk[future]
                try:
                    processed_chunk = future.result()
                    processed_chunks.append(processed_chunk)
                    print(f"DEBUG: Successfully processed chunk {idx+1} with {len(processed_chunk)} rows")
                except Exception as e:
                    print(f"ERROR: Failed processing chunk {idx+1}: {e}")
        
        if not processed_chunks:
            raise ValueError("No data loaded. Check your file.")
        
        # Combine chunks
        full_df = pd.concat(processed_chunks, ignore_index=True)
        
        # Save the featured data
        full_df.to_csv(self.output_path, index=False)
        total_time = time.time() - start_time
        print(f"DEBUG: Saved featured data to {self.output_path} with {len(full_df)} rows and {len(full_df.columns)} columns. Total time: {total_time:.2f} seconds")
        
        return full_df

    def _engineer_chunk(self, chunk, chunk_idx):
        """
        Apply feature engineering to a single chunk (run by threads).
        """
        # 1. Time-Based Features
        if 'transaction_hour' in chunk.columns:
            chunk['is_night_time'] = (chunk['transaction_hour'].between(0, 5)).astype(int)
        
        # 2. Aggregate Velocity Features
        if 'vel_total_amount' in chunk.columns and 'vel_num_transactions' in chunk.columns:
            chunk['avg_amount_per_tx_last_hour'] = chunk['vel_total_amount'] / chunk['vel_num_transactions'].replace(0, 1)
            chunk['high_velocity_flag'] = (chunk['vel_num_transactions'] > 50).astype(int)  # Customize threshold
        
        # 3. Bin Numeric Features
        if 'amount' in chunk.columns:
            bins = [0, 100, 1000, float('inf')]
            labels = [0, 1, 2]  # 0: low, 1: medium, 2: high
            chunk['amount_bin'] = pd.cut(chunk['amount'], bins=bins, labels=labels, include_lowest=True).astype(int)
        
        if 'distance_from_home' in chunk.columns:
            bins = [0, 10, 100, float('inf')]
            labels = [0, 1, 2]  # 0: close, 1: medium, 2: far
            chunk['distance_from_home_bin'] = pd.cut(chunk['distance_from_home'], bins=bins, labels=labels, include_lowest=True).astype(int)
        
        # 4. Interaction Features
        if 'amount' in chunk.columns and 'vel_num_transactions' in chunk.columns:
            chunk['amount_vs_velocity'] = chunk['amount'] * chunk['vel_num_transactions']
        
        if 'merchant_category' in chunk.columns and 'high_risk_merchant' in chunk.columns:
            chunk['merchant_risk_combo'] = chunk['merchant_category'] * chunk['high_risk_merchant']
        
        # 5. Normalize/Scale Numeric Features (fit on first chunk, transform others)
        scale_cols = [col for col in ['amount', 'vel_total_amount'] if col in chunk.columns]
        if scale_cols:
            if self.scaler is None and chunk_idx == 0:
                self.scaler = MinMaxScaler().fit(chunk[scale_cols])
            if self.scaler:
                chunk[scale_cols] = self.scaler.transform(chunk[scale_cols])
        
        return chunk
