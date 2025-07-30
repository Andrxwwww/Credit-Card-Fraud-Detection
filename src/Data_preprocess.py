import pandas as pd
from sklearn.preprocessing import LabelEncoder  # For categorical encoding
import ast  # For faster dict parsing
import os  # For file checks
import time  # For timing debug
from concurrent.futures import ProcessPoolExecutor, as_completed  # For multiprocessing
import numpy as np  # For splitting chunks
import multiprocessing as mp  # For shared manager

# Helper function for sub-chunk preprocessing (must be top-level for pickling in multiprocessing)
def preprocess_subchunk(subchunk, label_encoders_dict, cat_cols):
    """
    Preprocessing logic for a sub-chunk (run by processes).
    Uses passed label_encoders_dict for consistency.
    """
    # Drop columns
    drop_cols = ['transaction_id', 'customer_id', 'card_number', 'merchant', 
                 'device_fingerprint', 'ip_address']
    subchunk = subchunk.drop(columns=[col for col in drop_cols if col in subchunk.columns])
    
    # Handle timestamp (vectorized)
    if 'timestamp' in subchunk.columns:
        subchunk['timestamp'] = pd.to_datetime(subchunk['timestamp'], errors='coerce')
        subchunk['transaction_hour'] = subchunk['timestamp'].dt.hour.fillna(-1).astype(int)
        subchunk['weekend_transaction'] = (subchunk['timestamp'].dt.weekday >= 5).fillna(0).astype(int)
        subchunk = subchunk.drop(columns=['timestamp'])
    
    # Parse velocity (vectorized with ast.literal_eval)
    if 'velocity_last_hour' in subchunk.columns:
        subchunk['velocity_last_hour'] = subchunk['velocity_last_hour'].str.replace("'", "\"")
        def safe_eval(x):
            try:
                return ast.literal_eval(x)
            except:
                return {'num_transactions': 0, 'total_amount': 0.0, 'unique_merchants': 0, 
                        'unique_countries': 0, 'max_single_amount': 0.0}
        
        velocity_dicts = subchunk['velocity_last_hour'].apply(safe_eval)
        velocity_df = pd.DataFrame(velocity_dicts.tolist(), index=subchunk.index)
        velocity_df.columns = ['vel_' + col for col in velocity_df.columns]
        subchunk = pd.concat([subchunk, velocity_df], axis=1)
        subchunk = subchunk.drop(columns=['velocity_last_hour'])
    
    # Handle booleans (vectorized)
    bool_cols = ['card_present', 'weekend_transaction', 'high_risk_merchant', 'is_fraud']
    for col in bool_cols:
        if col in subchunk.columns:
            subchunk[col] = subchunk[col].astype(int)
    
    # Handle categoricals (use passed encoders)
    for col in cat_cols:
        if col in subchunk.columns and col in label_encoders_dict:
            subchunk[col] = label_encoders_dict[col].transform(subchunk[col].astype(str))
    
    # Fill missing values (vectorized)
    subchunk = subchunk.fillna(0)
    
    return subchunk

class TransactionDataLoader:
    def __init__(self, file_path, chunk_size, n_threads):
        print(f"DEBUG: Initializing TransactionDataLoader with file: {file_path}, chunk_size: {chunk_size}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.n_threads = n_threads
        self.label_encoders = {}  # Fitted encoders
        self.cat_cols = ['merchant_category', 'merchant_type', 'currency', 'country', 'city', 
                         'city_size', 'card_type', 'device', 'channel']  # Defined here for passing to processes

    def load_and_preprocess(self):
        print("DEBUG: Starting load_and_preprocess (with per-chunk multiprocessing)...")
        start_time = time.time()
        
        chunks = pd.read_csv(self.file_path, chunksize=self.chunk_size, parse_dates=['timestamp'])
        
        processed_chunks = []
        total_rows_processed = 0
        for idx, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            print(f"DEBUG: Loaded chunk {idx+1} with {len(chunk)} rows")
            
            # Fit label encoders on first chunk only (sequential to avoid race conditions)
            if idx == 0:
                for col in self.cat_cols:
                    if col in chunk.columns:
                        self.label_encoders[col] = LabelEncoder().fit(chunk[col].astype(str))
                print("DEBUG: Fitted label encoders on first chunk")
            
            # Now preprocess the chunk with multiprocessing
            processed_chunk = self._preprocess_chunk(chunk)
            processed_chunks.append(processed_chunk)
            total_rows_processed += len(processed_chunk)
            
            chunk_time = time.time() - chunk_start_time
            print(f"DEBUG: Processed chunk {idx+1} in {chunk_time:.2f} seconds (total rows so far: {total_rows_processed})")
        
        if not processed_chunks:
            raise ValueError("No data loaded. Check your file.")
        full_df = pd.concat(processed_chunks, ignore_index=True)
        
        total_time = time.time() - start_time
        print(f"DEBUG: Completed processing all chunks ({len(full_df)} rows total) in {total_time:.2f} seconds")
        return full_df

    def _preprocess_chunk(self, chunk):
        # Split chunk into 6 sub-chunks
        if len(chunk) < 6:
            print("DEBUG: Chunk too small for splitting; processing sequentially")
            return preprocess_subchunk(chunk, self.label_encoders, self.cat_cols)
        
        subchunk_size = len(chunk) // 6
        subchunks = [chunk.iloc[i:i + subchunk_size] for i in range(0, len(chunk), subchunk_size)]
        if len(subchunks) > 6:
            subchunks[-2] = pd.concat([subchunks[-2], subchunks[-1]])
            subchunks = subchunks[:-1]
        
        # Process sub-chunks in parallel with 6 processes
        processed_subchunks = []
        with ProcessPoolExecutor(max_workers= self.n_threads) as executor:
            future_to_sub = {executor.submit(preprocess_subchunk, sub, self.label_encoders, self.cat_cols): i for i, sub in enumerate(subchunks)}
            for future in as_completed(future_to_sub):
                try:
                    processed_sub = future.result()
                    processed_subchunks.append(processed_sub)
                except Exception as e:
                    print(f"ERROR: Failed processing sub-chunk: {e}")
        
        # Combine and return
        return pd.concat(processed_subchunks, ignore_index=True)
