import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import os
from io import StringIO

# Create directories to save figures
os.makedirs('figures', exist_ok=True)
for dataset_num in range(1, 5):
    os.makedirs(f'figures/FD00{dataset_num}', exist_ok=True)

# Connect to S3 bucket
s3_client = boto3.client('s3')
bucket_name = 'predictive-maintenance-cmapss'

# Define column names
column_names = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [
    f'sensor_{i}' for i in range(1, 22)]

def load_data_from_s3(dataset_num=1, data_type='train'):
    """Loading dataset from S3 bucket"""
    file_key = f'raw/{data_type}_FD00{dataset_num}.txt'
    
    try:
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        data = response['Body'].read().decode('utf-8')
        
        # Use appropriate delimiter with warning handling
        df = pd.read_csv(StringIO(data), sep='\s+', header=None)
        
        # Apply column names
        if df.shape[1] == len(column_names):
            df.columns = column_names
        else:
            print(f"Warning: {file_key} has {df.shape[1]} columns, expected {len(column_names)}")
            if df.shape[1] > len(column_names):
                extra_cols = [f'extra_{i}' for i in range(df.shape[1] - len(column_names))]
                df.columns = column_names + extra_cols
            else:
                df.columns = column_names[:df.shape[1]]
        
        # Convert unit_id and cycle to integer types
        df['unit_id'] = df['unit_id'].astype(int)
        df['cycle'] = df['cycle'].astype(int)
        
        return df
    except Exception as e:
        print(f"Error loading {file_key}: {e}")
        return None

def load_rul_data_from_s3(dataset_num=1):
    """Loading RUL data from S3 bucket"""
    file_key = f'raw/RUL_FD00{dataset_num}.txt'
    
    try:
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        data = response['Body'].read().decode('utf-8')
        
        # Load RUL values (typically a single column)
        rul_values = pd.read_csv(StringIO(data), header=None, names=['RUL'])
        return rul_values
    except Exception as e:
        print(f"Error loading {file_key}: {e}")
        return None

def explore_dataset(dataset_num):
    """Comprehensive exploration of a dataset"""
    print(f"\n{'='*50}")
    print(f"Exploring Dataset FD00{dataset_num}")
    print(f"{'='*50}")
    
    # Load training data
    train_df = load_data_from_s3(dataset_num=dataset_num, data_type='train')
    
    # Load test data
    test_df = load_data_from_s3(dataset_num=dataset_num, data_type='test')
    
    # Load RUL data
    rul_df = load_rul_data_from_s3(dataset_num=dataset_num)
    
    if train_df is not None:
        # Basic info for training data
        print(f"\nTraining Dataset FD00{dataset_num}:")
        print(f"Shape: {train_df.shape}")
        print(f"Number of engines: {train_df['unit_id'].nunique()}")
        
        # Calculate statistics on training data
        train_cycles_per_unit = train_df.groupby('unit_id')['cycle'].max()
        print(f"\nCycle statistics for training data:")
        print(train_cycles_per_unit.describe())
        
        # Save training data distribution visualization
        plt.figure(figsize=(10, 6))
        plt.hist(train_cycles_per_unit, bins=20)
        plt.title(f'FD00{dataset_num} - Distribution of Training Cycles per Engine')
        plt.xlabel('Number of Cycles')
        plt.ylabel('Number of Engines')
        plt.savefig(f'figures/FD00{dataset_num}/train_cycle_distribution.png')
        plt.close()
        
        # Create correlation heatmap for training data
        plt.figure(figsize=(20, 16))
        sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
        corr_matrix = train_df[sensor_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'FD00{dataset_num} - Sensor Correlation Heatmap (Training Data)')
        plt.tight_layout()
        plt.savefig(f'figures/FD00{dataset_num}/train_sensor_correlation.png')
        plt.close()
        
        # Plot sensor trends for a representative engine
        unit_id = train_df['unit_id'].min()  # First engine
        unit_data = train_df[train_df['unit_id'] == unit_id]
        
        plt.figure(figsize=(15, 10))
        # Select 4 interesting sensors - can be customized based on correlation analysis
        interesting_sensors = ['sensor_2', 'sensor_7', 'sensor_8', 'sensor_11'] 
        for i, sensor in enumerate(interesting_sensors, 1):
            plt.subplot(2, 2, i)
            plt.plot(unit_data['cycle'], unit_data[sensor])
            plt.title(f'{sensor} Trend (Engine {unit_id})')
            plt.xlabel('Cycle')
            plt.ylabel('Sensor Value')
        plt.tight_layout()
        plt.savefig(f'figures/FD00{dataset_num}/train_sensor_trends.png')
        plt.close()
        
        # Operating conditions analysis
        plt.figure(figsize=(15, 5))
        for i, setting in enumerate(['op_setting_1', 'op_setting_2', 'op_setting_3'], 1):
            plt.subplot(1, 3, i)
            plt.hist(train_df[setting], bins=30)
            plt.title(f'Distribution of {setting}')
        plt.tight_layout()
        plt.savefig(f'figures/FD00{dataset_num}/operating_conditions.png')
        plt.close()
    
    if test_df is not None:
        # Basic info for test data
        print(f"\nTest Dataset FD00{dataset_num}:")
        print(f"Shape: {test_df.shape}")
        print(f"Number of engines: {test_df['unit_id'].nunique()}")
        
        # Calculate statistics on test data
        test_cycles_per_unit = test_df.groupby('unit_id')['cycle'].max()
        print(f"\nCycle statistics for test data:")
        print(test_cycles_per_unit.describe())
        
        # Save test data distribution visualization
        plt.figure(figsize=(10, 6))
        plt.hist(test_cycles_per_unit, bins=20)
        plt.title(f'FD00{dataset_num} - Distribution of Test Cycles per Engine')
        plt.xlabel('Number of Cycles')
        plt.ylabel('Number of Engines')
        plt.savefig(f'figures/FD00{dataset_num}/test_cycle_distribution.png')
        plt.close()
    
    if rul_df is not None:
        # RUL statistics
        print(f"\nRUL Data Statistics for FD00{dataset_num}:")
        print(rul_df.describe())
        
        # Plot RUL distribution
        plt.figure(figsize=(10, 6))
        plt.hist(rul_df['RUL'], bins=20)
        plt.title(f'FD00{dataset_num} - Remaining Useful Life Distribution')
        plt.xlabel('RUL (Cycles)')
        plt.ylabel('Number of Engines')
        plt.savefig(f'figures/FD00{dataset_num}/rul_distribution.png')
        plt.close()
        
        # If test data is available, combine with RUL
        if test_df is not None and len(rul_df) == test_df['unit_id'].nunique():
            # Create a mapping of last cycle data with RUL
            last_cycles = test_df.groupby('unit_id')['cycle'].max()
            last_cycle_data = []
            
            for unit_id in range(1, len(rul_df) + 1):
                if unit_id in test_df['unit_id'].values:
                    last_cycle = last_cycles[unit_id]
                    unit_last_data = test_df[(test_df['unit_id'] == unit_id) & 
                                            (test_df['cycle'] == last_cycle)]
                    
                    # Add RUL value
                    if not unit_last_data.empty:
                        row = unit_last_data.iloc[0].to_dict()
                        row['RUL'] = rul_df.iloc[unit_id-1]['RUL']
                        last_cycle_data.append(row)
            
            if last_cycle_data:
                last_cycle_df = pd.DataFrame(last_cycle_data)
                
                # Plot relationship between final sensor readings and RUL
                for sensor in ['sensor_2', 'sensor_7', 'sensor_8', 'sensor_11']:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(last_cycle_df[sensor], last_cycle_df['RUL'])
                    plt.title(f'FD00{dataset_num} - Relationship between {sensor} and RUL')
                    plt.xlabel(sensor)
                    plt.ylabel('RUL (Cycles)')
                    plt.savefig(f'figures/FD00{dataset_num}/{sensor}_vs_rul.png')
                    plt.close()

# Explore all datasets
for dataset_num in range(1, 5):
    explore_dataset(dataset_num)

print("\nExploration complete. Check the 'figures' directory for visualizations.")