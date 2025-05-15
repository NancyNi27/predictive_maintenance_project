import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# AWS SDK, used to access S3 
import boto3
import os
from io import StringIO

# create figures to save pictures
os.makedirs('figures', exist_ok=True)

# connect S3 bucket
s3_client = boto3.client('s3')
bucket_name = 'predictive-maintenance-cmapss'

# define column name with equipment number, cycle etc.
column_names = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [
    f'sensor_{i}' for i in range(1, 22)]

def load_data_from_s3(dataset_num=1, data_type='train'):
    """Loading dataset from S3 bucket"""
    file_key = f'raw/{data_type}_FD00{dataset_num}.txt'
    
    try:
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        data = response['Body'].read().decode('utf-8')
        
        # Use more explicit delimiter option to handle consecutive spaces
        df = pd.read_csv(StringIO(data), delim_whitespace=True, header=None)
        
        # Apply correct column names
        column_names = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [
            f'sensor_{i}' for i in range(1, 22)]
        
        # Ensure column count matches
        if df.shape[1] == len(column_names):
            df.columns = column_names
        else:
            # Handle extra columns if any
            print(f"Warning: Data has {df.shape[1]} columns, but column names defined only have {len(column_names)} columns")
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

# load training data
train_df = load_data_from_s3(dataset_num=1, data_type='train')

if train_df is not None:
    # basic infor
    print("Dataset Information:")
    print(f"Shape: {train_df.shape}")
    print("\n First 5 lines:")
    print(train_df.head())
    
    # basic statistic
    print("\n Basic Statistic:")
    print(train_df.describe())
    
    # Check NaN
    print("\n NAN number:")
    print(train_df.isnull().sum())
    
    #check the number of different devices
    unique_units = train_df['unit_id'].nunique()
    print(f"\n There are {unique_units} devices in total")
    
    # Number of Cycles per device
    cycles_per_unit = train_df.groupby('unit_id')['cycle'].max()
    print("\n Cycle count for device:")
    print(cycles_per_unit.describe())
    
    # Show the number distribution of cycles for some devices
    plt.figure(figsize=(10, 6))
    plt.hist(cycles_per_unit, bins=20)
    plt.title('Distribution of equipment cycle count')
    plt.xlabel('Number of Cycles')
    plt.ylabel('Amount of Equipments')
    plt.savefig('figures/cycle_distribution.png')
    
    # Correlation heat map of sensor data
    plt.figure(figsize=(20, 16))
    available_sensors = [col for col in train_df.columns if col.startswith('sensor_')]
    if available_sensors:
        corr_matrix = train_df[available_sensors].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('The Correlation of Sensor')
        plt.tight_layout()
        plt.savefig('figures/sensor_correlation.png')
    else:
        print("Warning: The sensor column was not found!")
    
    # Change trends of several key sensors on a device
    unit_id = train_df['unit_id'].min()  
    plt.figure(figsize=(15, 10))

    key_sensors = [s for s in ['sensor_2', 'sensor_7', 'sensor_8', 'sensor_11'] 
                  if s in train_df.columns]
    if key_sensors:
        for i, sensor in enumerate(key_sensors, 1):
            plt.subplot(2, 2, i)
            unit_data = train_df[train_df['unit_id'] == unit_id]
            plt.plot(unit_data['cycle'], unit_data[sensor])
            plt.title(f'{sensor} Change over time (Equipment {unit_id})')
            plt.xlabel('Cycle')
            plt.ylabel('Sensor value')
        plt.tight_layout()
        plt.savefig('figures/sensor_trends.png')
    else:
        print("Warning: the key sensor column to be drawn cannot be found！")
    
    print("Analysis is completed. Please check the visualization in the figures folder.")
else:
    print("Loading faliure, please check S3 configure and loading path.")