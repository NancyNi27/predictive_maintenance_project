import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# AWS SDK, used to access S3 
import boto3
import os
from io import StringIO

# create contents, if not exists
os.makedirs('figures', exist_ok=True)

# S3客户端
s3_client = boto3.client('s3')
bucket_name = 'predictive-maintenance-cmapss'

# 定义列名
column_names = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [
    f'sensor_{i}' for i in range(1, 22)]

def load_data_from_s3(dataset_num=1, data_type='train'):
    """加载S3中的数据集"""
    file_key = f'raw/{data_type}_FD00{dataset_num}.txt'
    
    try:
        # 从S3下载文件
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        data = response['Body'].read().decode('utf-8')
        
        # 加载数据集
        df = pd.read_csv(StringIO(data), sep=' ', header=None, names=column_names)
        # 移除末尾的NaN列
        df = df.drop(df.columns[[26, 27]], axis=1, errors='ignore')
        return df
    except Exception as e:
        print(f"Error loading {file_key}: {e}")
        return None

# 加载训练数据
train_df = load_data_from_s3(dataset_num=1, data_type='train')

if train_df is not None:
    # 显示基本信息
    print("数据集信息:")
    print(f"形状: {train_df.shape}")
    print("\n前5行:")
    print(train_df.head())
    
    # 基本统计
    print("\n基本统计:")
    print(train_df.describe())
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(train_df.isnull().sum())
    
    # 查看不同设备数量
    unique_units = train_df['unit_id'].nunique()
    print(f"\n总共有 {unique_units} 个设备")
    
    # 每个设备的循环数
    cycles_per_unit = train_df.groupby('unit_id')['cycle'].max()
    print("\n每个设备的循环数统计:")
    print(cycles_per_unit.describe())
    
    # 绘制一些设备的循环数分布
    plt.figure(figsize=(10, 6))
    plt.hist(cycles_per_unit, bins=20)
    plt.title('设备循环数分布')
    plt.xlabel('循环数')
    plt.ylabel('设备数')
    plt.savefig('figures/cycle_distribution.png')
    
    # 绘制相关性热图
    plt.figure(figsize=(20, 16))
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    corr_matrix = train_df[sensor_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('传感器数据相关性')
    plt.tight_layout()
    plt.savefig('figures/sensor_correlation.png')
    
    # 为一个设备绘制关键传感器趋势
    unit_id = 1
    plt.figure(figsize=(15, 10))
    for i, sensor in enumerate(['sensor_2', 'sensor_7', 'sensor_8', 'sensor_11'], 1):
        plt.subplot(2, 2, i)
        unit_data = train_df[train_df['unit_id'] == unit_id]
        plt.plot(unit_data['cycle'], unit_data[sensor])
        plt.title(f'{sensor} 随时间变化 (设备 {unit_id})')
        plt.xlabel('循环')
        plt.ylabel('传感器值')
    plt.tight_layout()
    plt.savefig('figures/sensor_trends.png')
    
    print("分析完成。请查看figures文件夹中的可视化结果。")
else:
    print("数据加载失败。请检查S3配置和文件路径。")