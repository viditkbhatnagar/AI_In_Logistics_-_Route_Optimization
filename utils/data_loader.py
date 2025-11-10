"""
Data loading and preprocessing utilities
"""

import pandas as pd
import os
import json
import glob


def load_dataset(scenario_name, data_dir=None):
    """Load a complete dataset from the appropriate directory"""
    # If data_dir not specified, search in both locations
    if data_dir is None:
        # Try pre_generated first, then datasets
        for check_dir in ['data/pre_generated', 'data/datasets']:
            base_path = f'{check_dir}/{scenario_name}'
            metadata_path = f'{base_path}_metadata.json'
            if os.path.exists(metadata_path):
                data_dir = check_dir
                break
        else:
            # Default to pre_generated if not found
            data_dir = 'data/pre_generated'
    
    base_path = f'{data_dir}/{scenario_name}'
    
    dataset = {
        'scenario_name': scenario_name,
        'delivery_locations': pd.read_csv(f'{base_path}_delivery_locations.csv'),
        'orders': pd.read_csv(f'{base_path}_orders.csv'),
        'vehicles': pd.read_csv(f'{base_path}_vehicles.csv'),
        'depots': pd.read_csv(f'{base_path}_depots.csv'),
        'traffic': pd.read_csv(f'{base_path}_traffic.csv'),
        'weather': pd.read_csv(f'{base_path}_weather.csv'),
        'demand_forecast': pd.read_csv(f'{base_path}_demand_forecast.csv')
    }
    
    # Load metadata if available
    metadata_path = f'{base_path}_metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            dataset['metadata'] = json.load(f)
    else:
        dataset['metadata'] = {}
    
    # Convert datetime columns
    if 'order_timestamp' in dataset['orders'].columns:
        dataset['orders']['order_timestamp'] = pd.to_datetime(dataset['orders']['order_timestamp'])
    
    if 'datetime' in dataset['traffic'].columns:
        dataset['traffic']['datetime'] = pd.to_datetime(dataset['traffic']['datetime'])
    
    if 'date' in dataset['weather'].columns:
        dataset['weather']['date'] = pd.to_datetime(dataset['weather']['date'])
    
    if 'date' in dataset['demand_forecast'].columns:
        dataset['demand_forecast']['date'] = pd.to_datetime(dataset['demand_forecast']['date'])
    
    return dataset


def list_available_datasets(data_dir='data/pre_generated'):
    """List all available datasets from multiple directories"""
    # Check both pre_generated and datasets directories
    directories_to_check = ['data/pre_generated', 'data/datasets']
    
    all_datasets = {}
    
    for check_dir in directories_to_check:
        if os.path.exists(check_dir):
            pattern = f'{check_dir}/*_metadata.json'
            metadata_files = glob.glob(pattern)
            
            for metadata_file in metadata_files:
                scenario_name = os.path.basename(metadata_file).replace('_metadata.json', '')
                # Only add if not already found (prefer pre_generated over datasets)
                if scenario_name not in all_datasets:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        all_datasets[scenario_name] = {
                            'name': scenario_name,
                            'metadata': metadata,
                            'path': check_dir
                        }
                    except:
                        all_datasets[scenario_name] = {
                            'name': scenario_name,
                            'metadata': {},
                            'path': check_dir
                        }
    
    # Return as list sorted by name
    return sorted(all_datasets.values(), key=lambda x: x['name'])


def preprocess_orders(orders_df):
    """Preprocess orders data"""
    orders_df = orders_df.copy()
    
    # Ensure priority is categorical
    if 'priority' in orders_df.columns:
        orders_df['priority'] = orders_df['priority'].astype('category')
    
    return orders_df


def preprocess_locations(locations_df):
    """Preprocess delivery locations data"""
    locations_df = locations_df.copy()
    
    # Ensure coordinates are numeric
    if 'latitude' in locations_df.columns:
        locations_df['latitude'] = pd.to_numeric(locations_df['latitude'])
    if 'longitude' in locations_df.columns:
        locations_df['longitude'] = pd.to_numeric(locations_df['longitude'])
    
    return locations_df

