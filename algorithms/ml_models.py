"""
Machine Learning models for delivery time prediction and SLA risk assessment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report

# Try to import xgboost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    # Catch any error during xgboost import (including library loading errors)
    XGBOOST_AVAILABLE = False
    xgb = None
    # Silently fail - we'll use RandomForest as fallback


class DeliveryTimePredictor:
    """Predict delivery time using ML models"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_importance_ = None
    
    def prepare_features(self, orders_df, delivery_locations, distance_matrix, traffic_data=None, weather_data=None):
        """Prepare features for prediction"""
        features = []
        
        for idx, order in orders_df.iterrows():
            delivery_loc = delivery_locations[delivery_locations['delivery_id'] == order['delivery_id']]
            if len(delivery_loc) == 0:
                continue
            
            delivery_loc = delivery_loc.iloc[0]
            
            # Basic features
            feature_dict = {
                'weight_kg': order['weight_kg'],
                'volume_m3': order['volume_m3'],
                'priority_encoded': {'standard': 0, 'priority': 1, 'express': 2}.get(order['priority'], 0),
                'time_window_start': order['time_window_start'],
                'time_window_end': order['time_window_end'],
                'window_duration': order['time_window_end'] - order['time_window_start'],
                'latitude': delivery_loc['latitude'],
                'longitude': delivery_loc['longitude']
            }
            
            # Traffic features
            if traffic_data is not None and len(traffic_data) > 0:
                hour = order['order_timestamp'].hour if 'order_timestamp' in order else 12
                traffic_row = traffic_data[traffic_data['hour'] == hour]
                if len(traffic_row) > 0:
                    feature_dict['traffic_congestion'] = traffic_row.iloc[0]['congestion_factor']
                    feature_dict['traffic_speed_multiplier'] = traffic_row.iloc[0]['speed_multiplier']
                else:
                    feature_dict['traffic_congestion'] = 0.3
                    feature_dict['traffic_speed_multiplier'] = 0.8
            else:
                feature_dict['traffic_congestion'] = 0.3
                feature_dict['traffic_speed_multiplier'] = 0.8
            
            # Weather features
            if weather_data is not None and len(weather_data) > 0:
                weather_row = weather_data.iloc[-1]  # Use most recent
                feature_dict['precipitation'] = weather_row['precipitation_mm']
                feature_dict['temperature'] = weather_row['temperature_c']
                feature_dict['wind_speed'] = weather_row['wind_speed_kmh']
            else:
                feature_dict['precipitation'] = 0
                feature_dict['temperature'] = 20
                feature_dict['wind_speed'] = 10
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def train(self, X, y):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6)
        else:
            # Fallback to RandomForest if XGBoost not available
            if self.model_type == 'xgboost':
                print("Warning: XGBoost not available, using RandomForest instead")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return {'mae': mae, 'rmse': rmse}
    
    def predict(self, X):
        """Predict delivery times"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)


class SLARiskPredictor:
    """Predict SLA violation risk using classification"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_importance_ = None
    
    def prepare_features(self, orders_df, delivery_locations, current_time, estimated_delivery_time):
        """Prepare features for SLA risk prediction"""
        features = []
        
        for idx, order in orders_df.iterrows():
            feature_dict = {
                'priority_encoded': {'standard': 0, 'priority': 1, 'express': 2}.get(order['priority'], 0),
                'sla_hours': order['sla_hours'],
                'time_until_window_start': max(0, order['time_window_start'] - current_time),
                'time_until_window_end': max(0, order['time_window_end'] - current_time),
                'estimated_delivery_time': estimated_delivery_time,
                'time_buffer': order['time_window_end'] - estimated_delivery_time - current_time,
                'weight_kg': order['weight_kg'],
                'volume_m3': order['volume_m3']
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def train(self, X, y):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return {'accuracy': accuracy}
    
    def predict(self, X):
        """Predict SLA violation risk"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict SLA violation probability"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)

