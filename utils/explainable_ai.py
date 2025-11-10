"""
Explainable AI utilities using SHAP and LIME
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def calculate_shap_values(model, X, X_sample=None, max_samples=100):
    """Calculate SHAP values for model explainability"""
    if not SHAP_AVAILABLE:
        return None
    
    if X_sample is None:
        # Sample for faster computation
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X
    
    try:
        # Use TreeExplainer for tree-based models
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_sample)
        
        shap_values = explainer(X_sample)
        return shap_values, X_sample
    except:
        return None, X_sample


def plot_shap_summary(shap_values, feature_names, max_features=20):
    """Plot SHAP summary plot"""
    if shap_values is None:
        return None
    
    try:
        # Get mean absolute SHAP values
        if hasattr(shap_values, 'values'):
            shap_vals = shap_values.values
        else:
            shap_vals = shap_values
        
        if len(shap_vals.shape) > 2:
            shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] > 1 else shap_vals[:, :, 0]
        
        mean_shap = np.abs(shap_vals).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_shap)[-max_features:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_shap = mean_shap[top_indices]
        
        fig = go.Figure(data=go.Bar(
            x=top_shap,
            y=top_features,
            orientation='h',
            marker=dict(color=top_shap, colorscale='Reds')
        ))
        
        fig.update_layout(
            title='SHAP Feature Importance',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Feature',
            height=600
        )
        
        return fig
    except:
        return None


def plot_shap_waterfall(shap_values, instance_idx=0, max_features=10):
    """Plot SHAP waterfall plot for a single instance"""
    if shap_values is None:
        return None
    
    try:
        if hasattr(shap_values, 'values'):
            values = shap_values.values[instance_idx]
            base_value = shap_values.base_values[instance_idx] if hasattr(shap_values, 'base_values') else 0
        else:
            return None
        
        if len(values.shape) > 1:
            values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
        
        # Get top contributing features
        abs_values = np.abs(values)
        top_indices = np.argsort(abs_values)[-max_features:][::-1]
        
        top_values = values[top_indices]
        top_features = [f'Feature {i}' for i in top_indices]
        
        # Create waterfall
        cumulative = base_value
        x_pos = []
        y_pos = []
        text = []
        
        for i, val in enumerate(top_values):
            x_pos.append(cumulative)
            y_pos.append(i)
            text.append(f'{val:.3f}')
            cumulative += val
        
        fig = go.Figure(data=go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='lines+markers+text',
            text=text,
            textposition='middle right',
            name='SHAP Values'
        ))
        
        fig.update_layout(
            title=f'SHAP Waterfall Plot (Instance {instance_idx})',
            xaxis_title='SHAP Value',
            yaxis_title='Feature',
            height=500
        )
        
        return fig
    except:
        return None

