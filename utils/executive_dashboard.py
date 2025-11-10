"""
Executive Dashboard with advanced KPIs and business intelligence
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_executive_dashboard(metrics_dict, trends_data=None):
    """Create executive dashboard with key metrics"""
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Cost Performance', 'Time Efficiency', 'Emissions Impact',
            'Fleet Utilization', 'SLA Compliance', 'Cost Breakdown',
            'ROI Analysis', 'Customer Satisfaction', 'Operational Efficiency'
        ),
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # Cost Performance Gauge
    cost = metrics_dict.get('total_cost', 0)
    cost_target = metrics_dict.get('cost_target', cost * 1.2)
    cost_performance = (1 - cost / cost_target) * 100 if cost_target > 0 else 0
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=cost_performance,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Cost Performance"},
            delta={'reference': 50},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}
        ),
        row=1, col=1
    )
    
    # Time Efficiency Gauge
    time = metrics_dict.get('total_time', 0)
    time_target = metrics_dict.get('time_target', time * 1.2)
    time_efficiency = (1 - time / time_target) * 100 if time_target > 0 else 0
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=time_efficiency,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Time Efficiency"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"}}
        ),
        row=1, col=2
    )
    
    # Emissions Impact Gauge
    emissions = metrics_dict.get('emissions_kg', 0)
    emissions_target = metrics_dict.get('emissions_target', emissions * 1.2)
    emissions_reduction = (1 - emissions / emissions_target) * 100 if emissions_target > 0 else 0
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=emissions_reduction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Emissions Reduction"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"}}
        ),
        row=1, col=3
    )
    
    # Fleet Utilization Bar
    utilization = metrics_dict.get('weight_utilization', 0) * 100
    fig.add_trace(
        go.Bar(x=['Weight', 'Volume'], 
               y=[utilization, metrics_dict.get('volume_utilization', 0) * 100],
               name='Utilization'),
        row=2, col=1
    )
    
    # SLA Compliance Bar
    sla_compliance = metrics_dict.get('sla_compliance_rate', 0) * 100
    fig.add_trace(
        go.Bar(x=['Compliance'], y=[sla_compliance], name='SLA'),
        row=2, col=2
    )
    
    # Cost Breakdown Pie
    fuel_cost = metrics_dict.get('fuel_cost', 0)
    driver_cost = metrics_dict.get('driver_cost', 0)
    maintenance_cost = metrics_dict.get('maintenance_cost', 0)
    
    fig.add_trace(
        go.Pie(labels=['Fuel', 'Driver', 'Maintenance'],
               values=[fuel_cost, driver_cost, maintenance_cost],
               name="Cost Breakdown"),
        row=2, col=3
    )
    
    # ROI Analysis (if trends available)
    if trends_data is not None and len(trends_data) > 0:
        fig.add_trace(
            go.Scatter(x=trends_data.get('dates', []),
                      y=trends_data.get('roi', []),
                      mode='lines+markers',
                      name='ROI Trend'),
            row=3, col=1
        )
    
    # Customer Satisfaction
    csat = metrics_dict.get('customer_satisfaction', 85)
    fig.add_trace(
        go.Bar(x=['CSAT'], y=[csat], name='Customer Satisfaction'),
        row=3, col=2
    )
    
    # Operational Efficiency
    efficiency_metrics = {
        'Cost/Delivery': metrics_dict.get('cost_per_delivery', 0),
        'Time/Delivery': metrics_dict.get('time_per_delivery', 0),
        'Distance/Delivery': metrics_dict.get('distance_per_delivery', 0)
    }
    
    fig.add_trace(
        go.Bar(x=list(efficiency_metrics.keys()),
               y=list(efficiency_metrics.values()),
               name='Efficiency'),
        row=3, col=3
    )
    
    fig.update_layout(
        height=1200,
        title_text="Executive Dashboard: Logistics Performance Overview",
        showlegend=True
    )
    
    return fig


def calculate_roi(optimization_cost, savings_per_month, months=12):
    """Calculate ROI for optimization implementation"""
    total_savings = savings_per_month * months
    roi_percentage = ((total_savings - optimization_cost) / optimization_cost) * 100 if optimization_cost > 0 else 0
    payback_period = optimization_cost / savings_per_month if savings_per_month > 0 else float('inf')
    
    return {
        'total_savings': total_savings,
        'roi_percentage': roi_percentage,
        'payback_period_months': payback_period,
        'net_benefit': total_savings - optimization_cost
    }

