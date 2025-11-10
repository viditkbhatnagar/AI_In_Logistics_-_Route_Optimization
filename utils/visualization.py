"""
Visualization utilities for maps, routes, charts, and graphs
"""

import folium
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import streamlit as st


def create_route_map(delivery_locations, depots, routes_dict, vehicles_df, center_lat=None, center_lon=None):
    """
    Create an interactive map showing routes
    
    Parameters:
    - delivery_locations: DataFrame with delivery locations
    - depots: DataFrame with depot locations
    - routes_dict: dict mapping route_name (or vehicle_id) to list of delivery indices
    - vehicles_df: DataFrame with vehicle information
    - center_lat, center_lon: Map center coordinates
    """
    # Determine map center
    if center_lat is None:
        center_lat = delivery_locations['latitude'].mean()
    if center_lon is None:
        center_lon = delivery_locations['longitude'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Color palette for routes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple',
              'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black']
    
    # Add depots
    for idx, depot in depots.iterrows():
        folium.Marker(
            [depot['latitude'], depot['longitude']],
            popup=f"Depot: {depot['name']}",
            icon=folium.Icon(color='black', icon='warehouse', prefix='fa'),
            tooltip=f"Depot: {depot['name']}"
        ).add_to(m)
    
    # Add routes
    color_idx = 0
    for route_name, route in routes_dict.items():
        if len(route) == 0:
            continue
        
        # Try to find vehicle by route_name (vehicle_id), or use first vehicle as default
        vehicle = None
        if route_name in vehicles_df['vehicle_id'].values:
            vehicle = vehicles_df[vehicles_df['vehicle_id'] == route_name].iloc[0]
        elif len(vehicles_df) > 0:
            # Use first vehicle as default
            vehicle = vehicles_df.iloc[0]
        else:
            # No vehicle info, use first depot
            if len(depots) > 0:
                depot = depots.iloc[0]
                depot_coords = [depot['latitude'], depot['longitude']]
            else:
                continue
        
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        # Get route coordinates
        route_coords = []
        
        # Get depot coordinates
        if vehicle is not None:
            depot_coords = [vehicle['depot_latitude'], vehicle['depot_longitude']]
        else:
            depot_coords = [depot['latitude'], depot['longitude']]
        
        # Start from depot
        route_coords.append(depot_coords)
        
        # Add delivery locations
        for delivery_idx in route:
            # Ensure index is within bounds and use iloc for positional access
            if isinstance(delivery_idx, (int, np.integer)) and 0 <= delivery_idx < len(delivery_locations):
                loc = delivery_locations.iloc[delivery_idx]
                route_coords.append([loc['latitude'], loc['longitude']])
            else:
                # Skip invalid indices
                continue
        
        # Return to depot
        route_coords.append(depot_coords)
        
        # Draw route line
        folium.PolyLine(
            route_coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"Route: {route_name}",
            tooltip=f"Route: {route_name} ({len(route)} deliveries)"
        ).add_to(m)
        
        # Add markers for deliveries
        for delivery_idx in route:
            # Ensure index is within bounds
            if isinstance(delivery_idx, (int, np.integer)) and 0 <= delivery_idx < len(delivery_locations):
                loc = delivery_locations.iloc[delivery_idx]
                folium.CircleMarker(
                    [loc['latitude'], loc['longitude']],
                    radius=6,
                    popup=f"Delivery: {loc['delivery_id']}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    tooltip=f"Delivery: {loc['delivery_id']}"
                ).add_to(m)
        
        # Add start/end depot marker with different icon
        folium.Marker(
            depot_coords,
            popup=f"Depot (Start/End)",
            icon=folium.Icon(color='black', icon='home', prefix='fa'),
            tooltip=f"Depot - Route: {route_name}"
        ).add_to(m)
    
    return m


def plot_cost_comparison(baseline_metrics, optimized_metrics):
    """Create a bar chart comparing baseline vs optimized costs"""
    categories = ['Total Cost', 'Distance (km)', 'Time (hours)', 'Fuel Cost']
    baseline_values = [
        baseline_metrics.get('total_cost', 0),
        baseline_metrics.get('total_distance', 0),
        baseline_metrics.get('total_time', 0),
        baseline_metrics.get('fuel_cost', 0)
    ]
    optimized_values = [
        optimized_metrics.get('total_cost', 0),
        optimized_metrics.get('total_distance', 0),
        optimized_metrics.get('total_time', 0),
        optimized_metrics.get('fuel_cost', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline',
        x=categories,
        y=baseline_values,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Optimized',
        x=categories,
        y=optimized_values,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Route Optimization Comparison',
        xaxis_title='Metric',
        yaxis_title='Value',
        barmode='group',
        height=400
    )
    
    return fig


def plot_cost_savings(baseline_metrics, optimized_metrics):
    """Create a chart showing cost savings percentage"""
    savings = {}
    for key in ['total_cost', 'total_distance', 'total_time', 'fuel_cost']:
        baseline = baseline_metrics.get(key, 0)
        optimized = optimized_metrics.get(key, 0)
        if baseline > 0:
            savings[key.replace('_', ' ').title()] = ((baseline - optimized) / baseline) * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(savings.keys()),
            y=list(savings.values()),
            marker_color='green',
            text=[f'{v:.1f}%' for v in savings.values()],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Cost Savings Percentage',
        xaxis_title='Metric',
        yaxis_title='Savings (%)',
        height=400
    )
    
    return fig


def plot_fleet_utilization(utilization_metrics):
    """Plot fleet utilization metrics"""
    metrics = ['Weight Utilization', 'Volume Utilization', 'Time Utilization']
    values = [
        utilization_metrics.get('average_weight_utilization', 0) * 100,
        utilization_metrics.get('average_volume_utilization', 0) * 100,
        utilization_metrics.get('average_time_utilization', 0) * 100
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color='steelblue',
            text=[f'{v:.1f}%' for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Fleet Utilization Metrics',
        xaxis_title='Utilization Type',
        yaxis_title='Utilization (%)',
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    return fig


def plot_sla_compliance(sla_metrics):
    """Plot SLA compliance metrics"""
    compliance_rate = sla_metrics.get('sla_compliance_rate', 0) * 100
    violations = sla_metrics.get('total_violations', 0)
    total = sla_metrics.get('total_deliveries', 0)
    
    fig = go.Figure(data=[
        go.Bar(
            x=['SLA Compliance Rate'],
            y=[compliance_rate],
            marker_color='green' if compliance_rate >= 95 else 'orange' if compliance_rate >= 80 else 'red',
            text=[f'{compliance_rate:.1f}%'],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'SLA Compliance ({violations}/{total} violations)',
        xaxis_title='',
        yaxis_title='Compliance Rate (%)',
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    return fig


def plot_emissions_comparison(baseline_emissions, optimized_emissions):
    """Plot carbon emissions comparison"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Baseline', 'Optimized'],
            y=[baseline_emissions, optimized_emissions],
            marker_color=['lightcoral', 'lightgreen'],
            text=[f'{baseline_emissions:.1f} kg', f'{optimized_emissions:.1f} kg'],
            textposition='outside'
        )
    ])
    
    reduction = ((baseline_emissions - optimized_emissions) / baseline_emissions * 100) if baseline_emissions > 0 else 0
    
    fig.update_layout(
        title=f'Carbon Emissions Comparison (Reduction: {reduction:.1f}%)',
        xaxis_title='Scenario',
        yaxis_title='CO2 Emissions (kg)',
        height=400
    )
    
    return fig


def plot_algorithm_convergence(history):
    """Plot algorithm convergence history"""
    if isinstance(history, dict):
        generations = list(range(len(history.get('best_fitness', []))))
        best_fitness = history.get('best_fitness', [])
        avg_fitness = history.get('avg_fitness', [])
    else:
        generations = list(range(len(history)))
        best_fitness = history
        avg_fitness = None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=best_fitness,
        mode='lines',
        name='Best Fitness',
        line=dict(color='blue', width=2)
    ))
    
    if avg_fitness:
        fig.add_trace(go.Scatter(
            x=generations,
            y=avg_fitness,
            mode='lines',
            name='Average Fitness',
            line=dict(color='lightblue', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title='Algorithm Convergence',
        xaxis_title='Generation',
        yaxis_title='Fitness Value',
        height=400
    )
    
    return fig


def plot_delivery_zones(delivery_locations, cluster_labels, centers=None):
    """Plot delivery zones from clustering"""
    # Create a copy to avoid modifying original
    df = delivery_locations.copy()
    df['cluster'] = cluster_labels.astype(str)
    
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color='cluster',
        hover_data=['delivery_id', 'address'],
        zoom=11,
        height=600
    )
    
    fig.update_layout(
        mapbox_style='open-street-map',
        title='Delivery Zone Clustering',
        showlegend=True
    )
    
    return fig


def plot_demand_forecast(demand_df):
    """Plot demand forecast"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=demand_df['date'],
        y=demand_df['forecasted_orders'],
        mode='lines+markers',
        name='Forecasted Demand',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Demand Forecast (30 days)',
        xaxis_title='Date',
        yaxis_title='Number of Orders',
        height=400
    )
    
    return fig


def plot_traffic_patterns(traffic_df):
    """Plot traffic patterns throughout the day"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=traffic_df['hour'],
        y=traffic_df['congestion_factor'] * 100,
        mode='lines+markers',
        name='Congestion Factor',
        line=dict(color='red', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title='Traffic Patterns (24 hours)',
        xaxis_title='Hour of Day',
        yaxis_title='Congestion Factor (%)',
        xaxis=dict(range=[0, 23]),
        height=400
    )
    
    return fig


def plot_weather_impact(weather_df):
    """Plot weather conditions"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weather_df['date'],
        y=weather_df['precipitation_mm'],
        name='Precipitation (mm)',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Scatter(
        x=weather_df['date'],
        y=weather_df['temperature_c'],
        mode='lines+markers',
        name='Temperature (°C)',
        yaxis='y2',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Weather Conditions',
        xaxis_title='Date',
        yaxis=dict(title='Precipitation (mm)', side='left'),
        yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right'),
        height=400
    )
    
    return fig

