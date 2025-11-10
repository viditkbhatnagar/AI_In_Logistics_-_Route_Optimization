"""
Advanced visualization utilities
Heatmaps, animations, 3D visualizations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins


def create_delivery_density_heatmap(delivery_locations, orders_df=None):
    """Create heatmap showing delivery density"""
    # Calculate density using kernel density estimation (simplified)
    lats = delivery_locations['latitude'].values
    lons = delivery_locations['longitude'].values
    
    # Create density grid
    lat_range = np.linspace(lats.min(), lats.max(), 20)
    lon_range = np.linspace(lons.min(), lons.max(), 20)
    density = np.zeros((len(lat_range), len(lon_range)))
    
    for i, lat in enumerate(lat_range):
        for j, lon in enumerate(lon_range):
            # Count nearby deliveries
            distances = np.sqrt((lats - lat)**2 + (lons - lon)**2)
            density[i, j] = np.sum(np.exp(-distances * 100))  # Kernel density
    
    fig = go.Figure(data=go.Heatmap(
        z=density,
        x=lon_range,
        y=lat_range,
        colorscale='Viridis',
        colorbar=dict(title="Delivery Density")
    ))
    
    fig.update_layout(
        title='Delivery Density Heatmap',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        height=500
    )
    
    return fig


def create_traffic_heatmap(traffic_data, delivery_locations):
    """Create traffic heatmap overlay"""
    # Aggregate traffic by location and hour
    traffic_by_location = {}
    
    for _, row in traffic_data.iterrows():
        hour = row['hour']
        level = row['traffic_level']
        congestion = row['congestion_factor']
        
        # Map traffic level to intensity
        intensity_map = {'light': 0.2, 'moderate': 0.5, 'heavy': 0.8, 'severe': 1.0}
        intensity = intensity_map.get(level, 0.5)
        
        # Use delivery locations as reference points
        for idx, loc in delivery_locations.iterrows():
            key = (round(loc['latitude'], 2), round(loc['longitude'], 2))
            if key not in traffic_by_location:
                traffic_by_location[key] = []
            traffic_by_location[key].append(intensity)
    
    # Create heatmap data
    lats = []
    lons = []
    intensities = []
    
    for (lat, lon), ints in traffic_by_location.items():
        lats.append(lat)
        lons.append(lon)
        intensities.append(np.mean(ints))
    
    fig = go.Figure(data=go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(
            size=15,
            color=intensities,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Traffic Intensity")
        ),
        text=[f"Intensity: {i:.2f}" for i in intensities]
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=np.mean(lats), lon=np.mean(lons)),
            zoom=10
        ),
        height=600,
        title='Traffic Intensity Heatmap'
    )
    
    return fig


def create_animated_route_map(delivery_locations, route, depots, interval=1000):
    """Create animated route visualization"""
    # Create base map
    center_lat = delivery_locations['latitude'].mean()
    center_lon = delivery_locations['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add depot
    depot = depots.iloc[0]
    folium.Marker(
        [depot['latitude'], depot['longitude']],
        popup='Depot',
        icon=folium.Icon(color='black', icon='warehouse', prefix='fa')
    ).add_to(m)
    
    # Create route coordinates
    route_coords = []
    route_coords.append([depot['latitude'], depot['longitude']])
    
    for idx in route:
        loc = delivery_locations.iloc[idx]
        route_coords.append([loc['latitude'], loc['longitude']])
    
    route_coords.append([depot['latitude'], depot['longitude']])
    
    # Add route line
    folium.PolyLine(route_coords, color='blue', weight=3).add_to(m)
    
    # Add markers for each stop
    for i, idx in enumerate(route):
        loc = delivery_locations.iloc[idx]
        folium.Marker(
            [loc['latitude'], loc['longitude']],
            popup=f'Stop {i+1}',
            icon=folium.Icon(color='green', number=i+1)
        ).add_to(m)
    
    return m


def create_3d_route_visualization(delivery_locations, route, depots):
    """Create 3D route visualization"""
    # Get coordinates
    depot = depots.iloc[0]
    route_coords = [[depot['latitude'], depot['longitude'], 0]]
    
    for idx in route:
        loc = delivery_locations.iloc[idx]
        route_coords.append([loc['latitude'], loc['longitude'], 0])
    
    route_coords.append([depot['latitude'], depot['longitude'], 0])
    
    # Extract coordinates
    lats = [c[0] for c in route_coords]
    lons = [c[1] for c in route_coords]
    alts = [c[2] for c in route_coords]
    
    fig = go.Figure(data=go.Scatter3d(
        x=lons,
        y=lats,
        z=alts,
        mode='lines+markers',
        marker=dict(
            size=8,
            color=list(range(len(route_coords))),
            colorscale='Viridis',
            showscale=True
        ),
        line=dict(color='blue', width=4),
        text=[f'Stop {i}' for i in range(len(route_coords))],
        hovertemplate='<b>%{text}</b><br>Lat: %{y:.4f}<br>Lon: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='3D Route Visualization',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600
    )
    
    return fig

