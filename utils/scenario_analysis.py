"""
What-if scenario analysis and Monte Carlo simulation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_monte_carlo_simulation(base_route, distance_matrix, vehicle_info, orders_df, 
                               delivery_locations, n_simulations=100):
    """Run Monte Carlo simulation for route uncertainty"""
    results = []
    
    for sim in range(n_simulations):
        # Add random variations to distance (traffic, weather)
        traffic_mult = np.random.uniform(0.8, 1.5)
        weather_mult = np.random.uniform(0.9, 1.3)
        
        total_distance = 0
        total_time = 0
        
        for i in range(len(base_route) - 1):
            from_idx = base_route[i]
            to_idx = base_route[i + 1]
            distance = distance_matrix[from_idx][to_idx]
            adjusted_distance = distance * traffic_mult * weather_mult
            total_distance += adjusted_distance
            total_time += adjusted_distance / 50.0
        
        total_time += len(base_route) * 10 / 60
        
        fuel_cost = total_distance * vehicle_info['fuel_efficiency_l_per_km'] * 1.2
        driver_cost = total_time * 25.0
        maintenance_cost = total_distance * 0.15
        total_cost = fuel_cost + driver_cost + maintenance_cost
        
        results.append({
            'simulation': sim,
            'cost': total_cost,
            'distance': total_distance,
            'time': total_time,
            'traffic_mult': traffic_mult,
            'weather_mult': weather_mult
        })
    
    return pd.DataFrame(results)


def plot_monte_carlo_results(results_df):
    """Plot Monte Carlo simulation results"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cost Distribution', 'Time Distribution', 
                       'Distance Distribution', 'Cost vs Time'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Cost distribution
    fig.add_trace(
        go.Histogram(x=results_df['cost'], name='Cost', nbinsx=30),
        row=1, col=1
    )
    
    # Time distribution
    fig.add_trace(
        go.Histogram(x=results_df['time'], name='Time', nbinsx=30),
        row=1, col=2
    )
    
    # Distance distribution
    fig.add_trace(
        go.Histogram(x=results_df['distance'], name='Distance', nbinsx=30),
        row=2, col=1
    )
    
    # Cost vs Time scatter
    fig.add_trace(
        go.Scatter(x=results_df['time'], y=results_df['cost'], 
                  mode='markers', name='Simulations',
                  marker=dict(size=5, opacity=0.6)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_xaxes(title_text="Time (hrs)", row=1, col=2)
    fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hrs)", row=2, col=2)
    fig.update_yaxes(title_text="Cost ($)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="Monte Carlo Simulation: Route Uncertainty Analysis",
        showlegend=False
    )
    
    return fig


def analyze_what_if_scenarios(base_scenario, scenarios):
    """
    Analyze what-if scenarios
    base_scenario: dict with 'cost', 'time', 'distance', 'emissions'
    scenarios: list of dicts with scenario name and modified parameters
    """
    comparison_data = {
        'Scenario': ['Base'] + [s['name'] for s in scenarios],
        'Cost': [base_scenario['cost']] + [s.get('cost', base_scenario['cost']) for s in scenarios],
        'Time': [base_scenario['time']] + [s.get('time', base_scenario['time']) for s in scenarios],
        'Distance': [base_scenario['distance']] + [s.get('distance', base_scenario['distance']) for s in scenarios],
        'Emissions': [base_scenario.get('emissions', 0)] + [s.get('emissions', 0) for s in scenarios]
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cost Comparison', 'Time Comparison', 
                       'Distance Comparison', 'Emissions Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    scenarios_list = comparison_data['Scenario']
    
    fig.add_trace(go.Bar(x=scenarios_list, y=comparison_data['Cost'], name='Cost'), row=1, col=1)
    fig.add_trace(go.Bar(x=scenarios_list, y=comparison_data['Time'], name='Time'), row=1, col=2)
    fig.add_trace(go.Bar(x=scenarios_list, y=comparison_data['Distance'], name='Distance'), row=2, col=1)
    fig.add_trace(go.Bar(x=scenarios_list, y=comparison_data['Emissions'], name='Emissions'), row=2, col=2)
    
    fig.update_xaxes(title_text="Scenario", row=1, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_xaxes(title_text="Scenario", row=1, col=2)
    fig.update_yaxes(title_text="Time (hrs)", row=1, col=2)
    fig.update_xaxes(title_text="Scenario", row=2, col=1)
    fig.update_yaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_xaxes(title_text="Scenario", row=2, col=2)
    fig.update_yaxes(title_text="Emissions (kg CO₂)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="What-If Scenario Analysis",
        showlegend=False
    )
    
    return fig, comparison_data


def sensitivity_analysis(base_route, distance_matrix, vehicle_info, orders_df,
                        delivery_locations, parameter_name, parameter_range):
    """Perform sensitivity analysis on a parameter"""
    results = []
    
    for param_value in parameter_range:
        # Modify parameter and recalculate
        if parameter_name == 'traffic_multiplier':
            # Simulate different traffic conditions
            total_distance = 0
            total_time = 0
            
            for i in range(len(base_route) - 1):
                from_idx = base_route[i]
                to_idx = base_route[i + 1]
                distance = distance_matrix[from_idx][to_idx]
                adjusted_distance = distance * param_value
                total_distance += adjusted_distance
                total_time += adjusted_distance / (50.0 / param_value)
            
            total_time += len(base_route) * 10 / 60
            
            fuel_cost = total_distance * vehicle_info['fuel_efficiency_l_per_km'] * 1.2
            driver_cost = total_time * 25.0
            maintenance_cost = total_distance * 0.15
            total_cost = fuel_cost + driver_cost + maintenance_cost
            
            results.append({
                'parameter_value': param_value,
                'cost': total_cost,
                'time': total_time,
                'distance': total_distance
            })
    
    return pd.DataFrame(results)


def plot_sensitivity_analysis(results_df, parameter_name):
    """Plot sensitivity analysis results"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Cost Sensitivity', 'Time Sensitivity', 'Distance Sensitivity')
    )
    
    fig.add_trace(
        go.Scatter(x=results_df['parameter_value'], y=results_df['cost'],
                  mode='lines+markers', name='Cost'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results_df['parameter_value'], y=results_df['time'],
                  mode='lines+markers', name='Time', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=results_df['parameter_value'], y=results_df['distance'],
                  mode='lines+markers', name='Distance', showlegend=False),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text=parameter_name, row=1, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_xaxes(title_text=parameter_name, row=1, col=2)
    fig.update_yaxes(title_text="Time (hrs)", row=1, col=2)
    fig.update_xaxes(title_text=parameter_name, row=1, col=3)
    fig.update_yaxes(title_text="Distance (km)", row=1, col=3)
    
    fig.update_layout(
        height=400,
        title_text=f"Sensitivity Analysis: {parameter_name}",
        showlegend=True
    )
    
    return fig

