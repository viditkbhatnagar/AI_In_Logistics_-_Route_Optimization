"""
Multi-objective optimization utilities
Pareto front visualization and trade-off analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_pareto_front(solutions):
    """
    Calculate Pareto front from multiple solutions
    solutions: list of dicts with 'cost', 'time', 'emissions', 'route'
    """
    if len(solutions) == 0:
        return []
    
    # Normalize objectives (lower is better for all)
    costs = np.array([s['cost'] for s in solutions])
    times = np.array([s['time'] for s in solutions])
    emissions = np.array([s.get('emissions', 0) for s in solutions])
    
    # Normalize to 0-1 scale
    cost_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-10)
    time_norm = (times - times.min()) / (times.max() - times.min() + 1e-10)
    emissions_norm = (emissions - emissions.min()) / (emissions.max() - emissions.min() + 1e-10)
    
    # Calculate dominance
    pareto_indices = []
    for i, sol in enumerate(solutions):
        is_dominated = False
        for j, other in enumerate(solutions):
            if i != j:
                # Check if other dominates i
                if (cost_norm[j] <= cost_norm[i] and 
                    time_norm[j] <= time_norm[i] and 
                    emissions_norm[j] <= emissions_norm[i] and
                    (cost_norm[j] < cost_norm[i] or 
                     time_norm[j] < time_norm[i] or 
                     emissions_norm[j] < emissions_norm[i])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_indices.append(i)
    
    return [solutions[i] for i in pareto_indices]


def plot_pareto_front(solutions, pareto_solutions=None):
    """Plot Pareto front visualization"""
    if pareto_solutions is None:
        pareto_solutions = calculate_pareto_front(solutions)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cost vs Time', 'Cost vs Emissions', 'Time vs Emissions', '3D Pareto Front'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    
    # Extract data
    all_costs = [s['cost'] for s in solutions]
    all_times = [s['time'] for s in solutions]
    all_emissions = [s.get('emissions', 0) for s in solutions]
    
    pareto_costs = [s['cost'] for s in pareto_solutions]
    pareto_times = [s['time'] for s in pareto_solutions]
    pareto_emissions = [s.get('emissions', 0) for s in pareto_solutions]
    
    # Cost vs Time
    fig.add_trace(
        go.Scatter(x=all_times, y=all_costs, mode='markers', 
                  name='All Solutions', marker=dict(color='lightblue', size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=pareto_times, y=pareto_costs, mode='markers',
                  name='Pareto Front', marker=dict(color='red', size=12, symbol='star')),
        row=1, col=1
    )
    
    # Cost vs Emissions
    fig.add_trace(
        go.Scatter(x=all_emissions, y=all_costs, mode='markers',
                  name='All Solutions', marker=dict(color='lightblue', size=8), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=pareto_emissions, y=pareto_costs, mode='markers',
                  name='Pareto Front', marker=dict(color='red', size=12, symbol='star'), showlegend=False),
        row=1, col=2
    )
    
    # Time vs Emissions
    fig.add_trace(
        go.Scatter(x=all_emissions, y=all_times, mode='markers',
                  name='All Solutions', marker=dict(color='lightblue', size=8), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=pareto_emissions, y=pareto_times, mode='markers',
                  name='Pareto Front', marker=dict(color='red', size=12, symbol='star'), showlegend=False),
        row=2, col=1
    )
    
    # 3D Pareto Front
    fig.add_trace(
        go.Scatter3d(x=pareto_times, y=pareto_costs, z=pareto_emissions,
                     mode='markers', name='Pareto Front',
                     marker=dict(size=10, color='red', symbol='diamond')),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Time (hrs)", row=1, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_xaxes(title_text="Emissions (kg CO₂)", row=1, col=2)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
    fig.update_xaxes(title_text="Emissions (kg CO₂)", row=2, col=1)
    fig.update_yaxes(title_text="Time (hrs)", row=2, col=1)
    
    fig.update_layout(
        height=800,
        title_text="Multi-Objective Optimization: Pareto Front Analysis",
        showlegend=True
    )
    
    return fig

