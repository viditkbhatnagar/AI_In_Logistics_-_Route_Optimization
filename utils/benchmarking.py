"""
Benchmarking and performance comparison utilities
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time


class AlgorithmBenchmark:
    """Benchmark different algorithms"""
    
    def __init__(self):
        self.results = []
    
    def benchmark_algorithm(self, name, algorithm_func, *args, **kwargs):
        """Benchmark a single algorithm"""
        start_time = time.time()
        result = algorithm_func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        return {
            'algorithm': name,
            'result': result,
            'execution_time': execution_time
        }
    
    def compare_algorithms(self, algorithms, test_cases):
        """Compare multiple algorithms on test cases"""
        comparison_results = []
        
        for test_name, test_data in test_cases.items():
            for algo_name, algo_func in algorithms.items():
                result = self.benchmark_algorithm(
                    algo_name,
                    algo_func,
                    **test_data
                )
                result['test_case'] = test_name
                comparison_results.append(result)
        
        return pd.DataFrame(comparison_results)


def plot_algorithm_comparison(comparison_df):
    """Plot algorithm comparison"""
    algorithms = comparison_df['algorithm'].unique()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Execution Time Comparison', 'Solution Quality Comparison')
    )
    
    # Execution time comparison
    for algo in algorithms:
        algo_data = comparison_df[comparison_df['algorithm'] == algo]
        fig.add_trace(
            go.Bar(x=algo_data['test_case'], y=algo_data['execution_time'],
                  name=algo),
            row=1, col=1
        )
    
    # Solution quality (if available)
    if 'solution_cost' in comparison_df.columns:
        for algo in algorithms:
            algo_data = comparison_df[comparison_df['algorithm'] == algo]
            fig.add_trace(
                go.Bar(x=algo_data['test_case'], y=algo_data['solution_cost'],
                      name=algo, showlegend=False),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text="Test Case", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Test Case", row=1, col=2)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text="Algorithm Benchmark Comparison",
        barmode='group'
    )
    
    return fig


def calculate_performance_metrics(route, distance_matrix, vehicle_info, orders_df, delivery_locations):
    """Calculate comprehensive performance metrics"""
    # Basic metrics
    total_distance = sum(distance_matrix[route[i]][route[i+1]] 
                        for i in range(len(route)-1))
    
    # Calculate time
    total_time = total_distance / 50.0 + len(route) * 10 / 60
    
    # Calculate costs
    fuel_cost = total_distance * vehicle_info['fuel_efficiency_l_per_km'] * 1.2
    driver_cost = total_time * 25.0
    maintenance_cost = total_distance * 0.15
    total_cost = fuel_cost + driver_cost + maintenance_cost
    
    # Calculate emissions
    emissions_kg = total_distance * 0.2  # kg CO2 per km
    
    # Calculate utilization
    total_weight = orders_df.loc[
        orders_df['delivery_id'].isin(
            [delivery_locations.iloc[i]['delivery_id'] for i in route]
        ), 'weight_kg'
    ].sum()
    
    weight_utilization = total_weight / vehicle_info['capacity_kg'] if vehicle_info['capacity_kg'] > 0 else 0
    
    return {
        'total_cost': total_cost,
        'total_distance': total_distance,
        'total_time': total_time,
        'emissions_kg': emissions_kg,
        'weight_utilization': weight_utilization,
        'n_deliveries': len(route),
        'cost_per_delivery': total_cost / len(route) if len(route) > 0 else 0,
        'distance_per_delivery': total_distance / len(route) if len(route) > 0 else 0
    }

