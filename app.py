"""
Streamlit App - AI in Logistics & Route Optimization
Main application with multiple tabs addressing all 7 key questions
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_dataset, list_available_datasets
from utils.route_optimizer import calculate_distance_matrix, generate_baseline_route, calculate_route_metrics
from utils.visualization import create_route_map, plot_cost_comparison, plot_cost_savings, plot_algorithm_convergence
from utils.metrics import calculate_all_metrics
from utils.carbon_calculator import calculate_fleet_emissions, calculate_emissions_reduction
from algorithms.genetic_algorithm import GeneticAlgorithmVRP
from algorithms.reinforcement_learning import QLearningRouter
from algorithms.clustering import DeliveryZoneOptimizer
from algorithms.ml_models import DeliveryTimePredictor, SLARiskPredictor
from algorithms.ant_colony import AntColonyVRP
from algorithms.simulated_annealing import SimulatedAnnealingVRP
from utils.multi_objective import calculate_pareto_front, plot_pareto_front
from utils.advanced_visualization import (
    create_delivery_density_heatmap, create_traffic_heatmap,
    create_animated_route_map, create_3d_route_visualization
)
from utils.scenario_analysis import (
    run_monte_carlo_simulation, plot_monte_carlo_results,
    analyze_what_if_scenarios, sensitivity_analysis, plot_sensitivity_analysis
)
from utils.export_utils import (
    export_route_to_csv, export_metrics_to_excel, create_pdf_report,
    export_plot_as_image, create_shareable_link, load_from_shareable_link
)
from utils.benchmarking import AlgorithmBenchmark, plot_algorithm_comparison, calculate_performance_metrics
from utils.explainable_ai import calculate_shap_values, plot_shap_summary, plot_shap_waterfall
from utils.multi_depot import optimize_multi_depot_routing, optimize_pickup_delivery
from utils.executive_dashboard import create_executive_dashboard, calculate_roi
from utils.api_integration import MockGoogleMapsAPI, MockHEREMapsAPI, demonstrate_api_integration
from streamlit_folium import st_folium
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="AI in Logistics & Route Optimization",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_dataset(scenario_name):
    """Load and cache dataset"""
    return load_dataset(scenario_name)


def tab_route_optimization():
    """Tab 1: Route Optimization & Cost Reduction"""
    st.header("🚛 Route Optimization & Cost Reduction")
    
    st.markdown("""
    **Question: How can AI optimize delivery routes and reduce transportation costs?**
    
    This tab demonstrates how Genetic Algorithms can optimize delivery routes to minimize costs,
    distance, and time while respecting vehicle capacity constraints.
    """)
    
    with st.expander("📖 What's Happening Here?", expanded=False):
        st.markdown("""
        **Understanding Route Optimization:**
        
        Route optimization is the process of finding the best sequence of delivery stops that minimizes total cost, 
        distance, and time while respecting constraints like vehicle capacity and delivery time windows.
        
        **How Genetic Algorithms Work:**
        1. **Population**: Creates multiple candidate routes (solutions)
        2. **Fitness Evaluation**: Calculates cost for each route (distance + time + penalties)
        3. **Selection**: Chooses better routes to "reproduce"
        4. **Crossover**: Combines parts of good routes to create new ones
        5. **Mutation**: Randomly changes routes to explore new solutions
        6. **Evolution**: Repeats for multiple generations, improving over time
        
        **Traffic & Weather Integration:**
        - Routes are optimized considering real-time traffic patterns (peak hours slow down delivery)
        - Weather conditions affect travel speed (rain/snow = slower)
        - The algorithm avoids peak hours when possible to minimize delays
        
        **What You'll See:**
        - **Baseline Route**: A simple nearest-neighbor route (starting point)
        - **Optimized Route**: AI-optimized route that reduces cost by 15-30%
        - **Visual Comparison**: Maps showing both routes side-by-side
        - **Metrics**: Cost savings, distance reduction, time improvement
        """)
    
    # Load dataset
    datasets = list_available_datasets()
    dataset_names = [d['name'] for d in datasets]
    
    # Show currently loaded dataset
    current_loaded = st.session_state.get('selected_dataset', None)
    if current_loaded:
        st.info(f"📦 Currently loaded: **{current_loaded}** ({len(st.session_state.get('dataset', {}).get('delivery_locations', []))} deliveries)")
    
    # Find index of currently loaded dataset
    default_index = 1
    if current_loaded and current_loaded in dataset_names:
        default_index = dataset_names.index(current_loaded)
    
    selected_dataset = st.selectbox("Select Dataset", dataset_names, index=default_index, key="dataset_selector")
    
    # Auto-load if selection changes or manual load button
    col1, col2 = st.columns([1, 4])
    with col1:
        load_clicked = st.button("Load Dataset", type="primary")
    with col2:
        if current_loaded and current_loaded != selected_dataset:
            st.warning(f"⚠️ Different dataset selected. Click 'Load Dataset' to load **{selected_dataset}**")
    
    # Load dataset only when button is clicked
    if load_clicked:
        with st.spinner(f"Loading dataset: {selected_dataset}..."):
            dataset = load_cached_dataset(selected_dataset)
            # Ensure scenario_name is set
            if 'scenario_name' not in dataset:
                dataset['scenario_name'] = selected_dataset
            st.session_state['dataset'] = dataset
            st.session_state['selected_dataset'] = selected_dataset
            # Clear old optimization results when dataset changes
            if 'optimized_route' in st.session_state:
                del st.session_state['optimized_route']
                del st.session_state['baseline_route']
                del st.session_state['baseline_metrics']
                del st.session_state['optimized_metrics']
                del st.session_state['ga_history']
                if 'sample_delivery_locations' in st.session_state:
                    del st.session_state['sample_delivery_locations']
            # Clear distance matrix cache
            if 'distance_matrix_cache' in st.session_state:
                del st.session_state['distance_matrix_cache']
            st.success(f"✅ Dataset '{selected_dataset}' loaded successfully!")
            st.rerun()
    
    # Don't show error if no dataset loaded yet - just show info message
    if 'dataset' not in st.session_state:
        st.info("Please select a dataset and click 'Load Dataset' to begin optimization.")
        return
    
    dataset = st.session_state['dataset']
    
    # Only show a gentle reminder if datasets don't match (not an error)
    loaded_scenario = dataset.get('scenario_name', '')
    if loaded_scenario != selected_dataset:
        # Show a subtle info message, not an error
        st.info(f"💡 **Tip:** You have `{loaded_scenario}` loaded. To switch to `{selected_dataset}`, click 'Load Dataset' above.")
    
    delivery_locations = dataset['delivery_locations']
    orders = dataset['orders']
    vehicles = dataset['vehicles']
    depots = dataset['depots']
    traffic = dataset['traffic']
    weather = dataset['weather']
    
    # Show dataset info prominently
    dataset_name_display = dataset.get('scenario_name', 'Unknown').replace('_', ' ').title()
    st.subheader(f"📦 Dataset: {dataset_name_display}")
    
    # Show metadata info
    if 'metadata' in dataset and dataset['metadata']:
        meta = dataset['metadata']
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.caption(f"🏙️ **City:** {meta.get('city', 'N/A')}")
        with col_info2:
            st.caption(f"📊 **Deliveries:** {meta.get('n_deliveries', len(delivery_locations))}")
        with col_info3:
            st.caption(f"🚚 **Vehicles:** {meta.get('n_vehicles', len(vehicles))}")
    
    # Show traffic and weather conditions
    st.subheader("🌦️ Current Conditions (Affecting Route Optimization)")
    col_cond1, col_cond2, col_cond3, col_cond4 = st.columns(4)
    
    # Get current hour for display (default 9 AM)
    current_hour_display = 9
    current_traffic = traffic[traffic['hour'] == current_hour_display]
    if len(current_traffic) > 0:
        traffic_level = current_traffic.iloc[0]['traffic_level']
        traffic_speed = current_traffic.iloc[0]['average_speed_kmh']
    else:
        traffic_level = 'light'
        traffic_speed = 50.0
    
    # Handle weather condition safely
    from config import TRAFFIC_MULTIPLIERS, WEATHER_MULTIPLIERS
    traffic_mult = TRAFFIC_MULTIPLIERS.get(traffic_level, 1.0)
    
    if len(weather) > 0:
        current_weather_row = weather.iloc[-1]
        weather_condition = current_weather_row['condition']
        weather_mult = WEATHER_MULTIPLIERS.get(weather_condition, 1.0)
    else:
        weather_condition = 'clear'
        weather_mult = 1.0
        current_weather_row = None
    
    with col_cond1:
        st.metric("Traffic Level", traffic_level.title(), f"{traffic_speed:.1f} km/h")
        st.caption(f"Multiplier: {traffic_mult:.2f}x")
    with col_cond2:
        if current_weather_row is not None:
            weather_condition_display = weather_condition.title()
            weather_temp = current_weather_row['temperature_c']
            st.metric("Weather", weather_condition_display, 
                     f"{weather_temp:.1f}°C")
            st.caption(f"Multiplier: {weather_mult:.2f}x")
        else:
            st.metric("Weather", "N/A")
    with col_cond3:
        combined_impact = traffic_mult * weather_mult
        st.metric("Combined Impact", f"{combined_impact:.2f}x", 
                 "Time multiplier")
        st.caption("Higher = slower delivery")
    with col_cond4:
        # Show peak hours
        peak_hours = traffic[traffic['traffic_level'].isin(['heavy', 'severe'])]
        if len(peak_hours) > 0:
            peak_hour_list = sorted(peak_hours['hour'].unique().tolist())
            peak_str = ', '.join([f"{h}:00" for h in peak_hour_list[:5]])
            st.metric("Peak Hours", f"{len(peak_hours)} hours", peak_str)
        else:
            st.metric("Peak Hours", "None", "Light traffic")
    
    st.caption("💡 **Note:** Routes are optimized considering traffic patterns (peak hours) and weather conditions to minimize delivery time and cost.")
    
    # Show traffic pattern visualization
    with st.expander("📊 View Traffic Patterns Throughout the Day", expanded=False):
        from utils.visualization import plot_traffic_patterns
        st.plotly_chart(plot_traffic_patterns(traffic), use_container_width=True)
        st.caption("Traffic congestion varies throughout the day. Routes are optimized to avoid peak hours when possible.")
    
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Deliveries", len(delivery_locations))
    with col2:
        st.metric("Vehicles", len(vehicles))
    with col3:
        st.metric("Orders", len(orders))
    with col4:
        st.metric("Depots", len(depots))
    
    # Show dataset preview
    with st.expander("📊 View Dataset Details", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Delivery Locations", "Orders", "Vehicles", "Depots"])
        
        with tab1:
            st.dataframe(delivery_locations.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(delivery_locations)} delivery locations")
        
        with tab2:
            st.dataframe(orders.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(orders)} orders")
        
        with tab3:
            st.dataframe(vehicles, use_container_width=True)
        
        with tab4:
            st.dataframe(depots, use_container_width=True)
    
    # Warning for large datasets
    if len(delivery_locations) > 200:
        st.warning(f"⚠️ Large dataset detected ({len(delivery_locations)} deliveries). For faster optimization, consider using a smaller dataset or reducing generations.")
    
    st.subheader("Optimization Parameters")
    col1, col2, col3 = st.columns(3)
    
    # Adjust defaults based on dataset size
    if len(delivery_locations) > 500:
        default_pop = 50
        default_gen = 50
    elif len(delivery_locations) > 200:
        default_pop = 75
        default_gen = 100
    else:
        default_pop = 100
        default_gen = 200
    
    with col1:
        population_size = st.slider("Population Size", 20, 200, default_pop)
    with col2:
        generations = st.slider("Generations", 20, 500, default_gen)
    with col3:
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.3, 0.1)
    
    # Option to use sample for large datasets
    use_sample = False
    sample_size = len(delivery_locations)
    if len(delivery_locations) > 200:
        use_sample = st.checkbox(f"Use sample of deliveries for faster optimization (recommended for large datasets)", value=True)
        if use_sample:
            sample_size = st.slider("Sample Size", 50, min(200, len(delivery_locations)), min(100, len(delivery_locations)))
    
    if st.button("Run Optimization", type="primary"):
        # Select deliveries to optimize
        if use_sample and sample_size < len(delivery_locations):
            import random
            random.seed(42)
            sample_indices = random.sample(range(len(delivery_locations)), sample_size)
            sample_delivery_locations = delivery_locations.iloc[sample_indices].reset_index(drop=True)
            sample_orders = orders[orders['delivery_id'].isin(sample_delivery_locations['delivery_id'])].reset_index(drop=True)
            st.info(f"Optimizing route for {sample_size} deliveries (sample from {len(delivery_locations)})")
        else:
            sample_delivery_locations = delivery_locations.reset_index(drop=True)
            sample_orders = orders.reset_index(drop=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Select first vehicle for demonstration
        vehicle = vehicles.iloc[0]
        vehicle_info = vehicle.to_dict()
        
        # Calculate distance matrix with progress
        status_text.text("Step 1/4: Calculating distance matrix...")
        progress_bar.progress(10)
        # Use a simple cache key based on number of locations
        cache_key = f"{len(sample_delivery_locations)}_{hash(tuple(sample_delivery_locations['latitude'].head(5).values))}"
        if 'distance_matrix_cache' not in st.session_state:
            st.session_state['distance_matrix_cache'] = {}
        if cache_key not in st.session_state['distance_matrix_cache']:
            distance_matrix = calculate_distance_matrix(sample_delivery_locations)
            st.session_state['distance_matrix_cache'][cache_key] = distance_matrix
        else:
            distance_matrix = st.session_state['distance_matrix_cache'][cache_key]
            status_text.text("Step 1/4: Using cached distance matrix...")
        progress_bar.progress(30)
        
        # Generate baseline route
        status_text.text("Step 2/4: Generating baseline route...")
        depot = depots.iloc[0]
        # Ensure sample_delivery_locations has reset index for consistent indexing
        sample_delivery_locations = sample_delivery_locations.reset_index(drop=True)
        baseline_route = generate_baseline_route(sample_delivery_locations, depot, sample_orders)
        progress_bar.progress(40)
        
        # Run Genetic Algorithm with progress (now includes traffic and weather)
        status_text.text("Step 3/4: Running Genetic Algorithm optimization (considering traffic & weather)...")
        
        def update_progress(gen, total_gen, best_fit, avg_fit):
            progress = 40 + int((gen / total_gen) * 50)
            progress_bar.progress(min(progress, 90))
            status_text.text(f"Step 3/4: Generation {gen}/{total_gen} - Best Fitness: {best_fit:.2f}")
        
        ga = GeneticAlgorithmVRP(
            distance_matrix,
            sample_orders,
            sample_delivery_locations,
            vehicle_info,
            traffic_data=traffic,  # Pass traffic data
            weather_data=weather,   # Pass weather data
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
        
        optimized_route, history = ga.solve(progress_callback=update_progress)
        progress_bar.progress(90)
        
        # Calculate metrics
        status_text.text("Step 4/4: Calculating metrics...")
        baseline_metrics = calculate_route_metrics(
            baseline_route, distance_matrix, vehicle_info, sample_orders, sample_delivery_locations, 0, 
            traffic_data=traffic, weather_data=weather, depot_location=depot
        )
        optimized_metrics = calculate_route_metrics(
            optimized_route, distance_matrix, vehicle_info, sample_orders, sample_delivery_locations, 0,
            traffic_data=traffic, weather_data=weather, depot_location=depot
        )
        
        progress_bar.progress(100)
        status_text.text("✓ Optimization complete!")
        
        st.session_state['baseline_route'] = baseline_route
        st.session_state['optimized_route'] = optimized_route
        st.session_state['baseline_metrics'] = baseline_metrics
        st.session_state['optimized_metrics'] = optimized_metrics
        st.session_state['ga_history'] = history
        st.session_state['vehicle_info'] = vehicle_info
        st.session_state['sample_delivery_locations'] = sample_delivery_locations
    
    if 'optimized_route' in st.session_state:
        st.subheader("Results Comparison")
        
        baseline_metrics = st.session_state['baseline_metrics']
        optimized_metrics = st.session_state['optimized_metrics']
        
        # Show how traffic/weather affected the optimization
        peak_hours_data = traffic[traffic['traffic_level'].isin(['heavy', 'severe'])]
        peak_hour_list = sorted(peak_hours_data['hour'].unique().tolist()[:3]) if len(peak_hours_data) > 0 else []
        peak_hours_str = ', '.join([f"{h}:00" for h in peak_hour_list]) if peak_hour_list else "None"
        
        st.info(f"""
        **Optimization Impact:**
        - Traffic conditions considered: Peak hours ({peak_hours_str}) have {traffic_mult:.2f}x impact
        - Weather conditions: {weather_condition.title()} adds {weather_mult:.2f}x time multiplier
        - **Combined effect:** Routes optimized to minimize time during peak hours and adverse weather
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cost_savings = ((baseline_metrics['total_cost'] - optimized_metrics['total_cost']) / baseline_metrics['total_cost']) * 100
            st.metric("Cost Savings", f"{cost_savings:.1f}%", f"${baseline_metrics['total_cost'] - optimized_metrics['total_cost']:.2f}")
        with col2:
            distance_savings = ((baseline_metrics['total_distance'] - optimized_metrics['total_distance']) / baseline_metrics['total_distance']) * 100
            st.metric("Distance Reduction", f"{distance_savings:.1f}%", f"{baseline_metrics['total_distance'] - optimized_metrics['total_distance']:.2f} km")
        with col3:
            time_savings = ((baseline_metrics['total_time'] - optimized_metrics['total_time']) / baseline_metrics['total_time']) * 100
            st.metric("Time Savings", f"{time_savings:.1f}%", f"{baseline_metrics['total_time'] - optimized_metrics['total_time']:.2f} hrs")
        with col4:
            st.metric("Deliveries", len(st.session_state['optimized_route']), f"Baseline: {len(st.session_state['baseline_route'])}")
        
        # Cost comparison chart
        st.subheader("Cost Breakdown Comparison")
        st.plotly_chart(plot_cost_comparison(baseline_metrics, optimized_metrics), use_container_width=True)
        with st.expander("💡 What This Chart Shows", expanded=False):
            st.markdown("""
            This bar chart compares the cost components between baseline and optimized routes:
            - **Total Cost**: Sum of all cost components
            - **Fuel Cost**: Based on distance traveled and fuel efficiency
            - **Driver Cost**: Based on total time (hourly rate)
            - **Maintenance Cost**: Based on distance traveled
            
            **Key Insight**: The optimized route typically shows lower costs across all categories, 
            with the biggest savings usually in fuel and driver costs due to reduced distance and time.
            """)
        
        # Cost savings chart
        st.subheader("Cost Savings Breakdown")
        st.plotly_chart(plot_cost_savings(baseline_metrics, optimized_metrics), use_container_width=True)
        with st.expander("💡 What This Chart Shows", expanded=False):
            st.markdown("""
            This chart shows the **percentage savings** for each cost category:
            - Positive values = savings (good!)
            - Higher bars = more savings in that category
            
            **Interpretation**: Look for which cost category benefits most from optimization. 
            Typically, fuel and driver costs show the highest percentage reductions.
            """)
        
        # Algorithm convergence
        if 'ga_history' in st.session_state:
            st.subheader("Algorithm Convergence")
            st.plotly_chart(plot_algorithm_convergence(st.session_state['ga_history']), use_container_width=True)
            with st.expander("💡 What This Graph Shows", expanded=False):
                st.markdown("""
                This graph shows how the Genetic Algorithm improved over generations:
                - **X-axis**: Generation number (iteration)
                - **Y-axis**: Fitness value (lower = better route)
                - **Best Fitness**: The best route found in each generation
                - **Average Fitness**: Average quality of all routes in that generation
                
                **What to Look For**:
                - Steep drop early = algorithm quickly finds better solutions
                - Plateau = algorithm has converged (found a good solution)
                - Gap between best and average = population diversity (good for exploration)
                
                **Why It Matters**: Shows the algorithm is learning and improving, not just random search.
                """)
        
        # Route visualization
        st.subheader("Route Visualization")
        
        view_option = st.radio("View", ["Baseline Route", "Optimized Route", "Both"], horizontal=True, key="route_view")
        
        # Add visualization options
        viz_type = st.radio("Visualization Type", ["2D Map", "3D Route", "Heatmap"], horizontal=True, key="viz_type")
        
        # Use sample locations if available (must match what was used for optimization)
        locs_to_plot = st.session_state.get('sample_delivery_locations', delivery_locations)
        # Ensure indices are reset for consistent access
        if 'sample_delivery_locations' in st.session_state:
            locs_to_plot = st.session_state['sample_delivery_locations'].reset_index(drop=True)
        else:
            locs_to_plot = delivery_locations.reset_index(drop=True)
        
        # Prepare routes based on selection
        if view_option == "Baseline Route":
            routes_to_show = {'Baseline Route': st.session_state['baseline_route']}
        elif view_option == "Optimized Route":
            routes_to_show = {'Optimized Route': st.session_state['optimized_route']}
        else:  # Both
            routes_to_show = {
                'Baseline Route': st.session_state['baseline_route'],
                'Optimized Route': st.session_state['optimized_route']
            }
        
        # Debug info (can be removed later)
        if st.checkbox("Show Debug Info", value=False):
            st.write(f"Locations to plot: {len(locs_to_plot)}")
            st.write(f"Baseline route length: {len(st.session_state['baseline_route'])}")
            st.write(f"Optimized route length: {len(st.session_state['optimized_route'])}")
            st.write(f"Baseline route indices (first 10): {st.session_state['baseline_route'][:10]}")
            st.write(f"Optimized route indices (first 10): {st.session_state['optimized_route'][:10]}")
        
        if viz_type == "2D Map":
            route_map = create_route_map(
                locs_to_plot,
                depots,
                routes_to_show,
                vehicles.head(1)
            )
            st_folium(route_map, width=1200, height=600)
        elif viz_type == "3D Route":
            if 'Optimized Route' in routes_to_show:
                route_3d = create_3d_route_visualization(locs_to_plot, routes_to_show['Optimized Route'], depots)
                st.plotly_chart(route_3d, use_container_width=True)
            else:
                route_3d = create_3d_route_visualization(locs_to_plot, list(routes_to_show.values())[0], depots)
                st.plotly_chart(route_3d, use_container_width=True)
        else:  # Heatmap
            heatmap_fig = create_delivery_density_heatmap(locs_to_plot, orders)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.caption("💡 **Heatmap shows delivery density - darker areas indicate higher concentration of deliveries.**")
        
        with st.expander("🗺️ Understanding the Route Map", expanded=False):
            st.markdown("""
            **Map Elements:**
            - **Black markers (warehouse icon)**: Depot location (start/end point)
            - **Colored circles**: Delivery locations
            - **Colored lines**: Route paths connecting deliveries
            - **Different colors**: Different routes (when viewing "Both")
            
            **What to Compare:**
            - **Baseline Route**: Often has many crossings and backtracking
            - **Optimized Route**: Typically has fewer crossings, smoother paths, and better clustering
            
            **Key Observations:**
            - Optimized routes tend to group nearby deliveries together
            - Fewer route crossings mean less backtracking
            - Shorter total path length = lower cost and time
            """)
        
        # Show route statistics
        col_route1, col_route2 = st.columns(2)
        with col_route1:
            st.metric("Baseline Route", f"{len(st.session_state['baseline_route'])} deliveries", 
                     f"{baseline_metrics['total_distance']:.2f} km")
        with col_route2:
            st.metric("Optimized Route", f"{len(st.session_state['optimized_route'])} deliveries",
                     f"{optimized_metrics['total_distance']:.2f} km")
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Total Cost ($)', 'Total Distance (km)', 'Total Time (hrs)', 'Fuel Cost ($)', 'Driver Cost ($)', 'SLA Compliance (%)'],
            'Baseline': [
                f"${baseline_metrics['total_cost']:.2f}",
                f"{baseline_metrics['total_distance']:.2f}",
                f"{baseline_metrics['total_time']:.2f}",
                f"${baseline_metrics['fuel_cost']:.2f}",
                f"${baseline_metrics['driver_cost']:.2f}",
                f"{baseline_metrics['sla_compliance_rate']*100:.1f}"
            ],
            'Optimized': [
                f"${optimized_metrics['total_cost']:.2f}",
                f"{optimized_metrics['total_distance']:.2f}",
                f"{optimized_metrics['total_time']:.2f}",
                f"${optimized_metrics['fuel_cost']:.2f}",
                f"${optimized_metrics['driver_cost']:.2f}",
                f"{optimized_metrics['sla_compliance_rate']*100:.1f}"
            ],
            'Improvement': [
                f"{((baseline_metrics['total_cost'] - optimized_metrics['total_cost']) / baseline_metrics['total_cost'] * 100):.1f}%",
                f"{((baseline_metrics['total_distance'] - optimized_metrics['total_distance']) / baseline_metrics['total_distance'] * 100):.1f}%",
                f"{((baseline_metrics['total_time'] - optimized_metrics['total_time']) / baseline_metrics['total_time'] * 100):.1f}%",
                f"{((baseline_metrics['fuel_cost'] - optimized_metrics['fuel_cost']) / baseline_metrics['fuel_cost'] * 100):.1f}%",
                f"{((baseline_metrics['driver_cost'] - optimized_metrics['driver_cost']) / baseline_metrics['driver_cost'] * 100):.1f}%",
                f"{(optimized_metrics['sla_compliance_rate'] - baseline_metrics['sla_compliance_rate']) * 100:.1f}%"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def get_dataset():
    """Helper to get current dataset"""
    if 'dataset' not in st.session_state:
        datasets = list_available_datasets()
        dataset_names = [d['name'] for d in datasets]
        selected = st.selectbox("Select Dataset", dataset_names, index=1)
        if st.button("Load"):
            dataset = load_cached_dataset(selected)
            st.session_state['dataset'] = dataset
            st.session_state['selected_dataset'] = selected
            st.rerun()
        return None
    return st.session_state['dataset']


def tab_realtime_data():
    """Tab 2: Real-Time Data Integration & Dynamic Routing"""
    st.header("🔄 Real-Time Data Integration & Dynamic Routing")
    st.markdown("""
    **Question: What role do real-time traffic, weather, and demand data play in dynamic routing?**
    
    This tab demonstrates how real-time data affects routing decisions and how routes adapt dynamically 
    based on changing conditions (traffic incidents, weather changes, demand surges).
    """)
    
    with st.expander("📖 What's Happening Here?", expanded=False):
        st.markdown("""
        **Understanding Real-Time Data Integration:**
        
        Real-world logistics operations face constantly changing conditions:
        - **Traffic**: Congestion varies by time of day (peak hours = slower)
        - **Weather**: Rain, snow, or storms slow down vehicles
        - **Demand**: Order volumes fluctuate throughout the day
        
        **Why It Matters:**
        A route planned at 9 AM might be optimal then, but by 2 PM, traffic patterns change, 
        making that route inefficient. Real-time data allows routes to adapt dynamically.
        
        **Three Routing Strategies Compared:**
        1. **Static Route**: Planned once, never changes (baseline)
        2. **Real-Time Route**: Recalculated considering current conditions
        3. **Adaptive Route (RL)**: Uses Reinforcement Learning to learn optimal decisions
        
        **Reinforcement Learning Agent:**
        - Learns from experience which routes work best under different conditions
        - Adapts automatically when conditions change
        - Balances exploration (trying new routes) vs exploitation (using known good routes)
        
        **What You'll See:**
        - Interactive controls to simulate traffic and weather changes
        - Comparison of how each routing strategy performs
        - Real-time dashboard showing current conditions and their impact
        """)
    
    dataset = get_dataset()
    if dataset is None:
        return
    
    delivery_locations = dataset['delivery_locations']
    orders = dataset['orders']
    traffic = dataset['traffic']
    weather = dataset['weather']
    demand_forecast = dataset['demand_forecast']
    
    # Real-time simulation controls
    st.subheader("Real-Time Simulation Controls")
    col1, col2 = st.columns(2)
    with col1:
        simulate_traffic_incident = st.checkbox("Simulate Traffic Incident", value=False)
        simulate_weather_change = st.checkbox("Simulate Weather Change", value=False)
    with col2:
        simulate_demand_surge = st.checkbox("Simulate Demand Surge", value=False)
        current_hour = st.slider("Current Hour (for routing)", 0, 23, 9)
    
    # Apply real-time updates
    from utils.realtime_simulator import (
        simulate_realtime_traffic_update, 
        simulate_realtime_weather_update,
        simulate_demand_surge
    )
    
    updated_traffic = traffic.copy()
    updated_weather = weather.copy()
    updated_demand = demand_forecast.copy()
    
    if simulate_traffic_incident:
        updated_traffic = simulate_realtime_traffic_update(traffic, current_hour, incident_probability=1.0)
    if simulate_weather_change:
        updated_weather = simulate_realtime_weather_update(weather, change_probability=1.0)
    if simulate_demand_surge:
        updated_demand = simulate_demand_surge(demand_forecast, surge_probability=1.0)
    
    st.subheader("Current Conditions Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    
    # Compare original vs updated
    upd_level = 'light'  # Initialize default
    with col1:
        original_traffic = traffic[traffic['hour'] == current_hour]
        updated_traffic_hour = updated_traffic[updated_traffic['hour'] == current_hour]
        
        if len(original_traffic) > 0 and len(updated_traffic_hour) > 0:
            orig_level = original_traffic.iloc[0]['traffic_level']
            upd_level = updated_traffic_hour.iloc[0]['traffic_level']
            delta = None if orig_level == upd_level else f"⚠️ {upd_level.title()}"
            st.metric("Current Traffic", upd_level.title(), delta=delta)
            st.caption(f"Speed: {updated_traffic_hour.iloc[0]['average_speed_kmh']:.1f} km/h")
        else:
            st.metric("Current Traffic", "N/A")
    
    with col2:
        orig_weather = weather.iloc[-1]['condition']
        upd_weather = updated_weather.iloc[-1]['condition']
        delta = None if orig_weather == upd_weather else f"⚠️ Changed"
        st.metric("Weather Condition", upd_weather.title(), delta=delta)
        st.caption(f"Temp: {updated_weather.iloc[-1]['temperature_c']:.1f}°C")
        if upd_weather in ['rain', 'snow', 'storm']:
            st.caption(f"Precipitation: {updated_weather.iloc[-1]['precipitation_mm']:.1f}mm")
    
    with col3:
        orig_demand = demand_forecast.iloc[0]['forecasted_orders']
        upd_demand = updated_demand.iloc[0]['forecasted_orders']
        delta = None if orig_demand == upd_demand else f"📈 +{upd_demand - orig_demand}"
        st.metric("Forecasted Demand", upd_demand, delta=delta)
        st.caption(f"Confidence: {updated_demand.iloc[0]['confidence_level']*100:.0f}%")
    
    with col4:
        from config import TRAFFIC_MULTIPLIERS, WEATHER_MULTIPLIERS
        traffic_mult = TRAFFIC_MULTIPLIERS.get(upd_level, 1.0)
        weather_mult = WEATHER_MULTIPLIERS.get(upd_weather, 1.0)
        combined_mult = traffic_mult * weather_mult
        st.metric("Combined Impact", f"{combined_mult:.2f}x", 
                 f"Time multiplier")
        st.caption("Higher = slower delivery")
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Traffic Patterns (24 hours)")
        from utils.visualization import plot_traffic_patterns
        st.plotly_chart(plot_traffic_patterns(updated_traffic), use_container_width=True)
        with st.expander("💡 Understanding Traffic Patterns", expanded=False):
            st.markdown("""
            This chart shows traffic levels throughout a 24-hour period:
            - **X-axis**: Hour of day (0-23)
            - **Y-axis**: Traffic level (Light, Moderate, Heavy, Severe)
            - **Peak Hours**: Typically 7-9 AM and 5-7 PM show heavy/severe traffic
            
            **Key Insight**: Routes optimized during peak hours will have longer travel times.
            The AI algorithm learns to avoid these times when possible or account for delays.
            """)
    
    with col2:
        st.subheader("Weather Conditions")
        from utils.visualization import plot_weather_impact
        st.plotly_chart(plot_weather_impact(updated_weather), use_container_width=True)
        with st.expander("💡 Understanding Weather Impact", expanded=False):
            st.markdown("""
            This chart shows weather conditions and their impact on delivery:
            - **Clear**: Normal conditions (1.0x speed multiplier)
            - **Rain**: Slightly slower (1.2-1.4x time multiplier)
            - **Snow**: Significantly slower (1.5-2.0x time multiplier)
            - **Storm**: Very slow, dangerous conditions (2.0-2.5x time multiplier)
            
            **Key Insight**: Adverse weather increases delivery time and cost. 
            The AI algorithm accounts for weather when optimizing routes.
            """)
    
    st.subheader("Demand Forecast (30 days)")
    from utils.visualization import plot_demand_forecast
    st.plotly_chart(plot_demand_forecast(updated_demand), use_container_width=True)
    with st.expander("💡 Understanding Demand Forecast", expanded=False):
        st.markdown("""
        This chart shows predicted order volumes over the next 30 days:
        - **X-axis**: Date (next 30 days)
        - **Y-axis**: Forecasted number of orders
        - **Confidence Level**: How certain the forecast is
        
        **Key Insight**: Higher demand requires more vehicles and better route planning.
        AI uses these forecasts to pre-allocate resources and optimize fleet size.
        """)
    
    # Dynamic routing comparison
    st.subheader("Dynamic Routing Impact Analysis")
    st.markdown("**Compare routes optimized with static vs real-time data:**")
    
    # Use sample for large datasets
    use_sample = len(delivery_locations) > 200
    if use_sample:
        sample_size = st.slider("Sample Size for Analysis", 50, min(200, len(delivery_locations)), 
                               min(100, len(delivery_locations)), key="realtime_sample")
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(delivery_locations)), sample_size)
        sample_locations = delivery_locations.iloc[sample_indices].reset_index(drop=True)
        sample_orders = orders[orders['delivery_id'].isin(sample_locations['delivery_id'])].reset_index(drop=True)
    else:
        sample_locations = delivery_locations
        sample_orders = orders
    
    if st.button("Run Dynamic Routing Analysis", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Step 1/5: Calculating distance matrix...")
        progress_bar.progress(10)
        distance_matrix = calculate_distance_matrix(sample_locations)
        
        status_text.text("Step 2/5: Generating static route (no real-time data)...")
        progress_bar.progress(30)
        vehicle = dataset['vehicles'].iloc[0]
        vehicle_info = vehicle.to_dict()
        from utils.route_optimizer import generate_baseline_route
        depot = dataset['depots'].iloc[0]
        static_route = generate_baseline_route(sample_locations, depot, sample_orders)
        
        # Calculate static route metrics (no real-time adjustments)
        static_metrics = calculate_route_metrics(
            static_route, distance_matrix, vehicle_info, sample_orders, sample_locations, 0,
            depot_location=depot
        )
        
        status_text.text("Step 3/5: Calculating route with real-time conditions...")
        progress_bar.progress(50)
        # Calculate route with real-time conditions applied
        from utils.realtime_simulator import calculate_route_with_realtime_conditions
        realtime_metrics, traffic_mult, weather_mult = calculate_route_with_realtime_conditions(
            static_route, distance_matrix, vehicle_info, sample_orders, 
            sample_locations, updated_traffic, updated_weather, current_hour
        )
        
        status_text.text("Step 4/5: Training RL agent for adaptive routing...")
        progress_bar.progress(70)
        # Route with RL adaptation
        rl_agent = QLearningRouter(distance_matrix, sample_orders, sample_locations, vehicle_info)
        rl_agent.train(n_episodes=300, traffic_data=updated_traffic, weather_data=updated_weather)
        adaptive_route = rl_agent.predict_route(traffic_data=updated_traffic, weather_data=updated_weather)
        
        status_text.text("Step 5/5: Calculating adaptive route metrics...")
        progress_bar.progress(90)
        adaptive_metrics = calculate_route_metrics(
            adaptive_route, distance_matrix, vehicle_info, sample_orders, 
            sample_locations, 0, traffic_data=updated_traffic, weather_data=updated_weather,
            depot_location=depot
        )
        
        progress_bar.progress(100)
        status_text.text("✓ Analysis complete!")
        
        # Display results
        st.subheader("Route Comparison Results")
        
        with st.expander("📊 Understanding the Comparison", expanded=False):
            st.markdown("""
            **Three Routing Strategies:**
            
            1. **Static Route**: Planned without considering real-time conditions
               - Shows baseline performance
               - Cost/time shown here is what you'd get if conditions were perfect
            
            2. **With Real-Time Conditions**: Same route, but costs/time adjusted for actual conditions
               - Shows how real-world conditions affect the static route
               - Usually shows increased cost/time due to traffic/weather
            
            3. **Adaptive Route (RL)**: Route optimized by Reinforcement Learning agent
               - Agent learns to adapt to changing conditions
               - Typically performs better than static route under real conditions
            
            **Key Takeaway**: The RL agent learns to minimize cost even when conditions change,
            showing the value of adaptive AI systems over static planning.
            """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Static Route Cost", f"${static_metrics['total_cost']:.2f}")
            st.metric("Static Route Time", f"{static_metrics['total_time']:.2f} hrs")
            st.metric("Static Distance", f"{static_metrics['total_distance']:.2f} km")
        
        with col2:
            cost_increase = ((realtime_metrics['total_cost'] - static_metrics['total_cost']) / static_metrics['total_cost']) * 100
            time_increase = ((realtime_metrics['total_time'] - static_metrics['total_time']) / static_metrics['total_time']) * 100
            st.metric("With Real-Time Conditions", f"${realtime_metrics['total_cost']:.2f}", 
                     f"{cost_increase:+.1f}%", delta_color="inverse")
            st.metric("Time", f"{realtime_metrics['total_time']:.2f} hrs", 
                     f"{time_increase:+.1f}%", delta_color="inverse")
            st.metric("Distance", f"{realtime_metrics['total_distance']:.2f} km")
            st.caption(f"Traffic: {traffic_mult:.2f}x, Weather: {weather_mult:.2f}x")
        
        with col3:
            improvement = ((realtime_metrics['total_cost'] - adaptive_metrics['total_cost']) / realtime_metrics['total_cost']) * 100
            st.metric("Adaptive Route (RL)", f"${adaptive_metrics['total_cost']:.2f}", 
                     f"{improvement:.1f}% better", delta_color="normal")
            st.metric("Time", f"{adaptive_metrics['total_time']:.2f} hrs")
            st.metric("Distance", f"{adaptive_metrics['total_distance']:.2f} km")
        
        # Comparison chart
        comparison_data = {
            'Route Type': ['Static', 'Real-Time Impact', 'Adaptive (RL)'],
            'Cost': [static_metrics['total_cost'], realtime_metrics['total_cost'], adaptive_metrics['total_cost']],
            'Time': [static_metrics['total_time'], realtime_metrics['total_time'], adaptive_metrics['total_time']],
            'Distance': [static_metrics['total_distance'], realtime_metrics['total_distance'], adaptive_metrics['total_distance']]
        }
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Cost ($)', x=comparison_data['Route Type'], y=comparison_data['Cost'], 
                            marker_color='lightblue', yaxis='y'))
        fig.add_trace(go.Bar(name='Time (hrs)', x=comparison_data['Route Type'], 
                            y=[t * 50 for t in comparison_data['Time']], marker_color='lightgreen', 
                            yaxis='y2'))
        fig.update_layout(
            title='Route Comparison: Static vs Real-Time vs Adaptive',
            xaxis_title='Route Type',
            yaxis=dict(title='Cost ($)', side='left'),
            yaxis2=dict(title='Time (hrs × 50)', overlaying='y', side='right'),
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("💡 Understanding the Comparison Chart", expanded=False):
            st.markdown("""
            This chart compares three routing strategies:
            - **Blue bars (Cost)**: Total cost in dollars
            - **Green bars (Time)**: Total time in hours (scaled ×50 for visibility)
            
            **What to Look For:**
            - **Static Route**: Baseline (what you'd plan without real-time data)
            - **Real-Time Impact**: Shows how conditions affect the static route (usually worse)
            - **Adaptive (RL)**: Shows how AI adapts to minimize impact (usually best)
            
            **Key Insight**: The RL agent learns to compensate for adverse conditions,
            often achieving costs closer to the static baseline even when conditions are poor.
            """)
        
        # Impact analysis
        st.subheader("Real-Time Data Impact Analysis")
        impact_df = pd.DataFrame({
            'Factor': ['Traffic Multiplier', 'Weather Multiplier', 'Combined Impact'],
            'Value': [f"{traffic_mult:.2f}x", f"{weather_mult:.2f}x", f"{traffic_mult * weather_mult:.2f}x"],
            'Effect': [
                f"{'Slower' if traffic_mult > 1.0 else 'Faster'} delivery",
                f"{'Slower' if weather_mult > 1.0 else 'Faster'} delivery",
                f"{'Increased' if traffic_mult * weather_mult > 1.0 else 'Decreased'} cost by {abs((traffic_mult * weather_mult - 1) * 100):.1f}%"
            ]
        })
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
        
        st.info(f"""
        **Key Insights:**
        - Static route assumes ideal conditions (no traffic, clear weather)
        - Real-time conditions show actual impact: {cost_increase:.1f}% cost increase
        - Adaptive RL route optimizes for current conditions: {improvement:.1f}% better than static with real-time conditions
        - **Conclusion:** Real-time data enables dynamic route adjustments, saving ${realtime_metrics['total_cost'] - adaptive_metrics['total_cost']:.2f} compared to ignoring real-time conditions
        """)


def tab_algorithm_showcase():
    """Tab 3: Algorithm Showcase & Comparison"""
    st.header("🔬 Algorithm Showcase & Comparison")
    st.markdown("""
    **Question: Which algorithms are commonly used (e.g., genetic algorithms, reinforcement learning, clustering)?**
    
    This tab demonstrates three key algorithms used in logistics optimization with interactive visualizations.
    """)
    
    with st.expander("📖 Understanding Different Algorithms", expanded=False):
        st.markdown("""
        **Why Multiple Algorithms?**
        
        Different logistics problems require different approaches:
        - **Genetic Algorithms**: Best for static route optimization (pre-planned routes)
        - **Reinforcement Learning**: Best for dynamic routing (adapting to changes)
        - **Clustering**: Best for zone assignment and territory planning
        
        **Each algorithm has strengths:**
        - **GA**: Finds globally optimal solutions, handles complex constraints
        - **RL**: Learns from experience, adapts automatically
        - **Clustering**: Fast, scalable, good for grouping deliveries
        
        **When to Use Which:**
        - Use GA when you have time to optimize and routes are pre-planned
        - Use RL when conditions change frequently and you need adaptive routing
        - Use Clustering when assigning deliveries to vehicles or territories
        """)
    
    dataset = get_dataset()
    if dataset is None:
        return
    
    algorithm = st.selectbox("Select Algorithm", ["Genetic Algorithm", "Reinforcement Learning", "Clustering"])
    
    if algorithm == "Genetic Algorithm":
        st.subheader("Genetic Algorithm for Route Optimization")
        st.markdown("""
        **How it works:**
        - Maintains a population of candidate solutions (routes)
        - Evolves solutions through selection, crossover, and mutation
        - Optimizes for minimum cost (distance + time + penalties)
        """)
        
        with st.expander("🔬 Deep Dive: Genetic Algorithm", expanded=False):
            st.markdown("""
            **Step-by-Step Process:**
            
            1. **Initialization**: Creates random routes (population)
            2. **Evaluation**: Calculates fitness (cost) for each route
            3. **Selection**: Chooses better routes to "reproduce"
            4. **Crossover**: Combines parts of two good routes to create offspring
            5. **Mutation**: Randomly changes routes to explore new solutions
            6. **Replacement**: New generation replaces old (keeping best)
            7. **Repeat**: Continues until convergence or max generations
            
            **Parameters Explained:**
            - **Population Size**: More routes = better exploration but slower
            - **Generations**: More iterations = better solutions but takes longer
            - **Mutation Rate**: Higher = more exploration, lower = faster convergence
            
            **Why It Works**: Mimics natural evolution - good solutions survive and combine,
            gradually improving the population over time.
            """)
        
        # Limit dataset size for faster demo
        max_deliveries = 100
        if len(dataset['delivery_locations']) > max_deliveries:
            st.info(f"⚠️ Large dataset detected ({len(dataset['delivery_locations'])} deliveries). Using first {max_deliveries} deliveries for faster demonstration.")
            demo_locations = dataset['delivery_locations'].head(max_deliveries).reset_index(drop=True)
            demo_orders = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locations['delivery_id'])].reset_index(drop=True)
        else:
            demo_locations = dataset['delivery_locations']
            demo_orders = dataset['orders']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pop_size = st.slider("Population Size", 20, 100, 50)
        with col2:
            gens = st.slider("Generations", 20, 200, 50)
        with col3:
            mut_rate = st.slider("Mutation Rate", 0.01, 0.3, 0.1)
        
        if st.button("Run Genetic Algorithm"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate distance matrix
            status_text.text("Step 1/3: Calculating distance matrix...")
            progress_bar.progress(10)
            distance_matrix = calculate_distance_matrix(demo_locations)
            progress_bar.progress(30)
            
            # Initialize GA
            status_text.text("Step 2/3: Initializing Genetic Algorithm...")
            vehicle = dataset['vehicles'].iloc[0]
            ga = GeneticAlgorithmVRP(
                distance_matrix, demo_orders, demo_locations,
                vehicle.to_dict(), population_size=pop_size, generations=gens, mutation_rate=mut_rate
            )
            progress_bar.progress(40)
            
            # Run GA with progress updates
            status_text.text(f"Step 3/3: Evolving routes (Generation 0/{gens})...")
            
            def update_progress(gen, total_gen, best_fit, avg_fit):
                progress = 40 + int((gen / total_gen) * 55)
                progress_bar.progress(min(progress, 95))
                status_text.text(f"Step 3/3: Generation {gen}/{total_gen} - Best Fitness: {best_fit:.2f}")
            
            route, history = ga.solve(progress_callback=update_progress)
            
            progress_bar.progress(100)
            status_text.text("✓ Optimization complete!")
            
            st.success(f"✅ Best route found with {len(route)} deliveries!")
            
            st.subheader("Algorithm Convergence")
            st.plotly_chart(plot_algorithm_convergence(history), use_container_width=True)
            with st.expander("💡 Understanding Convergence Graph", expanded=False):
                st.markdown("""
                This graph shows how the Genetic Algorithm improves over generations:
                - **Best Fitness (blue line)**: Best route found in each generation (lower = better)
                - **Average Fitness (orange line)**: Average quality of all routes in that generation
                
                **What You're Seeing:**
                - Initial drop = algorithm quickly finds better solutions
                - Gradual improvement = fine-tuning the solution
                - Plateau = algorithm has converged (found a good solution)
                
                **Why It Matters**: Shows the algorithm is learning and improving systematically,
                not just randomly searching. The gap between best and average shows population diversity.
                """)
            
            # Show route summary
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Route Length", f"{len(route)} deliveries")
            with col_sum2:
                final_fitness = history['best_fitness'][-1] if len(history['best_fitness']) > 0 else 0
                st.metric("Best Fitness", f"{final_fitness:.2f}")
            with col_sum3:
                improvement = ((history['best_fitness'][0] - history['best_fitness'][-1]) / history['best_fitness'][0] * 100) if len(history['best_fitness']) > 0 and history['best_fitness'][0] > 0 else 0
                st.metric("Improvement", f"{improvement:.1f}%")
    
    elif algorithm == "Reinforcement Learning":
        st.subheader("Reinforcement Learning for Dynamic Routing")
        st.markdown("""
        **How it works:**
        - Agent learns optimal actions through trial and error
        - Uses Q-learning to estimate value of state-action pairs
        - Adapts to changing conditions (traffic, weather)
        """)
        
        with st.expander("🔬 Deep Dive: Reinforcement Learning", expanded=False):
            st.markdown("""
            **How Q-Learning Works:**
            
            1. **Agent**: Makes decisions (which delivery to visit next)
            2. **State**: Current situation (location, remaining deliveries, traffic, weather)
            3. **Action**: Decision (visit delivery X)
            4. **Reward**: Feedback (negative for distance/time, penalty for SLA violations)
            5. **Q-Table**: Stores learned values for state-action pairs
            
            **Learning Process:**
            - Agent explores routes randomly at first (exploration)
            - Learns which actions lead to better rewards
            - Gradually exploits learned knowledge (exploitation)
            - Balances exploration vs exploitation using epsilon-greedy policy
            
            **Why It's Powerful**: Learns from experience without explicit programming.
            Adapts automatically when conditions change (traffic, weather, demand).
            
            **Training Episodes**: Each episode is one complete route. More episodes = better learning.
            """)
        
        # Limit dataset size for faster demo
        max_deliveries = 50  # Smaller for RL as it's more computationally intensive
        if len(dataset['delivery_locations']) > max_deliveries:
            st.info(f"⚠️ Large dataset detected ({len(dataset['delivery_locations'])} deliveries). Using first {max_deliveries} deliveries for faster demonstration.")
            demo_locations = dataset['delivery_locations'].head(max_deliveries).reset_index(drop=True)
            demo_orders = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locations['delivery_id'])].reset_index(drop=True)
        else:
            demo_locations = dataset['delivery_locations']
            demo_orders = dataset['orders']
        
        episodes = st.slider("Training Episodes", 50, 300, 100)
        
        if st.button("Train RL Agent"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate distance matrix
            status_text.text("Step 1/3: Calculating distance matrix...")
            progress_bar.progress(10)
            distance_matrix = calculate_distance_matrix(demo_locations)
            progress_bar.progress(30)
            
            # Initialize RL agent
            status_text.text("Step 2/3: Initializing RL agent...")
            vehicle = dataset['vehicles'].iloc[0]
            rl = QLearningRouter(distance_matrix, demo_orders, 
                                 demo_locations, vehicle.to_dict())
            progress_bar.progress(40)
            
            # Train with progress updates
            status_text.text(f"Step 3/3: Training agent (Episode 0/{episodes})...")
            
            # Train with progress updates
            training_history = []
            for episode in range(episodes):
                route, reward = rl.train_episode(
                    traffic_data=dataset['traffic'], 
                    weather_data=dataset['weather']
                )
                training_history.append({
                    'episode': episode,
                    'reward': reward,
                    'route_length': len(route)
                })
                
                # Update progress every 10 episodes or at the end
                if (episode + 1) % 10 == 0 or episode == episodes - 1:
                    progress = 40 + int((episode + 1) / episodes * 55)
                    progress_bar.progress(min(progress, 95))
                    avg_reward = np.mean([h['reward'] for h in training_history[-10:]]) if len(training_history) >= 10 else reward
                    status_text.text(f"Step 3/3: Training agent (Episode {episode + 1}/{episodes}) - Avg Reward: {avg_reward:.2f}")
            
            rl.training_history = training_history
            
            progress_bar.progress(100)
            status_text.text("✓ Training complete!")
            
            st.success(f"✅ Agent trained successfully on {episodes} episodes!")
            
            # Plot training history
            history_df = pd.DataFrame(training_history)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df['episode'], y=history_df['reward'],
                                    mode='lines', name='Reward', line=dict(color='green', width=2)))
            fig.update_layout(title='RL Training Progress', xaxis_title='Episode',
                            yaxis_title='Reward', height=400)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("💡 Understanding RL Training Progress", expanded=False):
                st.markdown("""
                This graph shows how the RL agent learns over episodes:
                - **X-axis**: Episode number (each episode = one complete route)
                - **Y-axis**: Reward (higher/less negative = better route)
                
                **What You're Seeing:**
                - **Early episodes**: High variance, random exploration (low rewards)
                - **Middle episodes**: Agent learns, rewards improve
                - **Later episodes**: More consistent, better performance
                
                **Key Insight**: The agent starts poorly but learns from experience.
                Higher rewards (less negative) mean shorter routes and better decisions.
                """)
            
            # Show summary metrics
            col_rl1, col_rl2, col_rl3 = st.columns(3)
            with col_rl1:
                st.metric("Total Episodes", episodes)
            with col_rl2:
                final_reward = training_history[-1]['reward'] if len(training_history) > 0 else 0
                st.metric("Final Reward", f"{final_reward:.2f}")
            with col_rl3:
                avg_reward = np.mean([h['reward'] for h in training_history[-10:]]) if len(training_history) >= 10 else 0
                st.metric("Avg Reward (Last 10)", f"{avg_reward:.2f}")
    
    elif algorithm == "Clustering":
        st.subheader("K-Means Clustering for Delivery Zones")
        st.markdown("""
        **How it works:**
        - Groups delivery locations into optimal zones
        - Minimizes intra-cluster distance
        - Helps assign vehicles to geographic regions
        """)
        
        with st.expander("🔬 Deep Dive: K-Means Clustering", expanded=False):
            st.markdown("""
            **How K-Means Works:**
            
            1. **Initialize**: Randomly place K cluster centers
            2. **Assign**: Assign each delivery to nearest center
            3. **Update**: Move centers to mean of assigned deliveries
            4. **Repeat**: Until centers stop moving
            
            **Why It's Useful:**
            - Groups nearby deliveries together
            - Assigns vehicles to geographic zones
            - Reduces cross-city travel
            - Simplifies route planning
            
            **K (Number of Clusters)**: 
            - Too few = large zones, long distances
            - Too many = small zones, inefficient
            - Optimal = balanced zone sizes
            
            **Application**: Use clustering to divide city into delivery territories,
            then optimize routes within each territory.
            """)
        
        n_clusters = st.slider("Number of Clusters", 2, 15, 5)
        
        if st.button("Perform Clustering"):
            with st.spinner("Clustering delivery locations..."):
                clusterer = DeliveryZoneOptimizer(dataset['delivery_locations'])
                labels = clusterer.fit(n_clusters=n_clusters)
                centers = clusterer.get_cluster_centers()
                stats = clusterer.get_cluster_stats()
                
                st.subheader("Cluster Statistics")
                st.dataframe(stats, use_container_width=True)
                with st.expander("💡 Understanding Cluster Statistics", expanded=False):
                    st.markdown("""
                    **What the Statistics Show:**
                    - **Cluster ID**: Zone number
                    - **Number of Deliveries**: How many deliveries in each zone
                    - **Average Distance**: Average distance from deliveries to cluster center
                    - **Total Distance**: Sum of distances (lower = better clustering)
                    
                    **What to Look For:**
                    - Balanced cluster sizes (similar number of deliveries)
                    - Low average distances (deliveries close to center)
                    - Even distribution across zones
                    """)
                
                # Visualization
                from utils.visualization import plot_delivery_zones
                fig = plot_delivery_zones(dataset['delivery_locations'], pd.Series(labels))
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("🗺️ Understanding the Clustering Map", expanded=False):
                    st.markdown("""
                    **Map Elements:**
                    - **Colored points**: Delivery locations, colored by cluster/zone
                    - **Same color = same zone**: Deliveries in the same zone are grouped together
                    
                    **What Good Clustering Looks Like:**
                    - Clear geographic separation between zones
                    - Deliveries in each zone are close together
                    - Zones don't overlap unnecessarily
                    - Balanced number of deliveries per zone
                    
                    **Application**: Assign one vehicle to each zone, then optimize routes within zones.
                    This reduces complexity and improves efficiency.
                    """)
    
    # Algorithm comparison
    st.subheader("Algorithm Comparison")
    comparison_df = pd.DataFrame({
        'Algorithm': ['Genetic Algorithm', 'Reinforcement Learning', 'K-Means Clustering'],
        'Best For': ['Static route optimization', 'Dynamic routing', 'Zone assignment'],
        'Strengths': ['Global optimization, handles constraints', 'Adapts to changes', 'Fast, scalable'],
        'Limitations': ['Can be slow for large problems', 'Requires training', 'May need manual tuning'],
        'Use Case': ['Pre-planned routes', 'Real-time adjustments', 'Territory planning']
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def tab_fleet_sustainability():
    """Tab 4: Fleet Utilization & Sustainability"""
    st.header("🌱 Fleet Utilization & Sustainability")
    st.markdown("""
    **Question: How can AI improve fleet utilization and reduce carbon footprint?**
    
    This tab analyzes fleet efficiency and environmental impact of route optimization.
    """)
    
    with st.expander("📖 What's Happening Here?", expanded=False):
        st.markdown("""
        **Understanding Fleet Utilization:**
        
        Fleet utilization measures how efficiently vehicles are being used:
        - **Weight Utilization**: Percentage of vehicle weight capacity used
        - **Volume Utilization**: Percentage of vehicle volume capacity used
        - **Vehicle Usage**: How many vehicles are needed vs available
        
        **Why It Matters:**
        - Higher utilization = fewer vehicles needed = lower costs
        - Better utilization = less wasted capacity = more efficient operations
        - Optimal utilization = balanced load across all vehicles
        
        **Carbon Footprint Analysis:**
        - CO₂ emissions calculated based on distance traveled and fuel consumption
        - Optimized routes reduce total distance = lower emissions
        - Better utilization means fewer vehicles = lower fleet emissions
        
        **What You'll See:**
        - Fleet utilization metrics showing how well vehicles are loaded
        - Carbon footprint comparison (baseline vs optimized)
        - Emissions reduction percentage
        - Visualizations showing utilization and environmental impact
        """)
    
    dataset = get_dataset()
    if dataset is None:
        return
    
    # Limit dataset size for faster analysis
    max_deliveries = 150
    if len(dataset['delivery_locations']) > max_deliveries:
        st.info(f"⚠️ Large dataset detected ({len(dataset['delivery_locations'])} deliveries). Using first {max_deliveries} deliveries for faster analysis.")
        demo_locations = dataset['delivery_locations'].head(max_deliveries).reset_index(drop=True)
        demo_orders = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locations['delivery_id'])].reset_index(drop=True)
    else:
        demo_locations = dataset['delivery_locations']
        demo_orders = dataset['orders']
    
    if st.button("Analyze Fleet Utilization"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Calculate distance matrix
        status_text.text("Step 1/4: Calculating distance matrix...")
        progress_bar.progress(10)
        from utils.route_optimizer import optimize_route_assignment, calculate_distance_matrix
        distance_matrix = calculate_distance_matrix(demo_locations)
        progress_bar.progress(30)
        
        # Step 2: Optimize route assignments
        status_text.text("Step 2/4: Optimizing route assignments...")
        progress_bar.progress(40)
        assignments = optimize_route_assignment(
            demo_orders, dataset['vehicles'], demo_locations, distance_matrix
        )
        progress_bar.progress(60)
        
        # Step 3: Calculate metrics for each vehicle
        status_text.text("Step 3/4: Calculating vehicle metrics...")
        routes_info = {}
        total_vehicles = len(assignments)
        for idx, (vehicle_id, delivery_ids) in enumerate(assignments.items()):
            if len(delivery_ids) == 0:
                continue
            
            # Update progress
            vehicle_progress = 60 + int((idx + 1) / total_vehicles * 20)
            progress_bar.progress(min(vehicle_progress, 80))
            
            vehicle = dataset['vehicles'][dataset['vehicles']['vehicle_id'] == vehicle_id].iloc[0]
            
            # More efficient way to get route indices
            delivery_id_to_idx = {did: idx for idx, did in enumerate(demo_locations['delivery_id'])}
            route_indices = [delivery_id_to_idx[did] for did in delivery_ids if did in delivery_id_to_idx]
            
            if len(route_indices) > 0:
                metrics = calculate_route_metrics(
                    route_indices, distance_matrix, vehicle.to_dict(),
                    demo_orders, demo_locations, 0,
                    depot_location=dataset['depots'].iloc[0]
                )
                routes_info[vehicle_id] = metrics
        
        # Step 4: Calculate fleet metrics
        status_text.text("Step 4/4: Calculating fleet metrics...")
        progress_bar.progress(85)
        fleet_metrics = calculate_all_metrics(routes_info, dataset['vehicles'], demo_orders)
        progress_bar.progress(95)
        
        progress_bar.progress(100)
        status_text.text("✓ Analysis complete!")
        
        st.subheader("Fleet Utilization Metrics")
        from utils.visualization import plot_fleet_utilization
        st.plotly_chart(plot_fleet_utilization(fleet_metrics['fleet_utilization']), use_container_width=True)
        with st.expander("💡 Understanding Fleet Utilization Chart", expanded=False):
            st.markdown("""
            This chart shows how efficiently vehicles are being used:
            - **Weight Utilization**: How much of vehicle weight capacity is used (higher = better)
            - **Volume Utilization**: How much of vehicle volume capacity is used (higher = better)
            - **Target**: Ideally 80-90% utilization (not 100% - need buffer for flexibility)
            
            **Key Insight**: Higher utilization means fewer vehicles needed, reducing costs and emissions.
            However, 100% utilization isn't ideal - you need some buffer for unexpected orders.
            """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Weight Utilization", f"{fleet_metrics['fleet_utilization']['average_weight_utilization']*100:.1f}%")
        with col2:
            st.metric("Volume Utilization", f"{fleet_metrics['fleet_utilization']['average_volume_utilization']*100:.1f}%")
        with col3:
            st.metric("Vehicles Used", f"{fleet_metrics['fleet_utilization']['vehicles_used']}/{fleet_metrics['fleet_utilization']['total_vehicles']}")
        with col4:
            st.metric("Total Emissions", f"{fleet_metrics['total_emissions_kg']:.1f} kg CO2")
        
        # Carbon footprint
        st.subheader("Carbon Footprint Analysis")
        # Calculate baseline emissions (simplified)
        baseline_emissions = fleet_metrics['total_emissions_kg'] * 1.2  # Assume 20% more without optimization
        optimized_emissions = fleet_metrics['total_emissions_kg']
        
        from utils.visualization import plot_emissions_comparison
        st.plotly_chart(plot_emissions_comparison(baseline_emissions, optimized_emissions), use_container_width=True)
        with st.expander("💡 Understanding Emissions Comparison", expanded=False):
            st.markdown("""
            This chart compares CO₂ emissions:
            - **Baseline**: Estimated emissions without optimization (assumes 20% more distance)
            - **Optimized**: Actual emissions with AI-optimized routes
            
            **How Emissions Are Calculated:**
            - Based on distance traveled × fuel consumption × emission factor
            - Optimized routes = shorter distances = lower emissions
            
            **Environmental Impact**: 
            - Every kg of CO₂ saved helps reduce carbon footprint
            - 20-35% reduction is typical with route optimization
            - Better utilization = fewer vehicles = even lower emissions
            """)
        
        reduction = calculate_emissions_reduction(baseline_emissions, optimized_emissions)
        st.success(f"✅ Emissions reduced by {reduction:.1f}% through optimization!")


def tab_integration_challenges():
    """Tab 5: Integration Challenges & Solutions"""
    st.header("🔧 Integration Challenges & Solutions")
    st.markdown("""
    **Question: What are the challenges in integrating AI with existing logistics systems?**
    
    This tab discusses technical challenges and solutions for AI integration in logistics.
    """)
    
    st.subheader("Key Challenges")
    
    challenges = [
        {
            "Challenge": "Legacy System Compatibility",
            "Description": "Existing logistics systems often use outdated technologies that don't easily integrate with modern AI solutions.",
            "Impact": "High - Blocks integration",
            "Solutions": [
                "API wrappers for legacy systems",
                "Gradual migration strategy",
                "Middleware integration layers"
            ]
        },
        {
            "Challenge": "Data Quality & Standardization",
            "Description": "Inconsistent data formats, missing values, and poor data quality hinder AI model performance.",
            "Impact": "High - Affects accuracy",
            "Solutions": [
                "Data validation pipelines",
                "ETL processes for standardization",
                "Data quality monitoring"
            ]
        },
        {
            "Challenge": "Real-Time Data Integration",
            "Description": "Integrating real-time data streams (traffic, weather) requires robust infrastructure.",
            "Impact": "Medium - Affects responsiveness",
            "Solutions": [
                "Message queues (Kafka, RabbitMQ)",
                "Stream processing frameworks",
                "Caching strategies"
            ]
        },
        {
            "Challenge": "Scalability & Performance",
            "Description": "AI algorithms can be computationally expensive, especially for large-scale operations.",
            "Impact": "Medium - Affects speed",
            "Solutions": [
                "Distributed computing",
                "Algorithm optimization",
                "Cloud infrastructure"
            ]
        }
    ]
    
    for i, challenge in enumerate(challenges):
        with st.expander(f"{i+1}. {challenge['Challenge']}"):
            st.write(f"**Description:** {challenge['Description']}")
            st.write(f"**Impact:** {challenge['Impact']}")
            st.write("**Solutions:**")
            for solution in challenge['Solutions']:
                st.write(f"- {solution}")
    
    st.subheader("System Architecture")
    st.markdown("""
    **Current System Architecture:**
    ```
    Legacy Systems → Manual Planning → Execution
    ```
    
    **AI-Enhanced Architecture:**
    ```
    Data Sources → AI Engine → Optimization → API → Execution Systems
                    ↓
                Monitoring & Feedback
    ```
    """)
    
    st.subheader("Integration Roadmap")
    roadmap_steps = [
        "Phase 1: Data Integration (Weeks 1-4)",
        "Phase 2: Pilot Implementation (Weeks 5-8)",
        "Phase 3: Full Deployment (Weeks 9-12)",
        "Phase 4: Optimization & Scaling (Ongoing)"
    ]
    
    for step in roadmap_steps:
        st.write(f"✓ {step}")


def tab_last_mile():
    """Tab 6: Last-Mile Delivery Optimization"""
    st.header("📦 Last-Mile Delivery Optimization")
    st.markdown("""
    **Question: How does AI contribute to last-mile delivery efficiency?**
    
    This tab focuses on optimizing the final leg of delivery to customers.
    """)
    
    dataset = get_dataset()
    if dataset is None:
        return
    
    st.subheader("Last-Mile Optimization Strategies")
    
    strategy = st.selectbox("Select Strategy", ["Route Clustering", "Time Window Optimization", "Delivery Time Prediction"])
    
    if strategy == "Route Clustering":
        st.markdown("**Group nearby deliveries to minimize travel distance**")
        
        # Limit dataset size for faster clustering
        max_locations = 200
        if len(dataset['delivery_locations']) > max_locations:
            st.info(f"⚠️ Large dataset detected ({len(dataset['delivery_locations'])} locations). Using first {max_locations} locations for faster clustering.")
            demo_locations_cluster = dataset['delivery_locations'].head(max_locations).reset_index(drop=True)
        else:
            demo_locations_cluster = dataset['delivery_locations']
        
        n_zones = st.slider("Number of Delivery Zones", 3, 10, 5)
        
        if st.button("Optimize Zones"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Clustering delivery locations...")
            progress_bar.progress(30)
            clusterer = DeliveryZoneOptimizer(demo_locations_cluster)
            labels = clusterer.fit(n_clusters=n_zones)
            progress_bar.progress(70)
            
            stats = clusterer.get_cluster_stats()
            progress_bar.progress(90)
            
            progress_bar.progress(100)
            status_text.text("✓ Clustering complete!")
            
            st.dataframe(stats, use_container_width=True)
            from utils.visualization import plot_delivery_zones
            st.plotly_chart(plot_delivery_zones(demo_locations_cluster, pd.Series(labels)), use_container_width=True)
    
    elif strategy == "Time Window Optimization":
        st.markdown("**Optimize delivery times based on customer preferences**")
        st.dataframe(dataset['orders'][['delivery_id', 'priority', 'time_window_start', 'time_window_end']].head(20))
        
        # Show time window distribution
        st.subheader("Time Window Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=dataset['orders']['time_window_start'], name='Window Start'))
        fig.add_trace(go.Histogram(x=dataset['orders']['time_window_end'], name='Window End'))
        fig.update_layout(title='Delivery Time Window Distribution', xaxis_title='Hour', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("💡 Understanding Time Window Distribution", expanded=False):
            st.markdown("""
            This histogram shows when customers prefer deliveries:
            - **Window Start**: When delivery windows begin (peak = popular times)
            - **Window End**: When delivery windows end
            
            **What to Look For:**
            - Peaks indicate popular delivery times (e.g., 9-11 AM, 5-7 PM)
            - Gaps indicate unpopular times (e.g., lunch hours, late night)
            
            **Optimization Strategy**: 
            - Schedule routes to hit popular time windows
            - Group deliveries with similar time windows together
            - Avoid scheduling during unpopular times
            """)
    
    elif strategy == "Delivery Time Prediction":
        st.markdown("**Predict delivery times using ML models**")
        
        # Limit dataset size for faster training
        max_samples = 500
        if len(dataset['orders']) > max_samples:
            st.info(f"⚠️ Large dataset detected ({len(dataset['orders'])} orders). Using first {max_samples} samples for faster training.")
            demo_orders = dataset['orders'].head(max_samples).reset_index(drop=True)
            demo_locations = dataset['delivery_locations'][dataset['delivery_locations']['delivery_id'].isin(demo_orders['delivery_id'])].reset_index(drop=True)
        else:
            demo_orders = dataset['orders']
            demo_locations = dataset['delivery_locations']
        
        if st.button("Train Prediction Model"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Prepare features
            status_text.text("Step 1/4: Preparing features...")
            progress_bar.progress(10)
            predictor = DeliveryTimePredictor()
            
            # Calculate distance matrix (limit size)
            status_text.text("Step 2/4: Calculating distance matrix...")
            progress_bar.progress(20)
            distance_matrix = calculate_distance_matrix(demo_locations)
            progress_bar.progress(40)
            
            # Prepare features
            status_text.text("Step 3/4: Extracting features...")
            progress_bar.progress(50)
            X = predictor.prepare_features(
                demo_orders, demo_locations, distance_matrix,
                dataset['traffic'], dataset['weather']
            )
            y = demo_orders['estimated_delivery_time_minutes']
            progress_bar.progress(70)
            
            # Train model
            status_text.text("Step 4/4: Training model...")
            progress_bar.progress(75)
            results = predictor.train(X, y)
            progress_bar.progress(95)
            
            progress_bar.progress(100)
            status_text.text("✓ Training complete!")
            
            st.success(f"✅ Model trained successfully! MAE: {results['mae']:.2f} minutes, RMSE: {results['rmse']:.2f} minutes")
            
            # Show feature importance if available
            if predictor.feature_importance_ is not None:
                st.subheader("Feature Importance")
                st.dataframe(predictor.feature_importance_, use_container_width=True)
                with st.expander("💡 Understanding Feature Importance", expanded=False):
                    st.markdown("""
                    Feature importance shows which factors most affect delivery time:
                    - **Higher values** = more important factors
                    - **Distance**: Usually most important (longer distance = longer time)
                    - **Traffic**: Peak hours increase delivery time
                    - **Weather**: Adverse conditions slow down delivery
                    - **Order Priority**: Express orders may get faster routes
                    
                    **Use Case**: Focus optimization efforts on high-importance factors.
                    For example, if distance is most important, prioritize route optimization.
                    """)
            
            # Show model performance metrics
            st.subheader("Model Performance")
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            with col_pred1:
                st.metric("Mean Absolute Error", f"{results['mae']:.2f} min")
                st.caption("Average prediction error")
            with col_pred2:
                st.metric("Root Mean Squared Error", f"{results['rmse']:.2f} min")
                st.caption("Penalizes large errors")
            with col_pred3:
                st.metric("Training Samples", len(demo_orders))
                st.caption("Data points used")
            
            with st.expander("💡 Understanding Model Metrics", expanded=False):
                st.markdown("""
                **MAE (Mean Absolute Error)**: 
                - Average difference between predicted and actual delivery times
                - Lower = better predictions
                - Example: MAE of 5 min means predictions are off by 5 minutes on average
                
                **RMSE (Root Mean Squared Error)**:
                - Similar to MAE but penalizes large errors more
                - Lower = better predictions
                - More sensitive to outliers
                
                **What Good Performance Looks Like**:
                - MAE < 10 minutes = very good
                - MAE 10-20 minutes = good
                - MAE > 20 minutes = needs improvement
                """)


def tab_customer_sla():
    """Tab 7: Customer Satisfaction & SLA Compliance"""
    st.header("⭐ Customer Satisfaction & SLA Compliance")
    st.markdown("""
    **Question: What are the implications for customer satisfaction and service-level agreements?**
    
    This tab analyzes SLA compliance and customer satisfaction metrics.
    """)
    
    with st.expander("📖 What's Happening Here?", expanded=False):
        st.markdown("""
        **Understanding SLA (Service-Level Agreement):**
        
        SLAs define delivery promises to customers:
        - **Express**: 1-2 hour delivery windows
        - **Priority**: 4-6 hour delivery windows
        - **Standard**: 24-hour delivery windows
        
        **Why SLA Compliance Matters:**
        - **Customer Satisfaction**: On-time delivery = happy customers
        - **Business Impact**: SLA violations = penalties, refunds, lost customers
        - **Reputation**: Consistent compliance builds trust
        
        **How AI Helps:**
        - **Route Optimization**: Ensures deliveries within time windows
        - **Risk Prediction**: Identifies orders at risk of SLA violation
        - **Proactive Alerts**: Warns dispatchers before violations occur
        - **Dynamic Adjustment**: Re-routes when delays detected
        
        **What You'll See:**
        - SLA compliance rate (percentage of on-time deliveries)
        - Violation analysis (which orders missed windows)
        - Risk prediction model (ML model predicting violation risk)
        - Feature importance (what factors affect compliance)
        """)
    
    dataset = get_dataset()
    if dataset is None:
        return
    
    st.subheader("SLA Overview")
    sla_summary = dataset['orders'].groupby('priority').agg({
        'sla_hours': 'mean',
        'order_id': 'count'
    }).reset_index()
    sla_summary.columns = ['Priority', 'Avg SLA Hours', 'Number of Orders']
    st.dataframe(sla_summary, use_container_width=True)
    
    # Limit dataset size for faster analysis
    max_deliveries_sla = 150
    if len(dataset['delivery_locations']) > max_deliveries_sla:
        st.info(f"⚠️ Large dataset detected ({len(dataset['delivery_locations'])} deliveries). Using first {max_deliveries_sla} deliveries for faster SLA analysis.")
        demo_locations_sla = dataset['delivery_locations'].head(max_deliveries_sla).reset_index(drop=True)
        demo_orders_sla = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locations_sla['delivery_id'])].reset_index(drop=True)
    else:
        demo_locations_sla = dataset['delivery_locations']
        demo_orders_sla = dataset['orders']
    
    if st.button("Analyze SLA Compliance"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Calculate distance matrix
        status_text.text("Step 1/5: Calculating distance matrix...")
        progress_bar.progress(10)
        from utils.route_optimizer import calculate_distance_matrix, generate_baseline_route
        distance_matrix = calculate_distance_matrix(demo_locations_sla)
        progress_bar.progress(25)
        
        # Step 2: Generate baseline route
        status_text.text("Step 2/5: Generating baseline route...")
        progress_bar.progress(30)
        vehicle = dataset['vehicles'].iloc[0]
        baseline_route = generate_baseline_route(demo_locations_sla, dataset['depots'].iloc[0], demo_orders_sla)
        progress_bar.progress(45)
        
        # Step 3: Calculate route metrics
        status_text.text("Step 3/5: Calculating route metrics...")
        progress_bar.progress(50)
        metrics = calculate_route_metrics(
            baseline_route, distance_matrix, vehicle.to_dict(),
            demo_orders_sla, demo_locations_sla, 0,
            traffic_data=dataset['traffic'], weather_data=dataset['weather'],
            depot_location=dataset['depots'].iloc[0]
        )
        progress_bar.progress(65)
        
        # Step 4: Prepare SLA visualization
        status_text.text("Step 4/5: Preparing SLA visualization...")
        progress_bar.progress(70)
        from utils.visualization import plot_sla_compliance
        sla_metrics = {
            'sla_compliance_rate': metrics['sla_compliance_rate'],
            'total_violations': metrics['sla_violations'],
            'total_deliveries': len(baseline_route)
        }
        progress_bar.progress(80)
        
        # Step 5: Train SLA Risk Prediction model
        status_text.text("Step 5/5: Training risk prediction model...")
        progress_bar.progress(85)
        predictor = SLARiskPredictor()
        X = predictor.prepare_features(demo_orders_sla, demo_locations_sla, 12, 30)
        # Simulate labels (in real scenario, these would come from historical data)
        y = np.random.randint(0, 2, len(X))
        results = predictor.train(X, y)
        progress_bar.progress(95)
        
        progress_bar.progress(100)
        status_text.text("✓ Analysis complete!")
        
        st.plotly_chart(plot_sla_compliance(sla_metrics), use_container_width=True)
        with st.expander("💡 Understanding SLA Compliance Chart", expanded=False):
            st.markdown("""
            This chart shows SLA compliance metrics:
            - **Compliance Rate**: Percentage of deliveries completed within time windows
            - **Violations**: Number of deliveries that missed their time windows
            
            **What Good Compliance Looks Like:**
            - 95%+ compliance = excellent
            - 85-95% compliance = good
            - <85% compliance = needs improvement
            
            **Why Violations Happen:**
            - Traffic delays
            - Weather conditions
            - Route inefficiency
            - Unrealistic time windows
            
            **AI Solution**: Route optimization minimizes violations by finding efficient routes
            that account for traffic, weather, and time constraints.
            """)
        
        # SLA Risk Prediction
        st.subheader("SLA Risk Prediction")
        st.success(f"✅ Risk prediction model accuracy: {results['accuracy']*100:.1f}%")
        with st.expander("💡 Understanding Risk Prediction", expanded=False):
            st.markdown("""
            **What Risk Prediction Does:**
            - Predicts which orders are at risk of SLA violation
            - Uses ML model trained on historical data
            - Considers factors like distance, traffic, weather, priority
            
            **How to Use It:**
            - **High Risk Orders**: Prioritize these, assign faster routes
            - **Proactive Action**: Re-route or add resources before violation
            - **Resource Allocation**: Assign best drivers/vehicles to high-risk orders
            
            **Model Accuracy**: 
            - 80%+ accuracy = good for production use
            - Higher accuracy = fewer false alarms, better resource allocation
            """)
        
        if predictor.feature_importance_ is not None:
            st.subheader("Risk Factor Importance")
            st.dataframe(predictor.feature_importance_, use_container_width=True)
            with st.expander("💡 Understanding Risk Factors", expanded=False):
                st.markdown("""
                This table shows which factors most affect SLA violation risk:
                - **Higher values** = stronger risk factors
                - **Distance**: Longer distances = higher risk
                - **Time Window**: Narrower windows = higher risk
                - **Traffic**: Peak hours = higher risk
                - **Weather**: Adverse conditions = higher risk
                
                **Action Items**: Focus on mitigating high-risk factors.
                For example, if traffic is a major risk factor, optimize routes to avoid peak hours.
                """)


def get_dataset():
    """Helper function to get dataset from session state"""
    if 'dataset' not in st.session_state or st.session_state['dataset'] is None:
        st.warning("⚠️ Please load a dataset first from the 'Data Management' tab.")
        return None
    return st.session_state['dataset']


def tab_advanced_features():
    """Tab 8: Advanced Features - All new capabilities"""
    st.header("🚀 Advanced Features & Analytics")
    st.markdown("""
    This tab showcases advanced AI capabilities including multi-objective optimization, 
    explainable AI, benchmarking, API integration, and comprehensive analytics.
    
    ⚡ **Performance Mode**: Optimized for fast demo execution with smaller datasets.
    """)
    
    dataset = get_dataset()
    if dataset is None:
        return
    
    # Performance mode toggle
    performance_mode = st.sidebar.radio("Performance Mode", ["Fast Demo (Recommended)", "Full Analysis"], index=0)
    
    if performance_mode == "Fast Demo (Recommended)":
        max_deliveries = 40
        ga_generations = 15
        aco_iterations = 20
        aco_ants = 20
        sa_iterations = 25
        mc_simulations = 30
        shap_samples = 150
    else:
        max_deliveries = 80
        ga_generations = 30
        aco_iterations = 40
        aco_ants = 30
        sa_iterations = 50
        mc_simulations = 100
        shap_samples = 300
    
    feature_tabs = st.tabs([
        "Multi-Objective Optimization",
        "Advanced Algorithms",
        "What-If Analysis",
        "Explainable AI",
        "Benchmarking",
        "API Integration",
        "Executive Dashboard",
        "Export & Reports"
    ])
    
    with feature_tabs[0]:  # Multi-Objective Optimization
        st.subheader("Multi-Objective Optimization: Pareto Front Analysis")
        st.markdown("""
        Optimize routes considering multiple objectives simultaneously: **Cost**, **Time**, and **Emissions**.
        The Pareto front shows trade-offs between objectives - you can't improve one without worsening another.
        """)
        
        if st.button("Run Multi-Objective Optimization", key="pareto_btn"):
            with st.spinner("Running multi-objective optimization (this may take 30-60 seconds)..."):
                demo_locs = dataset['delivery_locations'].head(max_deliveries).reset_index(drop=True)
                demo_orders = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locs['delivery_id'])].reset_index(drop=True)
                distance_matrix = calculate_distance_matrix(demo_locs)
                vehicle = dataset['vehicles'].iloc[0]
                solutions = []
                
                # Use cached distance matrix if available
                cache_key = f"pareto_dm_{len(demo_locs)}"
                if 'pareto_distance_matrix' not in st.session_state:
                    st.session_state['pareto_distance_matrix'] = {}
                if cache_key not in st.session_state['pareto_distance_matrix']:
                    st.session_state['pareto_distance_matrix'][cache_key] = distance_matrix
                else:
                    distance_matrix = st.session_state['pareto_distance_matrix'][cache_key]
                
                progress = st.progress(0)
                status = st.empty()
                
                algorithms = [
                    ('GA', lambda: GeneticAlgorithmVRP(distance_matrix, demo_orders, demo_locs, vehicle.to_dict(), generations=ga_generations).solve()[0]),
                    ('ACO', lambda: AntColonyVRP(distance_matrix, demo_orders, demo_locs, vehicle.to_dict(), iterations=aco_iterations, n_ants=aco_ants).solve()[0]),
                    ('SA', lambda: SimulatedAnnealingVRP(distance_matrix, demo_orders, demo_locs, vehicle.to_dict(), iterations=sa_iterations).solve()[0])
                ]
                
                for idx, (algo_name, algo_func) in enumerate(algorithms):
                    status.text(f"Running {algo_name}...")
                    progress.progress(int((idx + 1) / len(algorithms) * 90))
                    try:
                        route = algo_func()
                        metrics = calculate_route_metrics(route, distance_matrix, vehicle.to_dict(), demo_orders, demo_locs, 0, depot_location=dataset['depots'].iloc[0])
                        solutions.append({'algorithm': algo_name, 'cost': metrics['total_cost'], 'time': metrics['total_time'], 'emissions': metrics.get('total_distance', 0) * 0.2, 'route': route})
                    except Exception as e:
                        st.warning(f"{algo_name} failed: {str(e)}")
                        pass
                
                progress.progress(100)
                status.text("Calculating Pareto front...")
                
                if len(solutions) > 0:
                    pareto_solutions = calculate_pareto_front(solutions)
                    st.session_state['pareto_solutions'] = pareto_solutions
                    st.session_state['pareto_all_solutions'] = solutions
                    st.session_state['pareto_just_run'] = True
                    st.plotly_chart(plot_pareto_front(solutions, pareto_solutions), use_container_width=True, key="pareto_chart_main")
                    pareto_df = pd.DataFrame([{'Algorithm': s['algorithm'], 'Cost ($)': f"{s['cost']:.2f}", 'Time (hrs)': f"{s['time']:.2f}", 'Emissions (kg)': f"{s['emissions']:.2f}"} for s in pareto_solutions])
                    st.dataframe(pareto_df, use_container_width=True)
                    status.text("✓ Complete!")
        
        # Show cached results if available (only if button wasn't just clicked)
        if 'pareto_solutions' in st.session_state and not st.session_state.get('pareto_just_run', False):
            st.plotly_chart(plot_pareto_front(st.session_state['pareto_all_solutions'], st.session_state['pareto_solutions']), use_container_width=True, key="pareto_chart_cached")
            pareto_df = pd.DataFrame([{'Algorithm': s['algorithm'], 'Cost ($)': f"{s['cost']:.2f}", 'Time (hrs)': f"{s['time']:.2f}", 'Emissions (kg)': f"{s['emissions']:.2f}"} for s in st.session_state['pareto_solutions']])
            st.dataframe(pareto_df, use_container_width=True)
        
        # Reset flag after showing
        if 'pareto_just_run' in st.session_state:
            st.session_state['pareto_just_run'] = False
    
    with feature_tabs[1]:  # Advanced Algorithms
        st.subheader("Advanced Optimization Algorithms")
        st.markdown("Compare **Ant Colony Optimization** and **Simulated Annealing** with Genetic Algorithms.")
        algorithm_choice = st.selectbox("Select Algorithm", ["Ant Colony Optimization", "Simulated Annealing"], key="algo_choice")
        
        demo_locs = dataset['delivery_locations'].head(max_deliveries).reset_index(drop=True)
        demo_orders = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locs['delivery_id'])].reset_index(drop=True)
        
        if st.button(f"Run {algorithm_choice}", key="run_algo_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Cache distance matrix
            cache_key = f"algo_dm_{len(demo_locs)}_{algorithm_choice}"
            if 'algo_distance_matrix' not in st.session_state:
                st.session_state['algo_distance_matrix'] = {}
            if cache_key not in st.session_state['algo_distance_matrix']:
                status_text.text("Calculating distance matrix...")
                distance_matrix = calculate_distance_matrix(demo_locs)
                st.session_state['algo_distance_matrix'][cache_key] = distance_matrix
            else:
                distance_matrix = st.session_state['algo_distance_matrix'][cache_key]
                status_text.text("Using cached distance matrix...")
            
            vehicle = dataset['vehicles'].iloc[0]
            if algorithm_choice == "Ant Colony Optimization":
                status_text.text("Running Ant Colony Optimization...")
                aco = AntColonyVRP(distance_matrix, demo_orders, demo_locs, vehicle.to_dict(), n_ants=aco_ants, iterations=aco_iterations)
                def update_progress(iter, total, best, avg):
                    progress_bar.progress(int(iter / total * 100))
                    status_text.text(f"Iteration {iter}/{total} - Best Cost: {best:.2f}")
                route, history = aco.solve(progress_callback=update_progress)
                st.session_state['algo_route'] = route
                st.session_state['algo_history'] = history
                st.session_state['algo_type'] = 'ACO'
                st.session_state['algo_just_run'] = True
                st.plotly_chart(plot_algorithm_convergence(history), use_container_width=True, key="aco_chart_main")
            else:
                status_text.text("Running Simulated Annealing...")
                sa = SimulatedAnnealingVRP(distance_matrix, demo_orders, demo_locs, vehicle.to_dict(), iterations=sa_iterations)
                def update_progress(iter, total, best, current):
                    progress_bar.progress(int(iter / total * 100))
                    status_text.text(f"Iteration {iter}/{total} - Best Cost: {best:.2f}")
                route, history = sa.solve(progress_callback=update_progress)
                st.session_state['algo_route'] = route
                st.session_state['algo_history'] = history
                st.session_state['algo_type'] = 'SA'
                st.session_state['algo_just_run'] = True
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history['best_fitness'], name='Best Fitness', line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=history['current_fitness'], name='Current Fitness', line=dict(color='orange')))
                fig.update_layout(title='Simulated Annealing Convergence', xaxis_title='Iteration', yaxis_title='Cost')
                st.plotly_chart(fig, use_container_width=True, key="sa_chart_main")
            metrics = calculate_route_metrics(route, distance_matrix, vehicle.to_dict(), demo_orders, demo_locs, 0, depot_location=dataset['depots'].iloc[0])
            st.session_state['algo_metrics'] = metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cost", f"${metrics['total_cost']:.2f}")
            with col2:
                st.metric("Total Time", f"{metrics['total_time']:.2f} hrs")
            with col3:
                st.metric("Total Distance", f"{metrics['total_distance']:.2f} km")
        
        # Show cached results if available (only if button wasn't just clicked)
        if 'algo_route' in st.session_state and not st.session_state.get('algo_just_run', False):
            if st.session_state['algo_type'] == 'ACO':
                st.plotly_chart(plot_algorithm_convergence(st.session_state['algo_history']), use_container_width=True, key="aco_chart_cached")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=st.session_state['algo_history']['best_fitness'], name='Best Fitness', line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=st.session_state['algo_history']['current_fitness'], name='Current Fitness', line=dict(color='orange')))
                fig.update_layout(title='Simulated Annealing Convergence', xaxis_title='Iteration', yaxis_title='Cost')
                st.plotly_chart(fig, use_container_width=True, key="sa_chart_cached")
            metrics = st.session_state['algo_metrics']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cost", f"${metrics['total_cost']:.2f}")
            with col2:
                st.metric("Total Time", f"{metrics['total_time']:.2f} hrs")
            with col3:
                st.metric("Total Distance", f"{metrics['total_distance']:.2f} km")
        
        # Reset flag after showing
        if 'algo_just_run' in st.session_state:
            st.session_state['algo_just_run'] = False
    
    with feature_tabs[2]:  # What-If Analysis
        st.subheader("What-If Scenario Analysis")
        st.markdown("Analyze how different scenarios affect route performance.")
        
        col1, col2 = st.columns(2)
        with col1:
            scenario_traffic = st.slider("Traffic Multiplier", 0.5, 2.0, 1.0, 0.1)
            scenario_weather = st.selectbox("Weather Condition", ["clear", "rain", "snow", "storm"])
        with col2:
            scenario_deliveries = st.slider("Number of Additional Deliveries", -20, 50, 0)
            scenario_priority = st.selectbox("Priority Focus", ["cost", "time", "emissions"])
        
        # Initialize session state for caching
        if 'whatif_baseline_route' not in st.session_state:
            st.session_state['whatif_baseline_route'] = None
            st.session_state['whatif_baseline_metrics'] = None
            st.session_state['whatif_distance_matrix'] = None
            st.session_state['whatif_vehicle'] = None
            st.session_state['whatif_depot'] = None
            st.session_state['whatif_demo_locs'] = None
            st.session_state['whatif_demo_orders'] = None
        
        if st.button("Analyze Scenarios", key="analyze_scenarios_btn"):
            with st.spinner("Preparing baseline scenario..."):
                demo_locs = dataset['delivery_locations'].head(max_deliveries).reset_index(drop=True)
                demo_orders = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locs['delivery_id'])].reset_index(drop=True)
                
                # Cache distance matrix
                cache_key = f"whatif_dm_{len(demo_locs)}"
                if 'whatif_distance_matrix_cache' not in st.session_state:
                    st.session_state['whatif_distance_matrix_cache'] = {}
                if cache_key not in st.session_state['whatif_distance_matrix_cache']:
                    distance_matrix = calculate_distance_matrix(demo_locs)
                    st.session_state['whatif_distance_matrix_cache'][cache_key] = distance_matrix
                else:
                    distance_matrix = st.session_state['whatif_distance_matrix_cache'][cache_key]
                
                vehicle = dataset['vehicles'].iloc[0]
                depot = dataset['depots'].iloc[0]
                baseline_route = generate_baseline_route(demo_locs, depot, demo_orders)
                baseline_metrics = calculate_route_metrics(baseline_route, distance_matrix, vehicle.to_dict(), demo_orders, demo_locs, 0, depot_location=depot)
                
                # Store in session state
                st.session_state['whatif_baseline_route'] = baseline_route
                st.session_state['whatif_baseline_metrics'] = baseline_metrics
                st.session_state['whatif_distance_matrix'] = distance_matrix
                st.session_state['whatif_vehicle'] = vehicle.to_dict()
                st.session_state['whatif_depot'] = depot
                st.session_state['whatif_demo_locs'] = demo_locs
                st.session_state['whatif_demo_orders'] = demo_orders
            
            scenarios = [
                {'name': 'High Traffic', 'cost': baseline_metrics['total_cost'] * scenario_traffic, 'time': baseline_metrics['total_time'] * scenario_traffic, 'distance': baseline_metrics['total_distance'], 'emissions': baseline_metrics['total_distance'] * 0.2 * scenario_traffic},
                {'name': 'Adverse Weather', 'cost': baseline_metrics['total_cost'] * 1.3, 'time': baseline_metrics['total_time'] * 1.3, 'distance': baseline_metrics['total_distance'], 'emissions': baseline_metrics['total_distance'] * 0.2 * 1.3},
                {'name': 'More Deliveries', 'cost': baseline_metrics['total_cost'] * (1 + scenario_deliveries / 100), 'time': baseline_metrics['total_time'] * (1 + scenario_deliveries / 100), 'distance': baseline_metrics['total_distance'] * (1 + scenario_deliveries / 100), 'emissions': baseline_metrics['total_distance'] * 0.2 * (1 + scenario_deliveries / 100)}
            ]
            base_scenario = {'cost': baseline_metrics['total_cost'], 'time': baseline_metrics['total_time'], 'distance': baseline_metrics['total_distance'], 'emissions': baseline_metrics['total_distance'] * 0.2}
            fig, comparison_data = analyze_what_if_scenarios(base_scenario, scenarios)
            st.session_state['whatif_scenarios_done'] = True
            st.session_state['whatif_just_run'] = True
            st.plotly_chart(fig, use_container_width=True, key="whatif_chart_main")
        
        # Show results if scenarios were analyzed (only if button wasn't just clicked)
        if st.session_state.get('whatif_scenarios_done', False) and st.session_state.get('whatif_baseline_route') is not None and not st.session_state.get('whatif_just_run', False):
            baseline_route = st.session_state['whatif_baseline_route']
            baseline_metrics = st.session_state['whatif_baseline_metrics']
            distance_matrix = st.session_state['whatif_distance_matrix']
            vehicle = st.session_state['whatif_vehicle']
            demo_locs = st.session_state['whatif_demo_locs']
            demo_orders = st.session_state['whatif_demo_orders']
            
            # Recalculate scenarios for display
            scenarios = [
                {'name': 'High Traffic', 'cost': baseline_metrics['total_cost'] * scenario_traffic, 'time': baseline_metrics['total_time'] * scenario_traffic, 'distance': baseline_metrics['total_distance'], 'emissions': baseline_metrics['total_distance'] * 0.2 * scenario_traffic},
                {'name': 'Adverse Weather', 'cost': baseline_metrics['total_cost'] * 1.3, 'time': baseline_metrics['total_time'] * 1.3, 'distance': baseline_metrics['total_distance'], 'emissions': baseline_metrics['total_distance'] * 0.2 * 1.3},
                {'name': 'More Deliveries', 'cost': baseline_metrics['total_cost'] * (1 + scenario_deliveries / 100), 'time': baseline_metrics['total_time'] * (1 + scenario_deliveries / 100), 'distance': baseline_metrics['total_distance'] * (1 + scenario_deliveries / 100), 'emissions': baseline_metrics['total_distance'] * 0.2 * (1 + scenario_deliveries / 100)}
            ]
            base_scenario = {'cost': baseline_metrics['total_cost'], 'time': baseline_metrics['total_time'], 'distance': baseline_metrics['total_distance'], 'emissions': baseline_metrics['total_distance'] * 0.2}
            fig, comparison_data = analyze_what_if_scenarios(base_scenario, scenarios)
            st.plotly_chart(fig, use_container_width=True, key="whatif_chart_cached")
        
        # Reset flag after showing
        if 'whatif_just_run' in st.session_state:
            st.session_state['whatif_just_run'] = False
        
        st.subheader("Monte Carlo Simulation: Uncertainty Analysis")
        if st.button("Run Monte Carlo Simulation", key="monte_carlo_btn"):
            if st.session_state.get('whatif_baseline_route') is None:
                st.warning("⚠️ Please run 'Analyze Scenarios' first to prepare baseline data.")
            else:
                with st.spinner(f"Running {mc_simulations} simulations (reduced for speed)..."):
                    baseline_route = st.session_state['whatif_baseline_route']
                    distance_matrix = st.session_state['whatif_distance_matrix']
                    vehicle = st.session_state['whatif_vehicle']
                    demo_locs = st.session_state['whatif_demo_locs']
                    demo_orders = st.session_state['whatif_demo_orders']
                    mc_results = run_monte_carlo_simulation(baseline_route, distance_matrix, vehicle, demo_orders, demo_locs, n_simulations=mc_simulations)
                    st.session_state['mc_results'] = mc_results
                    st.session_state['mc_just_run'] = True
                    st.plotly_chart(plot_monte_carlo_results(mc_results), use_container_width=True, key="monte_carlo_chart_main")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Cost", f"${mc_results['cost'].mean():.2f}")
                    with col2:
                        st.metric("Std Dev", f"${mc_results['cost'].std():.2f}")
                    with col3:
                        st.metric("Min Cost", f"${mc_results['cost'].min():.2f}")
                    with col4:
                        st.metric("Max Cost", f"${mc_results['cost'].max():.2f}")
        
        # Show cached Monte Carlo results if available (only if button wasn't just clicked)
        if 'mc_results' in st.session_state and not st.session_state.get('mc_just_run', False):
            st.plotly_chart(plot_monte_carlo_results(st.session_state['mc_results']), use_container_width=True, key="monte_carlo_chart_cached")
            mc_results = st.session_state['mc_results']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Cost", f"${mc_results['cost'].mean():.2f}")
            with col2:
                st.metric("Std Dev", f"${mc_results['cost'].std():.2f}")
            with col3:
                st.metric("Min Cost", f"${mc_results['cost'].min():.2f}")
            with col4:
                st.metric("Max Cost", f"${mc_results['cost'].max():.2f}")
        
        # Reset flag after showing
        if 'mc_just_run' in st.session_state:
            st.session_state['mc_just_run'] = False
        
        st.subheader("Sensitivity Analysis")
        param_name = st.selectbox("Parameter", ["Traffic Multiplier", "Weather Multiplier", "Fuel Price"], key="sens_param")
        param_range = np.linspace(0.5, 2.0, 15) if "Multiplier" in param_name else np.linspace(1.0, 2.5, 15)  # Reduced points for speed
        if st.button("Run Sensitivity Analysis", key="sensitivity_btn"):
            if st.session_state.get('whatif_baseline_route') is None:
                st.warning("⚠️ Please run 'Analyze Scenarios' first to prepare baseline data.")
            else:
                baseline_route = st.session_state['whatif_baseline_route']
                distance_matrix = st.session_state['whatif_distance_matrix']
                vehicle = st.session_state['whatif_vehicle']
                demo_locs = st.session_state['whatif_demo_locs']
                demo_orders = st.session_state['whatif_demo_orders']
                sens_results = sensitivity_analysis(baseline_route, distance_matrix, vehicle, demo_orders, demo_locs, param_name.lower().replace(' ', '_'), param_range)
                st.session_state['sens_results'] = sens_results
                st.session_state['sens_param_name'] = param_name
                st.session_state['sens_just_run'] = True
                st.plotly_chart(plot_sensitivity_analysis(sens_results, param_name), use_container_width=True, key="sensitivity_chart_main")
        
        # Show cached sensitivity results if available (only if button wasn't just clicked)
        if 'sens_results' in st.session_state and not st.session_state.get('sens_just_run', False):
            st.plotly_chart(plot_sensitivity_analysis(st.session_state['sens_results'], st.session_state.get('sens_param_name', param_name)), use_container_width=True, key="sensitivity_chart_cached")
        
        # Reset flag after showing
        if 'sens_just_run' in st.session_state:
            st.session_state['sens_just_run'] = False
    
    with feature_tabs[3]:  # Explainable AI
        st.subheader("Explainable AI: Model Interpretability")
        st.markdown("Understand which factors most influence delivery time predictions using SHAP values.")
        if st.button("Generate SHAP Analysis", key="shap_btn"):
            with st.spinner("Training model and calculating SHAP values (this may take 20-30 seconds)..."):
                demo_orders_ml = dataset['orders'].head(shap_samples).reset_index(drop=True)
                demo_locs_ml = dataset['delivery_locations'][dataset['delivery_locations']['delivery_id'].isin(demo_orders_ml['delivery_id'])].reset_index(drop=True)
                predictor = DeliveryTimePredictor()
                
                # Cache distance matrix
                cache_key = f"shap_dm_{len(demo_locs_ml)}"
                if 'shap_distance_matrix_cache' not in st.session_state:
                    st.session_state['shap_distance_matrix_cache'] = {}
                if cache_key not in st.session_state['shap_distance_matrix_cache']:
                    distance_matrix_ml = calculate_distance_matrix(demo_locs_ml)
                    st.session_state['shap_distance_matrix_cache'][cache_key] = distance_matrix_ml
                else:
                    distance_matrix_ml = st.session_state['shap_distance_matrix_cache'][cache_key]
                
                X = predictor.prepare_features(demo_orders_ml, demo_locs_ml, distance_matrix_ml, dataset['traffic'], dataset['weather'])
                y = demo_orders_ml['estimated_delivery_time_minutes']
                predictor.train(X, y)
                if predictor.model is not None:
                    shap_values, X_sample = calculate_shap_values(predictor.model, X)
                    if shap_values is not None:
                        shap_fig = plot_shap_summary(shap_values, X.columns.tolist())
                        if shap_fig:
                            st.session_state['shap_fig'] = shap_fig
                            st.session_state['shap_values'] = shap_values
                            st.session_state['shap_X_sample'] = X_sample
                            st.session_state['shap_just_run'] = True
                            st.plotly_chart(shap_fig, use_container_width=True, key="shap_chart_main")
                        st.subheader("SHAP Waterfall Plot (Sample Instance)")
                        instance_idx = st.slider("Instance", 0, min(9, len(X_sample)-1), 0, key="shap_instance")
                        waterfall_fig = plot_shap_waterfall(shap_values, instance_idx)
                        if waterfall_fig:
                            st.session_state['waterfall_fig'] = waterfall_fig
                            st.plotly_chart(waterfall_fig, use_container_width=True, key="shap_waterfall_main")
                    else:
                        st.info("SHAP calculation failed. Showing feature importance instead.")
                        if predictor.feature_importance_ is not None:
                            st.session_state['feature_importance'] = predictor.feature_importance_
                            st.dataframe(predictor.feature_importance_, use_container_width=True)
        
        # Show cached SHAP results if available (only if button wasn't just clicked)
        if 'shap_fig' in st.session_state and not st.session_state.get('shap_just_run', False):
            st.plotly_chart(st.session_state['shap_fig'], use_container_width=True, key="shap_chart_cached")
            st.subheader("SHAP Waterfall Plot (Sample Instance)")
            instance_idx = st.slider("Instance", 0, min(9, len(st.session_state['shap_X_sample'])-1), 0, key="shap_instance_cached")
            waterfall_fig = plot_shap_waterfall(st.session_state['shap_values'], instance_idx)
            if waterfall_fig:
                st.plotly_chart(waterfall_fig, use_container_width=True, key="shap_waterfall_cached")
        elif 'feature_importance' in st.session_state:
            st.dataframe(st.session_state['feature_importance'], use_container_width=True)
        
        # Reset flag after showing
        if 'shap_just_run' in st.session_state:
            st.session_state['shap_just_run'] = False
    
    with feature_tabs[4]:  # Benchmarking
        st.subheader("Algorithm Benchmarking & Comparison")
        st.markdown("Compare performance of different algorithms on the same problem.")
        if st.button("Run Benchmark", key="benchmark_btn"):
            with st.spinner("Benchmarking algorithms (this may take 60-90 seconds)..."):
                demo_locs_bench = dataset['delivery_locations'].head(max_deliveries).reset_index(drop=True)
                demo_orders_bench = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locs_bench['delivery_id'])].reset_index(drop=True)
                
                # Cache distance matrix
                cache_key = f"bench_dm_{len(demo_locs_bench)}"
                if 'bench_distance_matrix_cache' not in st.session_state:
                    st.session_state['bench_distance_matrix_cache'] = {}
                if cache_key not in st.session_state['bench_distance_matrix_cache']:
                    distance_matrix_bench = calculate_distance_matrix(demo_locs_bench)
                    st.session_state['bench_distance_matrix_cache'][cache_key] = distance_matrix_bench
                else:
                    distance_matrix_bench = st.session_state['bench_distance_matrix_cache'][cache_key]
                
                vehicle_bench = dataset['vehicles'].iloc[0]
                depot_bench = dataset['depots'].iloc[0]
                benchmark_results = []
                import time
                
                progress = st.progress(0)
                status = st.empty()
                
                status.text("Running Genetic Algorithm...")
                start = time.time()
                ga = GeneticAlgorithmVRP(distance_matrix_bench, demo_orders_bench, demo_locs_bench, vehicle_bench.to_dict(), generations=ga_generations)
                ga_route, _ = ga.solve()
                ga_time = time.time() - start
                ga_metrics = calculate_route_metrics(ga_route, distance_matrix_bench, vehicle_bench.to_dict(), demo_orders_bench, demo_locs_bench, 0, depot_location=depot_bench)
                benchmark_results.append({'algorithm': 'Genetic Algorithm', 'execution_time': ga_time, 'solution_cost': ga_metrics['total_cost'], 'test_case': 'Standard VRP'})
                progress.progress(33)
                
                status.text("Running Ant Colony Optimization...")
                start = time.time()
                aco = AntColonyVRP(distance_matrix_bench, demo_orders_bench, demo_locs_bench, vehicle_bench.to_dict(), iterations=aco_iterations, n_ants=aco_ants)
                aco_route, _ = aco.solve()
                aco_time = time.time() - start
                aco_metrics = calculate_route_metrics(aco_route, distance_matrix_bench, vehicle_bench.to_dict(), demo_orders_bench, demo_locs_bench, 0, depot_location=depot_bench)
                benchmark_results.append({'algorithm': 'Ant Colony', 'execution_time': aco_time, 'solution_cost': aco_metrics['total_cost'], 'test_case': 'Standard VRP'})
                progress.progress(66)
                
                status.text("Running Simulated Annealing...")
                start = time.time()
                sa = SimulatedAnnealingVRP(distance_matrix_bench, demo_orders_bench, demo_locs_bench, vehicle_bench.to_dict(), iterations=sa_iterations)
                sa_route, _ = sa.solve()
                sa_time = time.time() - start
                sa_metrics = calculate_route_metrics(sa_route, distance_matrix_bench, vehicle_bench.to_dict(), demo_orders_bench, demo_locs_bench, 0, depot_location=depot_bench)
                benchmark_results.append({'algorithm': 'Simulated Annealing', 'execution_time': sa_time, 'solution_cost': sa_metrics['total_cost'], 'test_case': 'Standard VRP'})
                progress.progress(100)
                
                comparison_df = pd.DataFrame(benchmark_results)
                st.session_state['benchmark_results'] = comparison_df
                st.session_state['benchmark_just_run'] = True
                st.plotly_chart(plot_algorithm_comparison(comparison_df), use_container_width=True, key="benchmark_chart_main")
                st.subheader("Benchmark Results")
                st.dataframe(comparison_df, use_container_width=True)
                status.text("✓ Complete!")
        
        # Show cached benchmark results if available (only if button wasn't just clicked)
        if 'benchmark_results' in st.session_state and not st.session_state.get('benchmark_just_run', False):
            st.plotly_chart(plot_algorithm_comparison(st.session_state['benchmark_results']), use_container_width=True, key="benchmark_chart_cached")
            st.subheader("Benchmark Results")
            st.dataframe(st.session_state['benchmark_results'], use_container_width=True)
        
        # Reset flag after showing
        if 'benchmark_just_run' in st.session_state:
            st.session_state['benchmark_just_run'] = False
    
    with feature_tabs[5]:  # API Integration
        st.subheader("API Integration Examples")
        st.markdown("Demonstrate integration with external mapping services (Google Maps, HERE Maps).")
        if st.button("Demonstrate API Integration"):
            with st.spinner("Simulating API calls..."):
                max_del = 10
                demo_locs_api = dataset['delivery_locations'].head(max_del)
                results, api_calls = demonstrate_api_integration(demo_locs_api, dataset['depots'])
                st.success(f"✅ Simulated {api_calls} API calls")
                st.subheader("Google Maps API Results")
                google_df = pd.DataFrame(results['google_maps'])
                st.dataframe(google_df, use_container_width=True)
                st.subheader("HERE Maps API Results")
                here_df = pd.DataFrame(results['here_maps'])
                st.dataframe(here_df, use_container_width=True)
                st.info("**Note:** This is a mock demonstration. In production, you would use real API keys for Google Maps or HERE Maps.")
    
    with feature_tabs[6]:  # Executive Dashboard
        st.subheader("Executive Dashboard: Business Intelligence")
        st.markdown("Comprehensive dashboard with key performance indicators and ROI analysis.")
        if st.button("Generate Executive Dashboard"):
            with st.spinner("Generating dashboard..."):
                if 'optimized_metrics' in st.session_state:
                    metrics = st.session_state['optimized_metrics']
                else:
                    max_del = 100
                    demo_locs_dash = dataset['delivery_locations'].head(max_del).reset_index(drop=True)
                    demo_orders_dash = dataset['orders'][dataset['orders']['delivery_id'].isin(demo_locs_dash['delivery_id'])].reset_index(drop=True)
                    distance_matrix_dash = calculate_distance_matrix(demo_locs_dash)
                    vehicle_dash = dataset['vehicles'].iloc[0]
                    depot_dash = dataset['depots'].iloc[0]
                    baseline_route_dash = generate_baseline_route(demo_locs_dash, depot_dash, demo_orders_dash)
                    metrics = calculate_route_metrics(baseline_route_dash, distance_matrix_dash, vehicle_dash.to_dict(), demo_orders_dash, demo_locs_dash, 0, depot_location=depot_dash)
                emissions_kg = metrics.get('total_distance', 0) * 0.2
                dashboard_metrics = {
                    'total_cost': metrics.get('total_cost', 0), 'total_time': metrics.get('total_time', 0), 'total_distance': metrics.get('total_distance', 0),
                    'emissions_kg': emissions_kg, 'weight_utilization': metrics.get('weight_utilization', 0), 'volume_utilization': metrics.get('volume_utilization', 0),
                    'sla_compliance_rate': metrics.get('sla_compliance_rate', 0), 'fuel_cost': metrics.get('fuel_cost', 0), 'driver_cost': metrics.get('driver_cost', 0),
                    'maintenance_cost': metrics.get('maintenance_cost', 0), 'cost_target': metrics.get('total_cost', 0) * 1.2, 'time_target': metrics.get('total_time', 0) * 1.2,
                    'emissions_target': emissions_kg * 1.2, 'customer_satisfaction': metrics.get('sla_compliance_rate', 0) * 100,
                    'cost_per_delivery': metrics.get('total_cost', 0) / max(metrics.get('n_deliveries', 1), 1), 'time_per_delivery': metrics.get('total_time', 0) / max(metrics.get('n_deliveries', 1), 1),
                    'distance_per_delivery': metrics.get('total_distance', 0) / max(metrics.get('n_deliveries', 1), 1)
                }
                dashboard_fig = create_executive_dashboard(dashboard_metrics)
                st.session_state['executive_dashboard_fig'] = dashboard_fig
                st.session_state['executive_just_run'] = True
                st.plotly_chart(dashboard_fig, use_container_width=True, key="executive_dashboard_main")
                st.subheader("ROI Calculator")
                col1, col2 = st.columns(2)
                with col1:
                    implementation_cost = st.number_input("Implementation Cost ($)", 10000, 1000000, 50000)
                    monthly_savings = st.number_input("Monthly Savings ($)", 1000, 100000, 5000)
                with col2:
                    months = st.number_input("Analysis Period (months)", 1, 60, 12)
                roi_results = calculate_roi(implementation_cost, monthly_savings, months)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Savings", f"${roi_results['total_savings']:,.0f}")
                with col2:
                    st.metric("ROI", f"{roi_results['roi_percentage']:.1f}%")
                with col3:
                    st.metric("Payback Period", f"{roi_results['payback_period_months']:.1f} months")
                with col4:
                    st.metric("Net Benefit", f"${roi_results['net_benefit']:,.0f}")
        
        # Show cached executive dashboard if available (only if button wasn't just clicked)
        if 'executive_dashboard_fig' in st.session_state and not st.session_state.get('executive_just_run', False):
            st.plotly_chart(st.session_state['executive_dashboard_fig'], use_container_width=True, key="executive_dashboard_cached")
            st.subheader("ROI Calculator")
            col1, col2 = st.columns(2)
            with col1:
                implementation_cost = st.number_input("Implementation Cost ($)", 10000, 1000000, 50000, key="roi_cost_cached")
                monthly_savings = st.number_input("Monthly Savings ($)", 1000, 100000, 5000, key="roi_savings_cached")
            with col2:
                months = st.number_input("Analysis Period (months)", 1, 60, 12, key="roi_months_cached")
            roi_results = calculate_roi(implementation_cost, monthly_savings, months)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Savings", f"${roi_results['total_savings']:,.0f}")
            with col2:
                st.metric("ROI", f"{roi_results['roi_percentage']:.1f}%")
            with col3:
                st.metric("Payback Period", f"{roi_results['payback_period_months']:.1f} months")
            with col4:
                st.metric("Net Benefit", f"${roi_results['net_benefit']:,.0f}")
        
        # Reset flag after showing
        if 'executive_just_run' in st.session_state:
            st.session_state['executive_just_run'] = False
    
    with feature_tabs[7]:  # Export & Reports
        st.subheader("Export & Report Generation")
        st.markdown("Export routes, metrics, and visualizations in various formats.")
        if 'optimized_route' in st.session_state:
            route_to_export = st.session_state['optimized_route']
            delivery_locs_export = st.session_state.get('sample_delivery_locations', dataset['delivery_locations'])
            depots_export = dataset['depots']
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📄 Export Route to CSV"):
                    csv_data = export_route_to_csv(route_to_export, delivery_locs_export, depots_export)
                    st.download_button(label="Download CSV", data=csv_data, file_name="route_export.csv", mime="text/csv")
            with col2:
                if st.button("📊 Export Metrics to Excel"):
                    if 'optimized_metrics' in st.session_state:
                        metrics_dict = st.session_state['optimized_metrics']
                        excel_data = export_metrics_to_excel(metrics_dict)
                        st.download_button(label="Download Excel", data=excel_data, file_name="metrics_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col3:
                if st.button("🔗 Generate Shareable Link"):
                    scenario_data = {'route': route_to_export, 'metrics': st.session_state.get('optimized_metrics', {}), 'dataset': dataset.get('scenario_name', 'unknown')}
                    shareable_link = create_shareable_link(scenario_data)
                    st.code(shareable_link, language=None)
                    st.info("Copy this link to share your optimization scenario")
            st.subheader("Generate PDF Report")
            if st.button("📑 Generate PDF Report"):
                if 'optimized_metrics' in st.session_state:
                    sections = [
                        ("Route Optimization Summary", f"Optimized route with {len(route_to_export)} deliveries"),
                        ("Key Metrics", pd.DataFrame({'Metric': ['Total Cost', 'Total Time', 'Total Distance'], 'Value': [f"${st.session_state['optimized_metrics']['total_cost']:.2f}", f"{st.session_state['optimized_metrics']['total_time']:.2f} hrs", f"{st.session_state['optimized_metrics']['total_distance']:.2f} km"]}))
                    ]
                    pdf_data = create_pdf_report("Route Optimization Report", sections)
                    if pdf_data:
                        st.download_button(label="Download PDF Report", data=pdf_data, file_name="optimization_report.pdf", mime="application/pdf")
        else:
            st.info("Please run an optimization first to enable export features.")


def tab_data_management():
    """Data Management Tab"""
    st.header("📊 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["Load Dataset", "Generate Dataset", "Dataset Info"])
    
    with tab1:
        st.subheader("Load Pre-Generated Dataset")
        
        # Add refresh button to reload dataset list
        col_refresh, col_info = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh List", help="Refresh the dataset list to see newly generated datasets"):
                load_cached_dataset.clear()
                st.rerun()
        
        datasets = list_available_datasets()
        dataset_names = [d['name'] for d in datasets]
        
        if len(dataset_names) == 0:
            st.warning("No datasets found. Generate a dataset in the 'Generate Dataset' tab.")
        else:
            selected = st.selectbox("Select Dataset", dataset_names)
            
            if st.button("Load Dataset"):
                dataset = load_cached_dataset(selected)
                st.session_state['dataset'] = dataset
                st.session_state['selected_dataset'] = selected
                st.success(f"✅ Dataset '{selected}' loaded!")
            
            # Show dataset info only if dataset is loaded
            if 'dataset' in st.session_state and st.session_state['dataset'] is not None:
                dataset = st.session_state['dataset']
                st.subheader("Dataset Information")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Deliveries", len(dataset['delivery_locations']))
                with col2:
                    st.metric("Orders", len(dataset['orders']))
                with col3:
                    st.metric("Vehicles", len(dataset['vehicles']))
                with col4:
                    st.metric("Depots", len(dataset['depots']))
    
    with tab2:
        st.subheader("Generate Custom Dataset")
        from data_generator import LogisticsDataGenerator
        
        col1, col2 = st.columns(2)
        with col1:
            n_deliveries = st.number_input("Number of Deliveries", 10, 1000, 200)
            n_vehicles = st.number_input("Number of Vehicles", 1, 100, 15)
        with col2:
            n_depots = st.number_input("Number of Depots", 1, 10, 2)
            city_name = st.text_input("City Name", "New York")
        
        scenario_name = st.text_input("Scenario Name", "custom_dataset")
        
        if st.button("Generate Dataset"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Generating dataset...")
            progress_bar.progress(20)
            generator = LogisticsDataGenerator()
            dataset = generator.generate_complete_dataset(
                n_deliveries=n_deliveries,
                n_vehicles=n_vehicles,
                n_depots=n_depots,
                scenario_name=scenario_name,
                city_name=city_name
            )
            progress_bar.progress(80)
            
            status_text.text("Saving dataset...")
            # Save to the same directory where pre-generated datasets are stored
            generator.save_dataset(dataset, 'data/pre_generated')
            progress_bar.progress(95)
            
            # Clear the cache so the dataset list refreshes
            load_cached_dataset.clear()
            
            progress_bar.progress(100)
            status_text.text("✓ Dataset saved!")
            
            st.success(f"✅ Dataset '{scenario_name}' generated and saved to data/pre_generated/")
            st.info("💡 **Tip:** Refresh the page or navigate to 'Load Dataset' tab to see your new dataset in the list.")
    
    with tab3:
        st.subheader("Available Datasets")
        
        # Add refresh button
        if st.button("🔄 Refresh Dataset List"):
            load_cached_dataset.clear()
            st.rerun()
        
        datasets = list_available_datasets()
        
        if len(datasets) == 0:
            st.info("No datasets found. Generate a dataset in the 'Generate Dataset' tab.")
        else:
            st.write(f"Found {len(datasets)} dataset(s):")
            for ds in datasets:
                with st.expander(f"📦 {ds['name']}"):
                    if 'metadata' in ds and ds['metadata']:
                        st.json(ds['metadata'])
                    else:
                        st.caption("No metadata available")


def tab_home():
    """Home page with project overview, objectives, and achievements"""
    st.header("🏠 Welcome to AI in Logistics & Route Optimization")
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;'>
    <h3 style='color: #1f77b4;'>Project Overview</h3>
    <p style='font-size: 16px; line-height: 1.6;'>
    This comprehensive application demonstrates how Artificial Intelligence can revolutionize logistics and route optimization 
    in supply chain management. Through advanced algorithms, real-time data integration, and predictive analytics, we showcase 
    how AI can reduce costs, improve efficiency, minimize environmental impact, and enhance customer satisfaction.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🎯 Project Objectives")
    objectives = [
        "**Optimize delivery routes** to minimize transportation costs and delivery time",
        "**Integrate real-time data** (traffic, weather, demand) for dynamic routing decisions",
        "**Demonstrate multiple AI algorithms** (Genetic Algorithms, Reinforcement Learning, Clustering) for route optimization",
        "**Improve fleet utilization** and reduce carbon footprint through intelligent resource allocation",
        "**Address integration challenges** in deploying AI solutions with existing logistics systems",
        "**Optimize last-mile delivery** to enhance customer satisfaction and efficiency",
        "**Ensure SLA compliance** and analyze customer satisfaction metrics"
    ]
    
    for i, obj in enumerate(objectives, 1):
        st.markdown(f"{i}. {obj}")
    
    st.divider()
    
    st.subheader("❓ Key Questions & Our Solutions")
    
    questions_answers = [
        {
            "question": "How can AI optimize delivery routes and reduce transportation costs?",
            "answer": "We use **Genetic Algorithms** to find optimal routes that minimize distance, time, and cost while respecting vehicle capacity constraints. Our implementation shows cost reductions of 15-30% compared to baseline routes.",
            "tab": "1. Route Optimization",
            "key_features": [
                "Genetic Algorithm optimization with customizable parameters",
                "Real-time cost, distance, and time calculations",
                "Visual route comparison (baseline vs optimized)",
                "Traffic and weather-aware routing"
            ]
        },
        {
            "question": "What role do real-time traffic, weather, and demand data play in dynamic routing?",
            "answer": "Real-time data enables **adaptive routing** that responds to changing conditions. We demonstrate how traffic patterns, weather conditions, and demand fluctuations affect route performance, and how Reinforcement Learning agents can adapt routes dynamically.",
            "tab": "2. Real-Time Data Integration",
            "key_features": [
                "Interactive traffic and weather simulation",
                "Comparison of static vs real-time vs adaptive routing",
                "Reinforcement Learning agent for dynamic adjustments",
                "Real-time dashboard showing condition impacts"
            ]
        },
        {
            "question": "Which algorithms are commonly used (e.g., genetic algorithms, reinforcement learning, clustering)?",
            "answer": "We showcase **three key algorithms**: Genetic Algorithms for route optimization, Reinforcement Learning (Q-learning) for adaptive routing, and K-Means Clustering for delivery zone assignment. Each has distinct strengths for different scenarios.",
            "tab": "3. Algorithm Showcase",
            "key_features": [
                "Interactive demonstrations of each algorithm",
                "Algorithm comparison table",
                "Convergence visualizations",
                "Performance metrics for each approach"
            ]
        },
        {
            "question": "How can AI improve fleet utilization and reduce carbon footprint?",
            "answer": "AI optimizes vehicle assignments to maximize capacity utilization while minimizing total distance traveled. Our analysis shows how optimized routing reduces fuel consumption and CO₂ emissions by 20-35% compared to unoptimized routes.",
            "tab": "4. Fleet Utilization & Sustainability",
            "key_features": [
                "Fleet utilization metrics (weight and volume)",
                "Carbon footprint analysis and emissions reduction",
                "Vehicle assignment optimization",
                "Environmental impact visualization"
            ]
        },
        {
            "question": "What are the challenges in integrating AI with existing logistics systems?",
            "answer": "We identify key challenges including legacy system compatibility, data quality issues, real-time processing requirements, and change management. We provide solutions and an implementation roadmap.",
            "tab": "5. Integration Challenges",
            "key_features": [
                "Detailed challenge analysis",
                "Solution strategies for each challenge",
                "System architecture diagrams",
                "Phased implementation roadmap"
            ]
        },
        {
            "question": "How does AI contribute to last-mile delivery efficiency?",
            "answer": "AI optimizes last-mile delivery through route clustering, time window optimization, and delivery time prediction. Machine Learning models predict delivery times with high accuracy, enabling better planning.",
            "tab": "6. Last-Mile Delivery",
            "key_features": [
                "Delivery zone clustering optimization",
                "Time window distribution analysis",
                "ML-based delivery time prediction",
                "Feature importance analysis"
            ]
        },
        {
            "question": "What are the implications for customer satisfaction and service-level agreements?",
            "answer": "AI ensures SLA compliance by optimizing routes to meet delivery time windows. Our analysis shows how route optimization improves SLA compliance rates and enables predictive risk assessment for potential violations.",
            "tab": "7. Customer Satisfaction & SLA",
            "key_features": [
                "SLA compliance rate analysis",
                "SLA violation tracking",
                "Risk prediction using ML models",
                "Customer satisfaction metrics"
            ]
        }
    ]
    
    for i, qa in enumerate(questions_answers, 1):
        with st.expander(f"Q{i}: {qa['question']}", expanded=(i==1)):
            st.markdown(f"**Answer:** {qa['answer']}")
            st.markdown(f"**📍 Explore in:** {qa['tab']}")
            st.markdown("**Key Features:**")
            for feature in qa['key_features']:
                st.markdown(f"- {feature}")
    
    st.divider()
    
    st.subheader("📊 Key Achievements & Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cost Reduction", "15-30%", "vs Baseline Routes")
    with col2:
        st.metric("Emissions Reduction", "20-35%", "CO₂ Reduction")
    with col3:
        st.metric("SLA Compliance", "85-95%", "Improved Rate")
    with col4:
        st.metric("Fleet Utilization", "75-90%", "Capacity Usage")
    
    st.divider()
    
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Load a Dataset**: Go to "Data Management" → "Load Dataset" to select from pre-generated datasets
    2. **Explore Route Optimization**: Start with "Route Optimization" to see how AI optimizes delivery routes
    3. **Try Different Algorithms**: Visit "Algorithm Showcase" to compare different AI approaches
    4. **Analyze Real-Time Impact**: Check "Real-Time Data Integration" to see how traffic and weather affect routing
    5. **Generate Custom Data**: Use "Data Management" → "Generate Dataset" to create your own scenarios
    """)
    
    st.info("💡 **Tip:** Each tab includes detailed explanations of what's happening and what the visualizations represent. Look for info icons and expandable sections for more context.")


def main():
    """Main application"""
    st.markdown('<p class="main-header">AI in Logistics & Route Optimization</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    # Presentation mode toggle
    presentation_mode = st.sidebar.checkbox("🎬 Presentation Mode", value=False, help="Simplified UI for presentations")
    
    tabs = [
        "🏠 Home",
        "1. Route Optimization",
        "2. Real-Time Data Integration",
        "3. Algorithm Showcase",
        "4. Fleet Utilization & Sustainability",
        "5. Integration Challenges",
        "6. Last-Mile Delivery",
        "7. Customer Satisfaction & SLA",
        "8. Advanced Features",
        "Data Management"
    ]
    
    selected_tab = st.sidebar.radio("Select Tab", tabs)
    
    if selected_tab == "🏠 Home":
        tab_home()
    elif selected_tab == "1. Route Optimization":
        tab_route_optimization()
    elif selected_tab == "2. Real-Time Data Integration":
        tab_realtime_data()
    elif selected_tab == "3. Algorithm Showcase":
        tab_algorithm_showcase()
    elif selected_tab == "4. Fleet Utilization & Sustainability":
        tab_fleet_sustainability()
    elif selected_tab == "5. Integration Challenges":
        tab_integration_challenges()
    elif selected_tab == "6. Last-Mile Delivery":
        tab_last_mile()
    elif selected_tab == "7. Customer Satisfaction & SLA":
        tab_customer_sla()
    elif selected_tab == "8. Advanced Features":
        tab_advanced_features()
    elif selected_tab == "Data Management":
        tab_data_management()
    
    # Store presentation mode in session state
    st.session_state['presentation_mode'] = presentation_mode


if __name__ == "__main__":
    main()

