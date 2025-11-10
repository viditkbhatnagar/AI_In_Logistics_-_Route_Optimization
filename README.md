# AI in Logistics & Route Optimization

A comprehensive Streamlit application demonstrating AI techniques for supply chain optimization, specifically focusing on logistics and route optimization.

## Project Overview

This project addresses 7 key questions about AI in logistics and route optimization:

1. **How can AI optimize delivery routes and reduce transportation costs?**
2. **What role do real-time traffic, weather, and demand data play in dynamic routing?**
3. **Which algorithms are commonly used (e.g., genetic algorithms, reinforcement learning, clustering)?**
4. **How can AI improve fleet utilization and reduce carbon footprint?**
5. **What are the challenges in integrating AI with existing logistics systems?**
6. **How does AI contribute to last-mile delivery efficiency?**
7. **What are the implications for customer satisfaction and service-level agreements?**

## Features

### Interactive Tabs

- **Route Optimization & Cost Reduction**: Genetic Algorithm implementation for VRP optimization
- **Real-Time Data Integration**: Dynamic routing with traffic, weather, and demand data
- **Algorithm Showcase**: Interactive demonstrations of GA, RL, and Clustering algorithms
- **Fleet Utilization & Sustainability**: Carbon footprint analysis and fleet efficiency metrics
- **Integration Challenges**: Discussion of technical challenges and solutions
- **Last-Mile Delivery**: Optimization strategies for final delivery leg
- **Customer Satisfaction & SLA**: Compliance tracking and risk prediction
- **Data Management**: Dataset generation and management tools

### Algorithms Implemented

- **Genetic Algorithm**: Vehicle Routing Problem (VRP) solver with visualization
- **Reinforcement Learning**: Q-learning for dynamic routing decisions
- **K-Means Clustering**: Delivery zone optimization
- **Machine Learning Models**: Delivery time prediction and SLA risk assessment

### Synthetic Data Generation

- Realistic delivery locations with coordinates
- Customer orders with priorities and time windows
- Vehicle fleet with capacity constraints
- Traffic patterns (24-hour cycles)
- Weather data (7-day forecasts)
- Demand forecasts (30-day projections)

## Installation

### Local Development (Recommended: Use Virtual Environment)

1. Clone the repository:
```bash
git clone <repository-url>
cd AI_In_Logistics_&_Route_Optimization
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Generate pre-made datasets (optional):
```bash
python data_generator.py
```

### Streamlit Cloud Deployment

**Important:** Streamlit Cloud automatically installs dependencies from `requirements.txt`, so you don't need to install anything manually on their servers.

1. **Push your code to GitHub** (make sure `requirements.txt` is in the root directory)

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set **Main file path** to: `app.py`
   - Streamlit Cloud will automatically:
     - Detect `requirements.txt`
     - Install all dependencies
     - Deploy your app

3. **Important Notes for Streamlit Cloud:**
   - ✅ `requirements.txt` is already properly formatted
   - ✅ All pre-generated datasets in `data/pre_generated/` will be included
   - ✅ No need to install anything manually - Streamlit Cloud handles it
   - ⚠️ Make sure all data files are committed to Git (they are in `data/pre_generated/`)
   - ⚠️ For large datasets, consider using Streamlit Cloud's file size limits

**Note:** For local development, always use a virtual environment to avoid conflicts with other projects. For Streamlit Cloud deployment, just ensure `requirements.txt` is in your repository root - Streamlit Cloud will handle the rest automatically!

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Navigate through the tabs using the sidebar to explore different aspects of AI in logistics

## Project Structure

```
AI_In_Logistics_&_Route_Optimization/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── config.py                      # Configuration settings
├── data_generator.py              # Synthetic data generation
├── utils/
│   ├── route_optimizer.py         # Route optimization logic
│   ├── visualization.py           # Plotting functions
│   ├── metrics.py                 # KPI calculations
│   ├── data_loader.py             # Data loading utilities
│   └── carbon_calculator.py       # Emissions calculations
├── algorithms/
│   ├── genetic_algorithm.py       # GA implementation
│   ├── reinforcement_learning.py  # RL implementation
│   ├── clustering.py              # Clustering implementation
│   └── ml_models.py               # ML models
└── data/
    ├── datasets/                  # Generated datasets
    └── pre_generated/             # 5 pre-made datasets
```

## Pre-Generated Datasets

The application includes 5 pre-generated datasets:

1. **small_urban**: 50 deliveries, 5 vehicles, 1 depot
2. **medium_city**: 200 deliveries, 15 vehicles, 2 depots
3. **large_metropolitan**: 500 deliveries, 30 vehicles, 3 depots
4. **multi_city_network**: 1000 deliveries, 50 vehicles, 5 depots
5. **peak_season**: 800 deliveries, 25 vehicles, 3 depots

## Key Metrics Tracked

- **Cost Metrics**: Total cost, fuel cost, driver cost, maintenance cost
- **Distance & Time**: Total distance, total time, deliveries per hour
- **Fleet Utilization**: Weight utilization, volume utilization, time utilization
- **Environmental Impact**: CO2 emissions, emissions reduction percentage
- **SLA Compliance**: Compliance rate, violations by priority level
- **Efficiency**: Cost per delivery, distance per delivery, time per delivery

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation
- **Plotly & Folium**: Interactive visualizations and maps
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Advanced ML models
- **Geopy**: Geocoding and distance calculations

## Contributing

This is an academic project. For questions or suggestions, please refer to the project documentation.

## License

This project is for educational purposes.

