"""
Reinforcement Learning (Q-learning) implementation for dynamic routing decisions
"""

import numpy as np
import random
from config import RL_LEARNING_RATE, RL_DISCOUNT_FACTOR, RL_EPSILON_START, RL_EPSILON_MIN, RL_EPSILON_DECAY


class QLearningRouter:
    """Q-learning agent for dynamic route optimization"""
    
    def __init__(self, distance_matrix, orders_df, delivery_locations, vehicle_info,
                 learning_rate=RL_LEARNING_RATE, discount_factor=RL_DISCOUNT_FACTOR,
                 epsilon_start=RL_EPSILON_START, epsilon_min=RL_EPSILON_MIN, epsilon_decay=RL_EPSILON_DECAY):
        self.distance_matrix = distance_matrix
        self.orders_df = orders_df
        self.delivery_locations = delivery_locations
        self.vehicle_info = vehicle_info
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.n_deliveries = len(delivery_locations)
        self.n_states = self.n_deliveries * 2  # Simplified state space
        self.n_actions = self.n_deliveries
        
        # Initialize Q-table
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.training_history = []
    
    def get_state(self, current_location, remaining_deliveries, time_of_day, traffic_level):
        """Encode state as integer"""
        # Ensure current_location is an integer
        current_location = int(current_location) if current_location is not None else 0
        
        # Simplified state encoding
        location_state = current_location % self.n_deliveries if self.n_deliveries > 0 else 0
        remaining_state = len(remaining_deliveries) % 2 if remaining_deliveries else 0
        time_state = (int(time_of_day) // 6) % 2  # Morning/afternoon
        traffic_state = {'light': 0, 'moderate': 1, 'heavy': 2, 'severe': 3}.get(traffic_level, 0) % 2
        
        state = location_state * 8 + remaining_state * 4 + time_state * 2 + traffic_state
        return int(min(state, self.n_states - 1))
    
    def get_available_actions(self, remaining_deliveries):
        """Get available actions (next deliveries to visit)"""
        return list(remaining_deliveries)
    
    def calculate_reward(self, from_location, to_location, time_taken, sla_violated=False):
        """Calculate reward for taking an action"""
        # Negative reward based on distance and time (we want to minimize)
        distance = self.distance_matrix[from_location][to_location]
        distance_cost = distance * 0.5
        
        time_cost = time_taken * 10
        
        # Penalty for SLA violations
        sla_penalty = 100 if sla_violated else 0
        
        reward = -(distance_cost + time_cost + sla_penalty)
        return reward
    
    def choose_action(self, state, available_actions, training=True):
        """Choose action using epsilon-greedy policy"""
        if len(available_actions) == 0:
            return None
        
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: best action according to Q-table
            # Map delivery indices to action indices (0 to n_actions-1)
            q_values = []
            for action_delivery_idx in available_actions:
                # Ensure action index is within bounds
                action_idx = min(action_delivery_idx, self.n_actions - 1)
                q_values.append(self.q_table[state, action_idx])
            
            if len(q_values) == 0:
                return random.choice(available_actions)
            
            best_action_idx = np.argmax(q_values)
            return available_actions[best_action_idx]
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula"""
        # Ensure state and action are integers and within bounds
        state = int(state)
        action_idx = int(min(action, self.n_actions - 1))
        next_state = int(next_state)
        
        # Ensure states are within bounds
        state = min(state, self.n_states - 1)
        next_state = min(next_state, self.n_states - 1)
        
        current_q = self.q_table[state, action_idx]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        self.q_table[state, action_idx] = current_q + self.learning_rate * (target_q - current_q)
    
    def train_episode(self, traffic_data=None, weather_data=None):
        """Train for one episode"""
        # Initialize episode
        remaining_deliveries = list(range(self.n_deliveries))
        current_location = 0  # Start at depot (index 0)
        total_reward = 0
        route = []
        time_of_day = 9  # Start at 9 AM
        
        # Get traffic level
        traffic_level = 'light'
        if traffic_data is not None and len(traffic_data) > 0:
            hour = time_of_day
            traffic_row = traffic_data[traffic_data['hour'] == hour]
            if len(traffic_row) > 0:
                traffic_level = traffic_row.iloc[0]['traffic_level']
        
        while len(remaining_deliveries) > 0:
            # Get current state
            state = self.get_state(current_location, remaining_deliveries, time_of_day, traffic_level)
            
            # Choose action
            available_actions = self.get_available_actions(remaining_deliveries)
            if len(available_actions) == 0:
                break
            
            action = self.choose_action(state, available_actions, training=True)
            
            # Execute action
            next_location = action
            distance = self.distance_matrix[current_location][next_location]
            time_taken = distance / 50.0 + 10 / 60  # Travel time + delivery time
            
            # Check SLA violation
            order = self.orders_df[self.orders_df['delivery_id'] == 
                                  self.delivery_locations.iloc[next_location]['delivery_id']]
            sla_violated = False
            if len(order) > 0:
                order = order.iloc[0]
                if time_of_day < order['time_window_start'] or time_of_day > order['time_window_end']:
                    sla_violated = True
            
            # Calculate reward
            reward = self.calculate_reward(current_location, next_location, time_taken, sla_violated)
            total_reward += reward
            
            # Update state
            remaining_deliveries.remove(next_location)
            route.append(next_location)
            current_location = next_location
            time_of_day += time_taken
            
            # Get next state
            if len(remaining_deliveries) > 0:
                next_state = self.get_state(current_location, remaining_deliveries, int(time_of_day), traffic_level)
                done = False
            else:
                next_state = state
                done = True
            
            # Update Q-table
            self.update_q_table(state, action, reward, next_state, done)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return route, total_reward
    
    def train(self, n_episodes=1000, traffic_data=None, weather_data=None):
        """Train the agent"""
        for episode in range(n_episodes):
            route, reward = self.train_episode(traffic_data, weather_data)
            self.training_history.append({
                'episode': episode,
                'reward': reward,
                'route_length': len(route)
            })
    
    def predict_route(self, traffic_data=None, weather_data=None):
        """Predict optimal route using learned policy"""
        remaining_deliveries = list(range(self.n_deliveries))
        current_location = 0
        route = []
        time_of_day = 9
        
        traffic_level = 'light'
        if traffic_data is not None and len(traffic_data) > 0:
            hour = time_of_day
            traffic_row = traffic_data[traffic_data['hour'] == hour]
            if len(traffic_row) > 0:
                traffic_level = traffic_row.iloc[0]['traffic_level']
        
        while len(remaining_deliveries) > 0:
            state = self.get_state(current_location, remaining_deliveries, time_of_day, traffic_level)
            available_actions = self.get_available_actions(remaining_deliveries)
            
            if len(available_actions) == 0:
                break
            
            # Use greedy policy (no exploration)
            action = self.choose_action(state, available_actions, training=False)
            
            route.append(action)
            remaining_deliveries.remove(action)
            current_location = action
            
            distance = self.distance_matrix[current_location][action] if current_location != action else 0
            time_of_day += distance / 50.0 + 10 / 60
        
        return route

