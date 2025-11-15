import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import joblib
import os
import json
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from numpy.linalg import inv, det
import random

from skopt.space import Real, Integer

class ChannelAgent:
    """
    Represents the "brain" for a single channel. It implements the
    Clustering-based Multi-Armed Bandit (CMAB) logic.
    """
    
    def __init__(self, channel_id, alpha, offline_model,weights):
        """
        Initializes the agent for a specific channel.
        
        Args:
            channel_id (int): The ID of this channel (e.g., 0-10).
            alpha (float): The exploration parameter (from UCB).
            offline_model (dict): A dictionary containing all loaded offline artifacts.
        """
        self.channel_id = channel_id
        self.alpha = alpha
        self.weights=weights

        # Load offline model components
        self.centroids_scaled = offline_model["centroids"]
        self.radii = offline_model["radii"]
        self.utilities_mu_bar = offline_model["utility_matrix"] # This is μ-bar
        
        self.k_clusters, self.n_actions = self.utilities_mu_bar.shape

        # Initialize online learning (regression) variables
        self.b_reg = np.zeros(self.n_actions) # b_t,e
        self.X_reg = np.identity(self.n_actions) # X_t,e
        self.mu_tilde = np.zeros(self.n_actions) # μ-tilde_t,e
        
        # Initialize agent's state
        self.current_cluster_id = -1
        self.current_centroid = None
        self.current_radius = np.inf
        self.current_avg_utility = np.zeros(self.n_actions) # μ-bar_k
        
        print(f"Agent {channel_id} initialized.")

    def _find_nearest_cluster(self, scaled_sf):
        """
        Finds the closest cluster (k*) for a given scaled Sensing Fingerprint.
        Uses Euclidean distance.
        """
        # Reshape sf to (1, n_features) for cdist
        sf_reshaped = scaled_sf.reshape(1, -1)
        
#        print("Scaled SF for cluster finding:", sf_reshaped, scaled_sf)
#        print(self.centroids_scaled)
        # Calculate distances to all K centroids
        distances = cdist(sf_reshaped, self.centroids_scaled, 'euclidean')
        # Return the index (k) of the minimum distance
        return np.argmin(distances)

    def _calculate_exploration_bonus(self, action_vec, epoch_t):
        """
        Calculates the UCB exploration bonus.
        (alpha * sqrt(x_a^T * X_inv * x_a * log(t+1)))
        """
        try:
            # We need the inverse of X_reg
            # Add a small value to diagonal for numerical stability
            X_inv = inv(self.X_reg + 1e-5 * np.identity(self.n_actions))
        except np.linalg.LinAlgError:
            # If matrix is singular, fall back to identity

            X_inv = np.identity(self.n_actions)
            
        confidence_term = np.sqrt(action_vec.T @ X_inv @ action_vec)
        # Add 1 to epoch_t to avoid log(0) or log(1)
        bonus = self.alpha * confidence_term * np.sqrt(np.log(epoch_t + 1))
        return bonus

    def update_model(self, last_scaled_sf, last_action_index, last_reward):
        """
        The core "Learn" step. Updates the agent's model based on
        the experience from the previous epoch.
        """
        
        # 1. Check for cluster change
        # Calculate distance from the last SF to the agent's current centroid
        # print("Last scaled SF for update:", last_scaled_sf)
        # print("Current centroid:", self.current_centroid)
        last_sf_reshaped = last_scaled_sf.reshape(1, -1)
        distance = cdist(last_sf_reshaped, self.current_centroid.reshape(1, -1), 'euclidean')

        if distance > self.current_radius:
            # --- CHANGE DETECTED: Find new cluster and reset ---
            new_cluster_id = self._find_nearest_cluster(last_scaled_sf)
            
            # Only reset if the cluster *actually* changes
            if new_cluster_id != self.current_cluster_id:
                # print(f"Agent {self.channel_id}: Cluster change! {self.current_cluster_id} -> {new_cluster_id}")
                self.current_cluster_id = new_cluster_id
                self.current_centroid = self.centroids_scaled[new_cluster_id]
                self.current_radius = self.radii[new_cluster_id]
                self.current_avg_utility = self.utilities_mu_bar[new_cluster_id]
                
                # Reset regression variables
                self.b_reg = np.zeros(self.n_actions)
                self.X_reg = np.identity(self.n_actions)
                self.mu_tilde = np.zeros(self.n_actions)
        
        # 2. --- NO CHANGE (or first run): Update regression model ---
        # This part runs *unless* a brand new cluster was just entered
        
        # Calculate residual reward (the "surprise")
        r_prime = last_reward - self.current_avg_utility[last_action_index]
        
        # Get one-hot action vector
        action_vec = np.zeros(self.n_actions)
        action_vec[last_action_index] = 1
        
        # Update b and X
        self.b_reg += r_prime * action_vec
        self.X_reg += np.outer(action_vec, action_vec)
        
        # 3. Re-calculate the fine-tuned correction vector (μ-tilde)
        try:
            # Add small value to diagonal for numerical stability (regularization)
            X_reg_stable = self.X_reg + 1e-5 * np.identity(self.n_actions)
            self.mu_tilde = inv(X_reg_stable) @ self.b_reg
        except np.linalg.LinAlgError:
            # In case of singular matrix, do not update mu_tilde
            pass 

    def initialize_state(self, initial_scaled_sf):
        """
        Called once at startup to set the agent's initial cluster.
        """
        # print(initial_scaled_sf)
        self.current_cluster_id = self._find_nearest_cluster(initial_scaled_sf)
        # print(self.current_cluster_id)
        self.current_centroid = self.centroids_scaled[self.current_cluster_id]
        self.current_radius = self.radii[self.current_cluster_id]
        self.current_avg_utility = self.utilities_mu_bar[self.current_cluster_id]
        print(f"Agent {self.channel_id} initialized to cluster {self.current_cluster_id}.")

    def predict_ucb_scores(self, epoch_t):
        """
        The "Estimate" step. Predicts the UCB score for all 10 actions.
        This function does NOT change the agent's state.
        """
        ucb_scores = np.zeros(self.n_actions)
        
        for j in range(self.n_actions):
            action_vec = np.zeros(self.n_actions)
            action_vec[j] = 1
            
            # 1. Pre-learned utility (from offline cluster)
            utility_prelearned = self.current_avg_utility[j]
            
            # 2. Fine-tuned utility (from online learning)
            # Use dot product, as mu_tilde is (n_actions,) and action_vec is (n_actions,)
            utility_finetuned = self.mu_tilde[j] 
            
            # 3. Exploration Bonus
            exploration_bonus = self._calculate_exploration_bonus(action_vec, epoch_t)
            
            # Total UCB score
            ucb_scores[j] = utility_prelearned + utility_finetuned + exploration_bonus
            
        return ucb_scores

class Allocator:
    """
    Solves the Multiple-Choice Knapsack Problem to allocate the budget.
    """
    def __init__(self, budget, action_costs):
        """
        Args:
            budget (int): The total budget (e.g., 890ms).
            action_costs (list): The cost of each action (e.g., [10, 20, ..., 100]).
        """
        self.budget = int(budget)
        self.action_costs = [int(cost) for cost in action_costs]
        self.n_actions = len(action_costs)
        
    def solve_knapsack(self, ucb_matrix):
        """
        Solves the budget allocation problem using Dynamic Programming.
        
        Args:
            ucb_matrix (np.array): An (n_channels x n_actions) matrix of UCB scores.
            
        Returns:
            list: A list of chosen action *indices* (one for each channel).
        """
        start = time.time()
        n_channels, n_actions = ucb_matrix.shape
        
        # dp_table[i][b] = Max score using channels 0..i-1 with budget b
        dp_table = np.full((n_channels + 1, self.budget + 1), -np.inf)
        # choices[i][b] = which action (j) was chosen for channel i-1 at budget b
        choices = np.zeros((n_channels + 1, self.budget + 1), dtype=int)
        
        # Base case: 0 channels, 0 budget = 0 reward
        dp_table[0, 0] = 0
        
        # Fill the DP table
        for i in range(1, n_channels + 1):
            channel_index = i - 1
            for b in range(self.budget + 1):
                for j in range(n_actions):
                    cost = self.action_costs[j]
                    if b >= cost:
                        # Reward for taking action j on this channel
                        reward = ucb_matrix[channel_index, j]
                        # Reward from previous state
                        prev_reward = dp_table[i - 1, b - cost]
                        
                        if prev_reward != -np.inf:
                            total_reward = reward + prev_reward
                            if total_reward > dp_table[i, b]:
                                dp_table[i, b] = total_reward
                                choices[i, b] = j
                                
        # Backtrack to find the chosen actions
        chosen_actions = []
        current_budget = self.budget
        
        # Find the best total reward at the end
        best_budget = np.argmax(dp_table[n_channels])
        
        for i in range(n_channels, 0, -1):
            action_index = choices[i, best_budget]
            chosen_actions.append(action_index)
            cost = self.action_costs[action_index]
            best_budget -= cost
            
        print("KNAPSACK TIME:", time.time()-start)
        return list(reversed(chosen_actions))

class OnlineController:
    """
    Orchestrates the entire online learning loop.
    """
    def __init__(self, num_channels, model_dir, epoch_budget, quick_sense_duration, alpha,weights,data, sf):
        print("Initializing Online Controller...")
        self.num_channels = num_channels
        self.epoch_budget = epoch_budget
        self.quick_sense_duration = quick_sense_duration
        self.data = data
        self.weights=weights
        self.epoch_t = 0
        
        # --- Load Offline Model Artifacts ---
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        self.clusterer = joblib.load(os.path.join(model_dir, "clusterer.joblib"))
        
        offline_model = {
            "centroids": self.clusterer.cluster_centers_,
            "radii": np.load(os.path.join(model_dir, "radii.npy")),
            "utility_matrix": np.load(os.path.join(model_dir, "utility_matrix.npy"))
        }
        
        with open(os.path.join(model_dir, "model_config.json"), 'r') as f:
            config = json.load(f)
            self.action_values = config["action_values"]
            self.k_clusters = config["k_clusters"]
            
        print("Offline model artifacts loaded.")

        # --- Create Allocator ---
        self.execution_budget = self.epoch_budget - (self.num_channels * self.quick_sense_duration)
        self.allocator = Allocator(self.execution_budget, self.action_values)
        
        # --- Create Channel Agents ---
        self.agents = [
            ChannelAgent(i, alpha, offline_model,weights) for i in range(self.num_channels)
        ]
        
        # --- Initialize History ---
        # Stores (scaled_sf, action_index, reward) from the last epoch
        self.history = [None] * self.num_channels
        
        self._initialize_agent_states(sf)


    #chaneg the code here

    def _mock_sense_all_channels(self, sf):
        """
        MOCK FUNCTION: Simulates getting 11 new Sensing Fingerprints.
        In a real system, you'd scan each channel for 'quick_sense_duration'.
        """
#        raw_sfs = []
#        damn=0
        # Get the feature names from the scaler



#        sf = (self.data.iloc[i,3:])
#        raw_sfs.append(sf.tolist())

#        for i in range(self.num_channels):
#            for j in range(len(raw_sfs[i])):
#                    damn=damn+raw_sfs[i][j]
#
        # feature_names = self.scaler.feature_names_in_
        # for i in range(self.num_channels):
        #     # Create a plausible random feature vector
        #     sf = {
        #         'mean_energy': np.random.normal(-70, 10),
        #         'variance': np.random.uniform(5, 20),
        #         'noise_floor': np.random.normal(-95, 2),
        #         'cca_busy_fraction': np.random.uniform(0, 1),
        #         'client_count': np.random.randint(0, 50)
        #     }
        #     # Ensure all features are present, fill with mean if not
        #     full_sf_dict = {name: sf.get(name, self.scaler.mean_[list(feature_names).index(name)]) for name in feature_names}
        #     # Order correctly
        #     raw_sfs.append([full_sf_dict[name] for name in feature_names])
            
        return np.array(sf)

    def _mock_execute_allocation(self, chosen_dwell_times, weights,sf):
        """
        Simulates executing dwell times and calculating rewards.
        # Reward aims to maximize variance (exploration potential) 
        while penalizing longer dwell times and noisy/occupied channels.
        """

        rewards = []
        rr_rewards=[]
        random_rewards=[]
        action_values = [150,300,450,600,750,900,1050]
        action_set = set(action_values)

        # --- Create random allocation baseline (for comparison) ---
        def generate_fast():
            while True:
                picks = random.choices(action_values, k=10)
                total = sum(picks)
                need = 2050 - total
                
                # choose the closest valid action value
                closest = min(action_values, key=lambda x: abs(x - need))
                return picks + [closest]

        result = generate_fast()

        total_ucb = total_rr = total_rnd = 0
        maxi=0


        for i in range(self.num_channels):
            
#]
            noise_floor       = sf[i][1]
            mean_energy       = sf[i][0]
            SNR = sf[i][2]

            dwell_ucb = chosen_dwell_times[i % self.num_channels]
            dwell_rr  = 2050 / 3
            dwell_rnd = result[i % self.num_channels]

            # --- Reward function emphasizing variance ---
            # The dwell time term acts as a denominator to penalize staying too long.
            # A small epsilon is added to prevent divide-by-zero.
            eps = 1e-6

            final=0

            def compute_reward(noise, energy, dwell,SNR,maxi=0):
                # variance term: high weight to encourage exploring variable channels
                # penalty term: based on noise, occupancy, and dwell time
                # print("weights:",weights)
                numerator = energy*weights['mean_energy'] - SNR*weights['SNR']+ noise*weights['noise_floor']
                # penalty   = (0.8 * noise + 0.7 * occ)
                reward = ((numerator)*(dwell))/(2050)
                return reward,numerator
            prev=0

            reward_ucb,maxi = compute_reward(noise_floor, mean_energy, dwell_ucb,SNR)
            reward_rr,how1  = compute_reward(noise_floor, mean_energy, dwell_rr,SNR)
            reward_rnd,how2 = compute_reward(noise_floor, mean_energy, dwell_rnd,SNR)

            total_ucb += reward_ucb
            total_rr  += reward_rr
            total_rnd += reward_rnd


            rewards.append(reward_ucb)
            rr_rewards.append(reward_rr)
            random_rewards.append(reward_rnd)
            if(maxi>final) :
                final=maxi

        np.sort(rewards)
        np.sort(rr_rewards)
        np.sort(random_rewards)

        sum1 = np.sum(rewards[-11:])
        sum2 = np.sum(rr_rewards[-11:])
        sum3 = np.sum(random_rewards[-11:])
        sum4=final

        return np.array(rewards), sum1,sum2,sum3,sum4

    

    def _initialize_agent_states(self, sf):
        """
        Run a first sense to get the initial state for all agents.
        """
        print("Running initial scan to initialize agent states...")
        raw_sfs = self._mock_sense_all_channels(sf)
        scaled_sfs = raw_sfs
        
        for i in range(self.num_channels):
            self.agents[i].initialize_state(scaled_sfs[i])

    def run_epoch_1(self, sf):
        """
        Runs one full epoch (Sense -> Update -> Predict -> Allocate -> Execute -> Store).
        """
        
        # --- 1. SENSE ---
        # Get the new state (e_t) for all channels
        current_raw_sfs = self._mock_sense_all_channels(sf=sf)
        current_scaled_sfs = (current_raw_sfs)
        #it is already scaled in mock sense function
        # current_scaled_sfs = self.scaler.transform(current_raw_sfs)
        
        # --- 2. UPDATE & PREDICT ---
        ucb_matrix = np.zeros((self.num_channels, len(self.action_values)))
        
        for i in range(self.num_channels):
            # A. Get history from last epoch (t-1)
            history_item = self.history[i]
            
            # B. Update model (this is the "Learn" step)
            if history_item is not None:
                last_scaled_sf, last_action_index, last_reward = history_item
                self.agents[i].update_model(last_scaled_sf, last_action_index, last_reward)
                
            # C. Predict UCB scores (for current epoch t)
            # Note: The prediction is based on the model *just updated*
            ucb_matrix[i] = self.agents[i].predict_ucb_scores(self.epoch_t)

        # --- 3. ALLOCATE ---
        chosen_action_indices = self.allocator.solve_knapsack(ucb_matrix)
        chosen_dwell_times = [self.action_values[idx] for idx in chosen_action_indices]

        # Step 1: Compute current sum and difference needed
        current_sum = sum(chosen_dwell_times)
        diff = 2050 - current_sum

        if diff != 0:
            # Step 2: Get sorted order (value + original index)
            sorted_vals = sorted(
                [(val, i) for i, val in enumerate(chosen_dwell_times)],
                key=lambda x: x[0]
            )
            
            # Smallest, middle, largest
            (min_val, min_idx), (mid_val, mid_idx), (max_val, max_idx) = sorted_vals
            
            # Step 3: Allocate differences
            add_to_max = (2/3) * diff
            add_to_mid = (1/3) * diff

            # Step 4: Update values in original order
            chosen_dwell_times[max_idx] += add_to_max
            chosen_dwell_times[mid_idx] += add_to_mid
            # min gets nothing
        return chosen_dwell_times, current_scaled_sfs, chosen_action_indices

    def run_epoch_2(self, sf, chosen_dwell_times, current_scaled_sfs, chosen_action_indices):
        # --- 4. EXECUTE ---
        actual_rewards,sum1,rr_rewards,random_rewards,optimal= self._mock_execute_allocation(chosen_dwell_times,self.weights, sf=sf)
        
        # --- 5. STORE HISTORY ---
        # Store the results from *this* epoch for use in the *next* epoch
        for i in range(self.num_channels):
            self.history[i] = (current_scaled_sfs[i], chosen_action_indices[i], actual_rewards[i])
            
        # --- 6. LOGGING ---
        # TOLOG: rl mab
        with open("rl.log", "a") as f:
            f.write(f"""--- Epoch {self.epoch_t} ---
Chosen Dwells: {chosen_dwell_times}
Rewards Got: {[round(r, 1) for r in actual_rewards]}
Total Reward: {sum1}
Total OPTIMAL Reward: {optimal}\n""")

        self.epoch_t += 1
        return sum1,rr_rewards,random_rewards,optimal, actual_rewards


weights={'variance': 0.0, 'mean_energy': 0.6727014066736716, 'occupied_fraction': 0.0, 'noise_floor': 0.1424049309128144, 'SNR': -0.06826676910128526, 'K_clusters': np.int64(40)}

MODEL_DIR = "offline_model_artifacts"  # Dir created by offline script
NUM_CHANNELS = 3
EPOCH_BUDGET = 2500
QUICK_SENSE_DURATION = 150  # 10ms per channel for sensing
ALPHA = 5  # UCB exploration parameter
# NUM_EPOCHS_TO_RUN = 1000

data = pd.read_csv("pre_processed_online.csv")  # Mock data for online learning
data = data.drop(columns=['variance'])
# data = data.drop(columns=['sample_id'])
# data = data.drop(columns=['reward'])

