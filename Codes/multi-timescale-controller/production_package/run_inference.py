#!/usr/bin/env python3
"""
Production Inference Script

Loads the ensemble model and generates predictions.

Usage:
    python run_inference.py --input data/rrm_dataset_expanded.h5 --output predictions.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import h5py
import yaml
from datetime import datetime, timedelta

from src.agents.general_ensemble import GeneralEnsembleAgent
from src.agents.safety import SafetyConfig
from src.explainability.explainer import ActionExplainer


def load_ensemble(model_dir, config_path):
    """Load production ensemble."""
    model_path = Path(model_dir)
    
    # Load metadata
    metadata_file = model_path / 'production_metadata.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create safety config
    safety_config_dict = config.get('safety', {})
    safety_config = SafetyConfig(
        tx_power_hard_min=safety_config_dict['hard']['tx_power_min'],
        tx_power_hard_max=safety_config_dict['hard']['tx_power_max'],
        obss_pd_hard_min=safety_config_dict['hard']['obss_pd_min'],
        obss_pd_hard_max=safety_config_dict['hard']['obss_pd_max'],
        tx_power_shield_min=safety_config_dict['shield']['tx_power_min'],
        tx_power_shield_max=safety_config_dict['shield']['tx_power_max'],
        obss_pd_shield_min=safety_config_dict['shield']['obss_pd_min'],
        obss_pd_shield_max=safety_config_dict['shield']['obss_pd_max']
    )
    
    # Load agents
    from src.agents.iql import IQLAgent
    from src.agents.rcpo import RCPOAgent
    from src.agents.cql import CQLAgent
    from src.agents.dqn import DQNAgent
    from src.agents.ppo import PPOAgent
    from src.agents.bcq import BCQAgent
    
    agent_map = {
        'IQL': IQLAgent,
        'RCPO': RCPOAgent,
        'CQL': CQLAgent,
        'DQN': DQNAgent,
        'PPO': PPOAgent,
        'BCQ': BCQAgent
    }
    
    agents = []
    agent_names = metadata['models']
    
    state_dim = 15
    action_dim = 9
    
    # Define valid parameters for each agent class
    import inspect
    valid_params = {}
    for model_name in agent_names:
        agent_class = agent_map.get(model_name.upper())
        if agent_class:
            sig = inspect.signature(agent_class.__init__)
            valid_params[model_name.upper()] = set(sig.parameters.keys()) - {'self'}
    
    for model_name in agent_names:
        agent_class = agent_map.get(model_name.upper())
        if not agent_class:
            continue
        
        # Load hyperparameters
        optuna_results_path = Path(f'tuning_results/{model_name.upper()}/optuna_results.json')
        if optuna_results_path.exists():
            with open(optuna_results_path, 'r') as f:
                optuna_data = json.load(f)
            best_trial = optuna_data.get('best_trial', {})
            hyperparams = best_trial.get('params', config.get(model_name.lower(), {}))
        else:
            hyperparams = config.get(model_name.lower(), {})
        
        # Filter hyperparameters to only include valid ones
        valid_hyperparams = {}
        valid_set = valid_params.get(model_name.upper(), set())
        for key, value in hyperparams.items():
            if key in valid_set:
                valid_hyperparams[key] = value
        
        # Create agent
        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            safety_config=safety_config,
            hidden_dims=config.get('network', {}).get('hidden_layers', [256, 256, 128]),
            **valid_hyperparams
        )
        
        # Load checkpoint
        agent_file = model_path / f'agent_{agent_names.index(model_name)}.pt'
        if agent_file.exists():
            agent.load(str(agent_file))
        else:
            # Try standard checkpoint
            checkpoint_paths = [
                Path(f'checkpoints/{model_name}/best.pt'),
                Path(f'checkpoints/{model_name}/model.pt')
            ]
            for checkpoint_path in checkpoint_paths:
                if checkpoint_path.exists():
                    agent.load(str(checkpoint_path))
                    break
        
        agents.append(agent)
    
    # Create ensemble
    ensemble = GeneralEnsembleAgent(
        agents=agents,
        voting_method=metadata['voting_method'],
        weights=metadata['weights'],
        agent_names=agent_names
    )
    
    # Set normalization
    norm_data = metadata.get('normalization', {})
    if norm_data:
        mean = np.array(norm_data['mean'])
        std = np.array(norm_data['std'])
        ensemble.set_normalization(mean, std)
    
    return ensemble, metadata


mapper = { "state_edge_p10_throughput": "throughput_mbps_p10", "state_avg_throughput": "throughput_mbps_p50", "state_p95_retry_rate": "retry_rate_p95", "state_p95_per": "retry_rate_p95", "state_median_rssi": "rssi_p50" }
def calculate_qoe(row):
    """Calculate QoE score."""
    edge_tput = row[mapper['state_edge_p10_throughput']]
    avg_tput = row[mapper['state_avg_throughput']]
    retry_rate = row[mapper['state_p95_retry_rate']] * 100
    per = row[mapper['state_p95_per']] * 100
    rssi = row[mapper['state_median_rssi']]
    
    edge_score = min(edge_tput / 200.0, 1.0)
    avg_score = min(avg_tput / 500.0, 1.0)
    
    if retry_rate <= 8.0:
        retry_score = 1.0
    elif retry_rate <= 15.0:
        retry_score = 1.0 - (retry_rate - 8.0) / 7.0 * 0.5
    else:
        retry_score = max(0.0, 1.0 - (retry_rate - 15.0) / 10.0)
    
    if per <= 5.0:
        per_score = 1.0
    elif per <= 10.0:
        per_score = 1.0 - (per - 5.0) / 5.0 * 0.5
    else:
        per_score = max(0.0, 1.0 - (per - 10.0) / 10.0)
    
    if -70 <= rssi <= -65:
        rssi_bonus = 0.1
    elif rssi > -65:
        rssi_bonus = 0.05
    elif rssi < -80:
        rssi_bonus = -0.1
    else:
        rssi_bonus = 0.0
    
    qoe = 0.40 * edge_score + 0.20 * avg_score + 0.25 * retry_score + 0.15 * per_score + rssi_bonus
    return round(max(0.0, min(1.0, qoe)), 4)


def get_action_name(action_id):
    """Get action name."""
    names = {
        0: "+2 dBm Tx Power",
        1: "-2 dBm Tx Power",
        2: "+4 dB OBSS-PD",
        3: "-4 dB OBSS-PD",
        4: "Increase Channel Width",
        5: "Decrease Channel Width",
        6: "Increase Channel Number",
        7: "Decrease Channel Number",
        8: "No-op"
    }
    return names.get(action_id, f"Unknown_{action_id}")


def get_q_values_from_ensemble(ensemble, state):
    """Get Q-values from ensemble for explainability."""
    try:
        # Try to get Q-values from select_action metadata (if using q_average)
        _, metadata = ensemble.select_action(state, deterministic=True, use_safety_shield=False)
        if 'q_values' in metadata:
            return np.array(metadata['q_values'])
        
        # Fallback: Get Q-values from individual agents and average them
        q_values_list = []
        for agent in ensemble.agents:
            q_vals = ensemble._get_q_values(agent, state)
            if q_vals is not None:
                q_values_list.append(q_vals.cpu().numpy()[0])
        
        if q_values_list:
            # Average Q-values (weighted by ensemble weights)
            weights = ensemble.weights if hasattr(ensemble, 'weights') else [1.0 / len(q_values_list)] * len(q_values_list)
            avg_q = np.average(q_values_list, axis=0, weights=weights[:len(q_values_list)])
            return avg_q
    except Exception:
        pass
    return None


def run_inference_on_state(row, ensemble, explainer, denormalize_fn):
    state = list(row.values())
    try:
        action, _ = ensemble.select_action(state, deterministic=True, use_safety_shield=True)
        action = int(action)
    except Exception as e:
        action = 8  # No-op on error
    
    try:
        is_safe = ensemble.safety.is_action_safe(state, action, ensemble.denormalize_state if hasattr(ensemble, 'denormalize_state') else denormalize_fn)
        status = "SAFE" if is_safe else "BLOCKED"
    except:
        status = "SAFE"
    
    # Get Q-values for explainability
    q_values = get_q_values_from_ensemble(ensemble, state)
    
    # Generate explanation
    try:
        explanation = explainer.explain_action(
            state=state,
            action=action,
            q_values=q_values,
            agent_type="Ensemble",
            denormalize_fn=denormalize_fn
        )
        reason = explanation.primary_reason.value
        confidence = explanation.confidence
    except Exception as e:
        reason = "HIGH_Q_VALUE"
        confidence = 0.5
    
    current_qoe = calculate_qoe(row)
    action_name = get_action_name(action)
    
    return {
        'ACTION': action_name,
        'REASON': reason,
        'CONFIDENCE': f"{confidence:.2%}",
        'STATUS': status,
        'Current_QoE': current_qoe
    }

def run_inference(input_path, output_path, model_dir, config_path):
    """Run inference on dataset."""
    print("Loading ensemble model...")
    ensemble, metadata = load_ensemble(model_dir, config_path)
    print(f"  Models: {', '.join(metadata['models'])}")
    
    print("\nLoading dataset...")
    with h5py.File(input_path, 'r') as f:
        states = f['states'][:]
        feature_names = [
            'client_count', 'median_rssi', 'p95_retry_rate', 'p95_per',
            'channel_utilization', 'avg_throughput', 'edge_p10_throughput',
            'neighbor_ap_rssi', 'obss_pd_threshold', 'tx_power',
            'noise_floor', 'channel_width', 'airtime_usage', 'cca_busy', 'roaming_rate'
        ]
        data = {}
        for i, name in enumerate(feature_names):
            data[f'state_{name}'] = states[:, i]
        df = pd.DataFrame(data)
    
    print(f"  Loaded {len(df)} samples")
    
    # Initialize explainer
    print("\nInitializing explainability module...")
    explainer = ActionExplainer(safety_module=ensemble.safety if hasattr(ensemble, 'safety') else None)
    
    print("\nGenerating predictions with explanations...")
    results = []
    state_cols = [col for col in df.columns if col.startswith('state_')]
    
    # Create denormalization function
    def denormalize_fn(state):
        mean = 0
        if hasattr(ensemble, 'state_mean') and ensemble.state_mean is not None:
            mean = ensemble.state_mean
        std = ensemble.state_std
        return state * std + mean
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        
        state = row[state_cols].values.astype(np.float32)
        
        results.append(run_inference_on_state(row, ensemble, explainer, denormalize_fn))
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"  Rows: {len(results_df)}")
    print(f"  Mean QoE: {results_df['Current_QoE'].mean():.4f}")
    print(f"  Safe Actions: {(results_df['STATUS'] == 'SAFE').sum() / len(results_df) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Run Ensemble Inference')
    parser.add_argument('--input', default='data/rrm_dataset_expanded.h5',
                       help='Input dataset path')
    parser.add_argument('--output', default='predictions.csv',
                       help='Output predictions CSV')
    parser.add_argument('--model-dir', default='model',
                       help='Model directory')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    run_inference(args.input, args.output, args.model_dir, args.config)


if __name__ == '__main__':
    main()
