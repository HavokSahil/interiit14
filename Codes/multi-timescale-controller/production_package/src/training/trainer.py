"""
Training Pipeline for Safe RL Agents.

Provides unified training interface for CQL, DQN, and PPO agents
with safety validation and logging.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import yaml
import json

from ..agents.cql import CQLAgent
from ..agents.dqn import DQNAgent
from ..agents.ppo import PPOAgent
from ..agents.rcpo import RCPOAgent
from ..dataset.dataset import RRMDataset, create_dataloaders


class Trainer:
    """
    Unified trainer for Safe RL agents.
    
    Supports training CQL (main), DQN, and PPO (baselines)
    with consistent logging and evaluation.
    """
    
    def __init__(
        self,
        agent: Union[CQLAgent, DQNAgent, PPOAgent, RCPOAgent],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: Optional[Dict] = None,
        save_dir: str = 'checkpoints',
        experiment_name: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            agent: RL agent to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
            save_dir: Directory for saving checkpoints
            experiment_name: Name for this experiment
        """
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        
        # Setup save directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent_type = type(agent).__name__
            experiment_name = f"{agent_type}_{timestamp}"
        
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('-inf')
        self.training_log = []
        
        # Learning rate scheduler
        self.lr_scheduler = None
        self._setup_lr_scheduler()
        
        # Get normalization from dataset
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'get_normalization_params'):
            mean, std = train_dataset.get_normalization_params()
            self.agent.set_normalization(mean, std)
    
    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler for the agent."""
        # Get optimizer from agent
        optimizer = None
        if hasattr(self.agent, 'optimizer'):
            optimizer = self.agent.optimizer
        elif hasattr(self.agent, 'q_optimizer'):  # For IQL
            optimizer = self.agent.q_optimizer
        
        if optimizer is not None:
            # Cosine annealing with warm restarts
            self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=20,  # Initial restart period
                T_mult=2,  # Period multiplier
                eta_min=1e-5  # Minimum learning rate
            )
    
    def train(
        self,
        epochs: int = 100,
        eval_freq: int = 10,
        save_freq: int = 20,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train the agent.
        
        Args:
            epochs: Number of training epochs
            eval_freq: Frequency of evaluation
            save_freq: Frequency of checkpoint saving
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                current_lr = self.lr_scheduler.get_last_lr()[0]
                train_metrics['learning_rate'] = current_lr
            
            # Evaluation
            if (epoch + 1) % eval_freq == 0:
                val_metrics = self._evaluate(self.val_loader)
                
                # Check for improvement
                val_score = self._compute_val_score(val_metrics)
                if val_score > self.best_val_metric:
                    self.best_val_metric = val_score
                    patience_counter = 0
                    self._save_checkpoint('best')
                else:
                    patience_counter += 1
                
                # Log
                log_entry = {
                    'epoch': epoch + 1,
                    'train': train_metrics,
                    'val': val_metrics
                }
                self.training_log.append(log_entry)
                
                if verbose:
                    self._print_progress(train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Final evaluation on test set
        test_metrics = self._evaluate(self.test_loader)
        
        # Save final results
        self._save_results(test_metrics)
        
        return {
            'training_log': self.training_log,
            'test_metrics': test_metrics,
            'best_val_metric': self.best_val_metric
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}
        
        if isinstance(self.agent, CQLAgent):
            for batch in self.train_loader:
                batch_dict = {
                    'states': batch['state'],
                    'actions': batch['action'].squeeze(-1),
                    'rewards': batch['reward'].squeeze(-1),
                    'next_states': batch['next_state'],
                    'dones': batch['done'].squeeze(-1),
                    'costs': batch['cost'].squeeze(-1)
                }
                metrics = self.agent.update(batch_dict)
                self._accumulate_metrics(epoch_metrics, metrics)
        
        elif isinstance(self.agent, DQNAgent):
            batch_size = self.train_loader.batch_size
            num_updates = len(self.agent.replay_buffer) // batch_size
            for _ in range(num_updates):
                batch = self.agent.replay_buffer.sample(batch_size)
                metrics = self.agent.update(batch)
                self._accumulate_metrics(epoch_metrics, metrics)
        
        elif isinstance(self.agent, (PPOAgent, RCPOAgent)):
            for batch in self.train_loader:
                batch_dict = {
                    'states': batch['state'],
                    'actions': batch['action'].squeeze(-1),
                    'rewards': batch['reward'].squeeze(-1),
                    'next_states': batch['next_state'],
                    'dones': batch['done'].squeeze(-1),
                    'costs': batch['cost'].squeeze(-1)
                }
                metrics = self.agent.update(batch_dict)
                self._accumulate_metrics(epoch_metrics, metrics)
        
        # Average metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def _accumulate_metrics(self, epoch_metrics: Dict, batch_metrics: Dict):
        """Accumulate batch metrics into epoch metrics."""
        for key, value in batch_metrics.items():
            if key not in epoch_metrics:
                epoch_metrics[key] = []
            epoch_metrics[key].append(value)
    
    def _evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate agent on a dataset."""
        return self.agent.evaluate(dataloader, use_safety_shield=True)
    
    def _compute_val_score(self, val_metrics: Dict) -> float:
        """Compute validation score for model selection."""
        # Higher is better: negative constraint violation rate + mean Q (for value-based)
        score = -val_metrics.get('constraint_violation_rate', 0)
        if 'mean_q' in val_metrics:
            score += val_metrics['mean_q'] * 0.1
        return score
    
    def _print_progress(self, train_metrics: Dict, val_metrics: Dict):
        """Print training progress."""
        print(f"\nEpoch {self.current_epoch}")
        print("-" * 50)
        
        print("Train:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        print("Validation:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list) and key == 'action_distribution':
                print(f"  {key}: {[f'{v:.2%}' for v in value]}")
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        filepath = self.save_dir / f'{name}.pt'
        self.agent.save(str(filepath))
    
    def _save_results(self, test_metrics: Dict):
        """Save final results."""
        results = {
            'config': self.config,
            'training_log': self.training_log,
            'test_metrics': test_metrics,
            'best_val_metric': self.best_val_metric
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        results = convert_types(results)
        
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)


def train_all_agents(
    dataset_path: str,
    config_path: Optional[str] = None,
    epochs: int = 100,
    save_dir: str = 'checkpoints'
) -> Dict[str, Dict]:
    """
    Train all three agents (CQL, DQN, PPO) and compare.
    
    Args:
        dataset_path: Path to dataset file
        config_path: Path to config file (optional)
        epochs: Number of training epochs
        save_dir: Directory for saving results
        
    Returns:
        Results for each agent
    """
    # Load config
    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create dataloaders
    batch_size = config.get('training', {}).get('batch_size', 256)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path, batch_size=batch_size
    )
    
    # Get dimensions from dataset
    train_dataset = train_loader.dataset
    state_dim = train_dataset.num_features
    action_dim = train_dataset.num_actions
    
    results = {}
    
    # Train CQL (Main)
    print("\n" + "=" * 60)
    print("Training CQL (Main Algorithm)")
    print("=" * 60)
    
    cql_agent = CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config.get('cql', {})
    )
    cql_trainer = Trainer(
        cql_agent, train_loader, val_loader, test_loader,
        config=config, save_dir=save_dir, experiment_name='CQL'
    )
    results['CQL'] = cql_trainer.train(epochs=epochs)
    
    # Train DQN (Baseline)
    print("\n" + "=" * 60)
    print("Training DQN (Baseline)")
    print("=" * 60)
    
    # Recreate dataloaders for fresh buffer
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path, batch_size=batch_size
    )
    
    dqn_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config.get('dqn', {})
    )
    # Populate replay buffer
    for batch in train_loader:
        for i in range(batch['state'].shape[0]):
            dqn_agent.replay_buffer.add(
                state=batch['state'][i].numpy(),
                action=batch['action'][i].item(),
                reward=batch['reward'][i].item(),
                next_state=batch['next_state'][i].numpy(),
                done=batch['done'][i].item() > 0.5,
                cost=batch['cost'][i].item()
            )
    
    dqn_trainer = Trainer(
        dqn_agent, train_loader, val_loader, test_loader,
        config=config, save_dir=save_dir, experiment_name='DQN'
    )
    results['DQN'] = dqn_trainer.train(epochs=epochs)
    
    # Train PPO (Baseline)
    print("\n" + "=" * 60)
    print("Training PPO (Baseline)")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path, batch_size=batch_size
    )
    
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config.get('ppo', {})
    )
    ppo_trainer = Trainer(
        ppo_agent, train_loader, val_loader, test_loader,
        config=config, save_dir=save_dir, experiment_name='PPO'
    )
    results['PPO'] = ppo_trainer.train(epochs=epochs)
    
    return results

