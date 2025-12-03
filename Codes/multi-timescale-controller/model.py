from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import math
import random

class PropagationModel(ABC):
    """Abstract base class for propagation models."""
    
    @abstractmethod
    def compute_received_power(self, tx_power: float, distance: float) -> float:
        """Compute received power in dBm."""
        pass


class PathLossModel(PropagationModel):
    """Free space path loss model."""
    
    def __init__(self, frequency_mhz: float = 2400, path_loss_exp: float = 2.0):
        self.freq = frequency_mhz
        self.n = path_loss_exp
    
    def compute_received_power(self, tx_power: float, distance: float) -> float:
        if distance == 0:
            return tx_power
        fspl = 20 * math.log10(distance) + 20 * math.log10(self.freq) - 27.55
        additional_loss = 10 * (self.n - 2) * math.log10(distance) if self.n != 2 else 0
        return tx_power - fspl - additional_loss


class MultipathFadingModel(PropagationModel):
    """Path loss with Rayleigh fading."""
    
    def __init__(self, base_model: PropagationModel, fading_margin_db: float = 10.0):
        self.base_model = base_model
        self.fading_margin = fading_margin_db
    
    def compute_received_power(self, tx_power: float, distance: float) -> float:
        base_power = self.base_model.compute_received_power(tx_power, distance)
        return base_power - self.fading_margin


class ClientMobility:
    """Handles client movement with velocity and direction."""
    
    def __init__(self, environment: 'Environment', 
                 max_velocity: float = 2.0, 
                 min_velocity: float = 0.5,
                 velocity_change_rate: float = 0.1,
                 direction_change_rate: float = math.pi / 8):
        self.env = environment
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.velocity_change_rate = velocity_change_rate
        self.direction_change_rate = direction_change_rate
    
    def random_walk(self, client: 'Client') -> Tuple[float, float]:
        """
        Update client position based on velocity and direction.
        Velocity and direction change gradually with random variations.
        Returns new (x, y) position within bounds.
        """
        # Randomly adjust direction (slightly left or right)
        direction_change = random.uniform(-self.direction_change_rate, self.direction_change_rate)
        client.direction += direction_change
        
        # Keep direction in [0, 2π) range
        client.direction = client.direction % (2 * math.pi)
        
        # Randomly adjust velocity
        velocity_change = random.uniform(-self.velocity_change_rate, self.velocity_change_rate)
        client.velocity += velocity_change
        
        # Clamp velocity to bounds
        client.velocity = max(self.min_velocity, min(self.max_velocity, client.velocity))
        
        # Compute new position
        new_x = client.x + client.velocity * math.cos(client.direction)
        new_y = client.y + client.velocity * math.sin(client.direction)
        
        # Handle boundary collisions with reflection
        if new_x < self.env.x_min or new_x > self.env.x_max:
            # Reflect horizontally
            client.direction = math.pi - client.direction
            new_x = max(self.env.x_min, min(new_x, self.env.x_max))
        
        if new_y < self.env.y_min or new_y > self.env.y_max:
            # Reflect vertically
            client.direction = -client.direction
            new_y = max(self.env.y_min, min(new_y, self.env.y_max))
        
        # Normalize direction after reflections
        client.direction = client.direction % (2 * math.pi)
        
        return new_x, new_y