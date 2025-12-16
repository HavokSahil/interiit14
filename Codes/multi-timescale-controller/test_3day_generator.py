"""
Quick test of 3-day log generator (1 hour simulation)
"""

import sys
sys.path.insert(0, '/home/sahil/Work/interiit14/Codes/multi-timescale-controller')

from generate_3day_logs import ThreeDaySimulation

# Create a short test (1 hour instead of 3 days)
class TestSimulation(ThreeDaySimulation):
    def __init__(self):
        super().__init__(
            output_dir="test_logs",
            audit_dir="test_audit"
        )
        # Override to just 1 hour for testing
        self.steps_per_hour = 60  # Faster
        self.hours_per_day = 1    # Just 1 hour
        self.num_days = 1         # Just 1 day
        self.total_steps = self.steps_per_hour * self.hours_per_day * self.num_days
        
        print(f"\nTEST MODE: Running 1-hour simulation ({self.total_steps} steps)")

if __name__ == "__main__":
    sim = TestSimulation()
    sim.run()
