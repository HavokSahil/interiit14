"""
Quick test for 3-day Fast Loop simulation (1 hour test)
"""

import sys
sys.path.insert(0, '/home/sahil/Work/interiit14/Codes/multi-timescale-controller')

# Modify the ThreeDayFastLoopSimulation to run for 1 hour
from generate_3day_fastloop_logs import ThreeDayFastLoopSimulation

class OneHourFastLoopTest(ThreeDayFastLoopSimulation):
    def __init__(self):
        super().__init__(output_dir="test_fastloop_logs", audit_dir="test_fastloop_audit")
        # Override to 1 hour
        self.num_days = 1
        self.hours_per_day = 1
        self.total_steps = self.steps_per_hour * 1
        
        print(f"\n{'='*70}")
        print("1-HOUR FAST LOOP TEST")
        print(f"{'='*70}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Duration: 1 hour")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    print("Running 1-hour Fast Loop simulation test...")
    sim = OneHourFastLoopTest()
    sim.run()
