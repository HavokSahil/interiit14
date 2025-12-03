
import unittest
from unittest.mock import MagicMock, patch
import networkx as nx
import torch
from sim import SimulationVisualizer, WirelessSimulation, AccessPoint, Client, Environment, PropagationModel, APMetricsManager

class TestGraphVisualization(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_env = MagicMock(spec=Environment)
        self.mock_env.x_min = 0
        self.mock_env.x_max = 100
        self.mock_env.y_min = 0
        self.mock_env.y_max = 100
        
        self.mock_prop_model = MagicMock(spec=PropagationModel)
        
        self.sim = WirelessSimulation(self.mock_env, self.mock_prop_model)
        
        # Create dummy APs
        self.ap1 = AccessPoint(id=0, x=10, y=10, tx_power=20, channel=1, bandwidth=20)
        self.ap2 = AccessPoint(id=1, x=90, y=90, tx_power=20, channel=6, bandwidth=20)
        self.sim.add_access_point(self.ap1)
        self.sim.add_access_point(self.ap2)
        
        # Mock GNN availability and model
        self.patcher = patch('sim.GNN_AVAILABLE', True)
        self.patcher.start()
        
        # Initialize visualizer (mocking pygame to avoid display issues)
        with patch('sim.pygame'):
            self.visualizer = SimulationVisualizer(self.sim)
            
        # Mock GNN model
        self.visualizer.gnn_model = MagicMock()
        # Mock predict_all_edges to return some edges
        # Returns (edge_index, edge_probs)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_probs = torch.tensor([0.8, 0.2], dtype=torch.float)
        self.visualizer.gnn_model.predict_all_edges.return_value = (edge_index, edge_probs)
        self.visualizer.gnn_device = 'cpu'

    def tearDown(self):
        self.patcher.stop()

    def test_predict_interference_graph_attributes(self):
        # Run prediction
        graph = self.visualizer._predict_interference_graph()
        
        self.assertIsNotNone(graph, "Graph should not be None")
        self.assertEqual(len(graph.nodes), 2, "Graph should have 2 nodes")
        
        # Check node attributes for AP 0
        node0 = graph.nodes[0]
        self.assertEqual(node0['x'], 10)
        self.assertEqual(node0['y'], 10)
        self.assertEqual(node0['channel'], 1)
        self.assertIn('load', node0)
        self.assertIn('num_clients', node0)
        
        # Check node attributes for AP 1
        node1 = graph.nodes[1]
        self.assertEqual(node1['x'], 90)
        self.assertEqual(node1['y'], 90)
        self.assertEqual(node1['channel'], 6)
        
        print("Verification Successful: Nodes have correct attributes.")

if __name__ == '__main__':
    unittest.main()
