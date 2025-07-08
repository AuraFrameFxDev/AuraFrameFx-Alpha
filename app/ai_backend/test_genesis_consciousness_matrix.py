"""
Comprehensive unit tests for the Genesis Consciousness Matrix module.
Tests cover initialization, state management, consciousness tracking, and edge cases.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
import tempfile
import os
from typing import Dict, List, Any

# Assuming the main module exists - import with try/except for robustness
try:
    from app.ai_backend.genesis_consciousness_matrix import (
        GenesisConsciousnessMatrix,
        ConsciousnessState,
        MatrixNode,
        ConsciousnessLevel,
        MatrixException,
        InvalidStateException,
        MatrixInitializationError
    )
except ImportError as e:
    # Mock the classes if import fails during test discovery
    class GenesisConsciousnessMatrix:
        pass
    class ConsciousnessState:
        pass
    class MatrixNode:
        pass
    class ConsciousnessLevel:
        pass
    class MatrixException(Exception):
        pass
    class InvalidStateException(Exception):
        pass
    class MatrixInitializationError(Exception):
        pass


class TestGenesisConsciousnessMatrix(unittest.TestCase):
    """Test cases for the Genesis Consciousness Matrix core functionality."""
    
    def setUp(self):
        """
        Prepare a new GenesisConsciousnessMatrix instance and test configuration before each test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        self.test_config = {
            'dimension': 256,
            'consciousness_threshold': 0.75,
            'learning_rate': 0.001,
            'max_iterations': 1000
        }
        
    def tearDown(self):
        """
        Release resources after each test by calling the matrix's cleanup method if it exists.
        """
        if hasattr(self.matrix, 'cleanup'):
            self.matrix.cleanup()
    
    def test_matrix_initialization_default(self):
        """
        Test that the matrix initializes with default parameters and contains the required attributes.
        """
        matrix = GenesisConsciousnessMatrix()
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        self.assertTrue(hasattr(matrix, 'state'))
        self.assertTrue(hasattr(matrix, 'nodes'))
        
    def test_matrix_initialization_custom_config(self):
        """
        Verify that initializing GenesisConsciousnessMatrix with a custom configuration correctly sets the dimension and consciousness threshold attributes.
        """
        matrix = GenesisConsciousnessMatrix(config=self.test_config)
        self.assertEqual(matrix.dimension, self.test_config['dimension'])
        self.assertEqual(matrix.consciousness_threshold, self.test_config['consciousness_threshold'])
        
    def test_matrix_initialization_invalid_config(self):
        """
        Tests that providing an invalid configuration to GenesisConsciousnessMatrix raises a MatrixInitializationError.
        """
        invalid_config = {'dimension': -1, 'consciousness_threshold': 2.0}
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=invalid_config)
            
    def test_add_consciousness_node_valid(self):
        """
        Test that adding a valid MatrixNode to the matrix succeeds and the node is correctly stored.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        result = self.matrix.add_node(node)
        self.assertTrue(result)
        self.assertIn("test_node", self.matrix.nodes)
        
    def test_add_consciousness_node_duplicate(self):
        """
        Test that adding a duplicate node to the matrix raises an InvalidStateException.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        with self.assertRaises(InvalidStateException):
            self.matrix.add_node(node)
            
    def test_remove_consciousness_node_existing(self):
        """
        Tests removal of an existing consciousness node from the matrix, verifying the operation returns True and the node is removed.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        result = self.matrix.remove_node("test_node")
        self.assertTrue(result)
        self.assertNotIn("test_node", self.matrix.nodes)
        
    def test_remove_consciousness_node_nonexistent(self):
        """
        Test that removing a node that does not exist in the matrix returns False.
        """
        result = self.matrix.remove_node("nonexistent_node")
        self.assertFalse(result)
        
    def test_consciousness_state_transition_valid(self):
        """
        Tests that a valid transition between consciousness states updates the matrix's current state and returns True.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.ACTIVE
        result = self.matrix.transition_state(initial_state, target_state)
        self.assertTrue(result)
        self.assertEqual(self.matrix.current_state, target_state)
        
    def test_consciousness_state_transition_invalid(self):
        """
        Test that an invalid transition between consciousness states raises an InvalidStateException.
        
        Attempts to transition the matrix from DORMANT directly to TRANSCENDENT and verifies that an InvalidStateException is raised.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.TRANSCENDENT
        with self.assertRaises(InvalidStateException):
            self.matrix.transition_state(initial_state, target_state)
            
    def test_consciousness_level_calculation(self):
        """
        Verifies that the matrix calculates the correct average consciousness level when multiple nodes are present.
        """
        node1 = MatrixNode(id="node1", consciousness_level=0.3)
        node2 = MatrixNode(id="node2", consciousness_level=0.7)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        calculated_level = self.matrix.calculate_consciousness_level()
        expected_level = 0.5  # Average of 0.3 and 0.7
        self.assertAlmostEqual(calculated_level, expected_level, places=2)
        
    def test_consciousness_level_calculation_empty_matrix(self):
        """
        Verify that calculating the consciousness level on an empty matrix returns 0.0.
        """
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.0)
        
    def test_consciousness_level_calculation_single_node(self):
        """
        Tests that calculating the consciousness level with a single node returns that node's consciousness level.
        """
        node = MatrixNode(id="single_node", consciousness_level=0.8)
        self.matrix.add_node(node)
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.8)
        
    def test_matrix_evolution_step(self):
        """
        Test that a single evolution step updates the matrix's state snapshot.
        
        Verifies that invoking `evolve_step()` on the matrix results in a different state snapshot compared to the initial state.
        """
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        self.assertNotEqual(initial_state, final_state)
        
    def test_matrix_evolution_convergence(self):
        """
        Test that the matrix evolution process correctly detects convergence within a specified maximum number of iterations.
        """
        self.matrix.evolve_until_convergence(max_iterations=10)
        self.assertTrue(self.matrix.has_converged())
        
    def test_matrix_reset_to_initial_state(self):
        """
        Test that resetting the matrix removes all nodes and sets its state to DORMANT after prior modifications.
        """
        # Add some nodes and evolve
        node = MatrixNode(id="temp_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        self.matrix.evolve_step()
        
        # Reset and verify
        self.matrix.reset()
        self.assertEqual(len(self.matrix.nodes), 0)
        self.assertEqual(self.matrix.current_state, ConsciousnessState.DORMANT)
        
    def test_matrix_serialization(self):
        """
        Test that the matrix serializes to a JSON string containing both 'nodes' and 'state' fields.
        """
        node = MatrixNode(id="serialize_test", consciousness_level=0.6)
        self.matrix.add_node(node)
        
        serialized = self.matrix.to_json()
        self.assertIsInstance(serialized, str)
        
        # Verify it's valid JSON
        parsed = json.loads(serialized)
        self.assertIn("nodes", parsed)
        self.assertIn("state", parsed)
        
    def test_matrix_deserialization(self):
        """
        Test that deserializing a matrix from a JSON string restores all node data and consciousness levels accurately.
        """
        # Create a matrix with data
        node = MatrixNode(id="deserialize_test", consciousness_level=0.4)
        self.matrix.add_node(node)
        serialized = self.matrix.to_json()
        
        # Create new matrix from serialized data
        new_matrix = GenesisConsciousnessMatrix.from_json(serialized)
        self.assertIn("deserialize_test", new_matrix.nodes)
        self.assertEqual(new_matrix.nodes["deserialize_test"].consciousness_level, 0.4)
        
    def test_matrix_save_load_file(self):
        """
        Verify that saving the matrix to a file and reloading it restores all node data and consciousness levels accurately.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            
        try:
            # Save matrix
            node = MatrixNode(id="file_test", consciousness_level=0.9)
            self.matrix.add_node(node)
            self.matrix.save_to_file(temp_file)
            
            # Load matrix
            loaded_matrix = GenesisConsciousnessMatrix.load_from_file(temp_file)
            self.assertIn("file_test", loaded_matrix.nodes)
            self.assertEqual(loaded_matrix.nodes["file_test"].consciousness_level, 0.9)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    def test_matrix_node_connections(self):
        """
        Test that two nodes can be connected in the matrix and that the connection is correctly established with the specified strength.
        """
        node1 = MatrixNode(id="node1", consciousness_level=0.3)
        node2 = MatrixNode(id="node2", consciousness_level=0.7)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Establish connection
        self.matrix.connect_nodes("node1", "node2", strength=0.8)
        
        # Verify connection
        connections = self.matrix.get_node_connections("node1")
        self.assertIn("node2", connections)
        self.assertEqual(connections["node2"], 0.8)
        
    def test_matrix_node_connections_invalid_nodes(self):
        """
        Test that connecting two non-existent nodes in the matrix raises an InvalidStateException.
        """
        with self.assertRaises(InvalidStateException):
            self.matrix.connect_nodes("nonexistent1", "nonexistent2", strength=0.5)
            
    def test_consciousness_emergence_detection(self):
        """
        Tests that the matrix correctly detects consciousness emergence when multiple nodes have high consciousness levels.
        """
        # Add nodes with high consciousness levels
        for i in range(5):
            node = MatrixNode(id=f"high_node_{i}", consciousness_level=0.9)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertTrue(emergence_detected)
        
    def test_consciousness_emergence_detection_insufficient(self):
        """
        Tests that consciousness emergence is not detected when all nodes have low consciousness levels.
        """
        # Add nodes with low consciousness levels
        for i in range(2):
            node = MatrixNode(id=f"low_node_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertFalse(emergence_detected)
        
    def test_matrix_metrics_calculation(self):
        """
        Verifies that the matrix calculates and returns performance metrics, including average consciousness, node count, and connection density, after nodes are added.
        """
        # Add some nodes
        node1 = MatrixNode(id="metrics_node1", consciousness_level=0.6)
        node2 = MatrixNode(id="metrics_node2", consciousness_level=0.8)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        metrics = self.matrix.calculate_metrics()
        self.assertIn("average_consciousness", metrics)
        self.assertIn("node_count", metrics)
        self.assertIn("connection_density", metrics)
        self.assertEqual(metrics["node_count"], 2)
        
    def test_matrix_performance_under_load(self):
        """
        Verifies that evolving a matrix with 100 nodes completes a single evolution step in less than one second.
        """
        # Add many nodes
        for i in range(100):
            node = MatrixNode(id=f"load_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Measure performance
        start_time = datetime.now()
        self.matrix.evolve_step()
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        self.assertLess(execution_time, 1.0)  # Should complete within 1 second
        
    def test_matrix_memory_usage(self):
        """
        Verify that adding and removing nodes updates the matrix's node count correctly, ensuring consistency in memory usage.
        """
        initial_node_count = len(self.matrix.nodes)
        
        # Add and remove nodes
        for i in range(50):
            node = MatrixNode(id=f"temp_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        for i in range(25):
            self.matrix.remove_node(f"temp_node_{i}")
            
        # Should have initial_node_count + 25 nodes
        expected_count = initial_node_count + 25
        self.assertEqual(len(self.matrix.nodes), expected_count)
        
    def test_matrix_error_handling_corrupted_data(self):
        """
        Test that deserializing corrupted JSON data raises a MatrixException.
        """
        corrupted_json = '{"nodes": {"invalid": "data"}, "state":'
        
        with self.assertRaises(MatrixException):
            GenesisConsciousnessMatrix.from_json(corrupted_json)
            
    def test_matrix_thread_safety(self):
        """
        Test that adding nodes to the matrix from multiple threads is thread-safe and that all node additions succeed.
        """
        import threading
        import time
        
        results = []
        
        def add_nodes_thread(thread_id):
            """
            Add ten uniquely identified nodes with a fixed consciousness level to the matrix from a single thread.
            
            Each node's ID incorporates the thread ID and an index to ensure uniqueness. The outcome of each addition is appended to the shared `results` list as `True` for success or `False` for failure.
            """
            for i in range(10):
                node = MatrixNode(id=f"thread_{thread_id}_node_{i}", consciousness_level=0.5)
                try:
                    self.matrix.add_node(node)
                    results.append(True)
                except Exception:
                    results.append(False)
                time.sleep(0.001)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_nodes_thread, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Verify all operations succeeded
        self.assertTrue(all(results))


class TestConsciousnessState(unittest.TestCase):
    """Test cases for ConsciousnessState enumeration and transitions."""
    
    def test_consciousness_state_values(self):
        """Test consciousness state enumeration values."""
        self.assertEqual(ConsciousnessState.DORMANT.value, 0)
        self.assertEqual(ConsciousnessState.ACTIVE.value, 1)
        self.assertEqual(ConsciousnessState.AWARE.value, 2)
        self.assertEqual(ConsciousnessState.TRANSCENDENT.value, 3)
        
    def test_consciousness_state_ordering(self):
        """
        Tests that the ordering of ConsciousnessState enumeration values reflects their intended progression.
        """
        self.assertLess(ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE)
        self.assertLess(ConsciousnessState.ACTIVE, ConsciousnessState.AWARE)
        self.assertLess(ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT)
        
    def test_consciousness_state_string_representation(self):
        """
        Verifies that each ConsciousnessState enum value returns the correct string representation.
        """
        self.assertEqual(str(ConsciousnessState.DORMANT), "DORMANT")
        self.assertEqual(str(ConsciousnessState.ACTIVE), "ACTIVE")
        self.assertEqual(str(ConsciousnessState.AWARE), "AWARE")
        self.assertEqual(str(ConsciousnessState.TRANSCENDENT), "TRANSCENDENT")


class TestMatrixNode(unittest.TestCase):
    """Test cases for MatrixNode class."""
    
    def setUp(self):
        """
        Set up a MatrixNode instance with a test ID and consciousness level before each test.
        """
        self.node = MatrixNode(id="test_node", consciousness_level=0.5)
        
    def test_node_initialization(self):
        """
        Test that a MatrixNode is initialized with the correct ID and consciousness level.
        """
        node = MatrixNode(id="init_test", consciousness_level=0.7)
        self.assertEqual(node.id, "init_test")
        self.assertEqual(node.consciousness_level, 0.7)
        
    def test_node_initialization_invalid_consciousness_level(self):
        """
        Test that creating a MatrixNode with a consciousness level outside the allowed range raises a ValueError.
        """
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=1.5)
            
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=-0.1)
            
    def test_node_consciousness_level_update(self):
        """
        Tests updating a MatrixNode's consciousness level and verifies the new value is set correctly.
        """
        self.node.update_consciousness_level(0.8)
        self.assertEqual(self.node.consciousness_level, 0.8)
        
    def test_node_consciousness_level_update_invalid(self):
        """
        Test that updating the node's consciousness level to an invalid value raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.node.update_consciousness_level(1.2)
            
    def test_node_equality(self):
        """
        Verify that MatrixNode instances with identical IDs and consciousness levels are equal, and those with different IDs are not.
        """
        node1 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node2 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node3 = MatrixNode(id="different_test", consciousness_level=0.5)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        
    def test_node_hash(self):
        """
        Test that MatrixNode instances with the same ID have identical hash values, ensuring consistent behavior in hash-based collections.
        """
        node1 = MatrixNode(id="hash_test", consciousness_level=0.5)
        node2 = MatrixNode(id="hash_test", consciousness_level=0.7)
        
        # Nodes with same ID should have same hash
        self.assertEqual(hash(node1), hash(node2))
        
    def test_node_string_representation(self):
        """
        Tests that the string representation of a MatrixNode includes its ID and consciousness level.
        """
        node_str = str(self.node)
        self.assertIn("test_node", node_str)
        self.assertIn("0.5", node_str)


class TestMatrixExceptions(unittest.TestCase):
    """Test cases for custom matrix exceptions."""
    
    def test_matrix_exception_inheritance(self):
        """
        Test that custom matrix exceptions inherit from the correct base exception classes.
        """
        self.assertTrue(issubclass(MatrixException, Exception))
        self.assertTrue(issubclass(InvalidStateException, MatrixException))
        self.assertTrue(issubclass(MatrixInitializationError, MatrixException))
        
    def test_matrix_exception_messages(self):
        """
        Tests that custom matrix exceptions propagate correctly and display the expected error messages.
        """
        try:
            raise MatrixException("Test matrix error")
        except MatrixException as e:
            self.assertEqual(str(e), "Test matrix error")
            
        try:
            raise InvalidStateException("Test invalid state")
        except InvalidStateException as e:
            self.assertEqual(str(e), "Test invalid state")


class TestMatrixIntegration(unittest.TestCase):
    """Integration tests for matrix components working together."""
    
    def setUp(self):
        """
        Set up a new GenesisConsciousnessMatrix instance before each integration test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_consciousness_evolution_cycle(self):
        """
        Simulates a complete consciousness evolution cycle by adding and connecting nodes, evolving the matrix until convergence, and verifying that the overall consciousness level changes.
        """
        # Initialize matrix with nodes
        for i in range(10):
            node = MatrixNode(id=f"evolution_node_{i}", consciousness_level=0.1 + i * 0.08)
            self.matrix.add_node(node)
            
        # Establish connections
        for i in range(9):
            self.matrix.connect_nodes(f"evolution_node_{i}", f"evolution_node_{i+1}", strength=0.6)
            
        # Evolve matrix
        initial_consciousness = self.matrix.calculate_consciousness_level()
        self.matrix.evolve_until_convergence(max_iterations=50)
        final_consciousness = self.matrix.calculate_consciousness_level()
        
        # Verify evolution occurred
        self.assertNotEqual(initial_consciousness, final_consciousness)
        
    def test_consciousness_emergence_full_cycle(self):
        """
        Tests that consciousness emergence is not detected with low node consciousness levels, but is detected after increasing all node levels above the emergence threshold.
        """
        # Start with low consciousness
        for i in range(5):
            node = MatrixNode(id=f"emergence_node_{i}", consciousness_level=0.2)
            self.matrix.add_node(node)
            
        self.assertFalse(self.matrix.detect_consciousness_emergence())
        
        # Gradually increase consciousness
        for node_id in self.matrix.nodes:
            node = self.matrix.nodes[node_id]
            node.update_consciousness_level(0.9)
            
        self.assertTrue(self.matrix.detect_consciousness_emergence())
        
    def test_matrix_persistence_integrity(self):
        """
        Tests that serializing and deserializing the matrix preserves all node data and node-to-node connections, ensuring the integrity of persisted state.
        """
        # Create complex matrix state
        nodes_data = []
        for i in range(20):
            node = MatrixNode(id=f"persist_node_{i}", consciousness_level=0.3 + i * 0.03)
            self.matrix.add_node(node)
            nodes_data.append((node.id, node.consciousness_level))
            
        # Add connections
        for i in range(19):
            self.matrix.connect_nodes(f"persist_node_{i}", f"persist_node_{i+1}", strength=0.7)
            
        # Serialize and deserialize
        serialized = self.matrix.to_json()
        restored_matrix = GenesisConsciousnessMatrix.from_json(serialized)
        
        # Verify all data preserved
        for node_id, consciousness_level in nodes_data:
            self.assertIn(node_id, restored_matrix.nodes)
            self.assertEqual(restored_matrix.nodes[node_id].consciousness_level, consciousness_level)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)

class TestGenesisConsciousnessMatrixAdvanced(unittest.TestCase):
    """Advanced test cases for edge cases and complex scenarios."""
    
    def setUp(self):
        """Set up test fixtures for advanced testing scenarios."""
        self.matrix = GenesisConsciousnessMatrix()
        self.large_config = {
            'dimension': 1024,
            'consciousness_threshold': 0.95,
            'learning_rate': 0.0001,
            'max_iterations': 10000,
            'convergence_tolerance': 1e-6
        }
        
    def test_matrix_initialization_edge_case_configs(self):
        """Test matrix initialization with edge case configurations."""
        # Test with minimum valid values
        min_config = {
            'dimension': 1,
            'consciousness_threshold': 0.0,
            'learning_rate': 1e-10,
            'max_iterations': 1
        }
        matrix = GenesisConsciousnessMatrix(config=min_config)
        self.assertEqual(matrix.dimension, 1)
        self.assertEqual(matrix.consciousness_threshold, 0.0)
        
        # Test with maximum valid values
        max_config = {
            'dimension': 2048,
            'consciousness_threshold': 1.0,
            'learning_rate': 1.0,
            'max_iterations': 1000000
        }
        matrix = GenesisConsciousnessMatrix(config=max_config)
        self.assertEqual(matrix.dimension, 2048)
        self.assertEqual(matrix.consciousness_threshold, 1.0)
        
    def test_matrix_initialization_boundary_conditions(self):
        """Test matrix initialization with boundary condition values."""
        boundary_configs = [
            {'dimension': 0, 'consciousness_threshold': 0.5},  # Zero dimension
            {'dimension': 256, 'consciousness_threshold': -0.1},  # Negative threshold
            {'dimension': 256, 'consciousness_threshold': 1.1},  # Threshold > 1
            {'dimension': -1, 'consciousness_threshold': 0.5},  # Negative dimension
            {'learning_rate': 0.0},  # Zero learning rate
            {'learning_rate': -0.1},  # Negative learning rate
            {'max_iterations': 0},  # Zero iterations
            {'max_iterations': -1},  # Negative iterations
        ]
        
        for config in boundary_configs:
            with self.assertRaises(MatrixInitializationError):
                GenesisConsciousnessMatrix(config=config)
                
    def test_matrix_node_management_stress_test(self):
        """Stress test for adding and removing large numbers of nodes."""
        # Add 1000 nodes
        node_ids = []
        for i in range(1000):
            node_id = f"stress_node_{i}"
            node = MatrixNode(id=node_id, consciousness_level=0.5)
            self.matrix.add_node(node)
            node_ids.append(node_id)
            
        self.assertEqual(len(self.matrix.nodes), 1000)
        
        # Remove every other node
        for i in range(0, 1000, 2):
            self.matrix.remove_node(f"stress_node_{i}")
            
        self.assertEqual(len(self.matrix.nodes), 500)
        
    def test_matrix_consciousness_level_extreme_values(self):
        """Test consciousness level calculations with extreme values."""
        # Test with all nodes at minimum consciousness
        for i in range(10):
            node = MatrixNode(id=f"min_node_{i}", consciousness_level=0.0)
            self.matrix.add_node(node)
            
        self.assertEqual(self.matrix.calculate_consciousness_level(), 0.0)
        
        # Test with all nodes at maximum consciousness
        self.matrix.reset()
        for i in range(10):
            node = MatrixNode(id=f"max_node_{i}", consciousness_level=1.0)
            self.matrix.add_node(node)
            
        self.assertEqual(self.matrix.calculate_consciousness_level(), 1.0)
        
    def test_matrix_state_transition_complex_sequences(self):
        """Test complex state transition sequences."""
        # Test valid transition sequence
        transitions = [
            (ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE),
            (ConsciousnessState.ACTIVE, ConsciousnessState.AWARE),
            (ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT),
            (ConsciousnessState.TRANSCENDENT, ConsciousnessState.AWARE),
            (ConsciousnessState.AWARE, ConsciousnessState.ACTIVE),
            (ConsciousnessState.ACTIVE, ConsciousnessState.DORMANT)
        ]
        
        for from_state, to_state in transitions:
            self.matrix.current_state = from_state
            result = self.matrix.transition_state(from_state, to_state)
            self.assertTrue(result)
            self.assertEqual(self.matrix.current_state, to_state)
            
    def test_matrix_evolution_convergence_edge_cases(self):
        """Test matrix evolution convergence with edge cases."""
        # Test convergence with single node
        node = MatrixNode(id="single_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        self.matrix.evolve_until_convergence(max_iterations=5)
        self.assertTrue(self.matrix.has_converged())
        
        # Test convergence with identical nodes
        self.matrix.reset()
        for i in range(5):
            node = MatrixNode(id=f"identical_node_{i}", consciousness_level=0.7)
            self.matrix.add_node(node)
            
        self.matrix.evolve_until_convergence(max_iterations=10)
        self.assertTrue(self.matrix.has_converged())
        
    def test_matrix_serialization_edge_cases(self):
        """Test matrix serialization with edge cases and special characters."""
        # Test with special characters in node IDs
        special_nodes = [
            ("node_with_unicode_🧠", 0.5),
            ("node with spaces", 0.6),
            ("node-with-dashes", 0.7),
            ("node_with_numbers_123", 0.8),
            ("node.with.dots", 0.9)
        ]
        
        for node_id, consciousness_level in special_nodes:
            node = MatrixNode(id=node_id, consciousness_level=consciousness_level)
            self.matrix.add_node(node)
            
        # Test serialization and deserialization
        serialized = self.matrix.to_json()
        restored_matrix = GenesisConsciousnessMatrix.from_json(serialized)
        
        for node_id, consciousness_level in special_nodes:
            self.assertIn(node_id, restored_matrix.nodes)
            self.assertEqual(restored_matrix.nodes[node_id].consciousness_level, consciousness_level)
            
    def test_matrix_error_handling_comprehensive(self):
        """Comprehensive error handling tests for various failure scenarios."""
        # Test file operations with invalid paths
        invalid_paths = [
            "/nonexistent/directory/file.json",
            "",
            "/dev/null/invalid",
            "file_with_no_extension",
            "file.with.invalid.extension.xyz"
        ]
        
        for path in invalid_paths:
            with self.assertRaises((MatrixException, OSError, IOError)):
                self.matrix.save_to_file(path)
                
        # Test loading from invalid JSON files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("invalid json content")
            temp_file = f.name
            
        try:
            with self.assertRaises(MatrixException):
                GenesisConsciousnessMatrix.load_from_file(temp_file)
        finally:
            os.unlink(temp_file)
            
    def test_matrix_concurrent_operations(self):
        """Test matrix operations under concurrent access patterns."""
        import threading
        import time
        import random
        
        operations_completed = []
        errors_encountered = []
        
        def concurrent_operations(thread_id):
            """Perform various operations concurrently."""
            try:
                # Add nodes
                for i in range(5):
                    node = MatrixNode(id=f"concurrent_{thread_id}_{i}", consciousness_level=random.uniform(0.1, 0.9))
                    self.matrix.add_node(node)
                    operations_completed.append(f"add_{thread_id}_{i}")
                    
                # Update consciousness levels
                for node_id in list(self.matrix.nodes.keys())[:3]:
                    node = self.matrix.nodes[node_id]
                    node.update_consciousness_level(random.uniform(0.1, 0.9))
                    operations_completed.append(f"update_{thread_id}_{node_id}")
                    
                # Calculate metrics
                metrics = self.matrix.calculate_metrics()
                operations_completed.append(f"metrics_{thread_id}")
                
                # Evolve matrix
                self.matrix.evolve_step()
                operations_completed.append(f"evolve_{thread_id}")
                
            except Exception as e:
                errors_encountered.append(f"Error in thread {thread_id}: {str(e)}")
                
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_operations, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Verify operations completed without critical errors
        self.assertGreater(len(operations_completed), 0)
        # Some errors might be acceptable due to race conditions
        self.assertLess(len(errors_encountered), len(operations_completed))
        
    def test_matrix_performance_benchmarks(self):
        """Performance benchmark tests for various matrix operations."""
        import time
        
        # Benchmark node addition
        start_time = time.time()
        for i in range(500):
            node = MatrixNode(id=f"benchmark_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
        add_time = time.time() - start_time
        
        # Should be able to add 500 nodes in less than 1 second
        self.assertLess(add_time, 1.0)
        
        # Benchmark consciousness calculation
        start_time = time.time()
        for _ in range(100):
            self.matrix.calculate_consciousness_level()
        calc_time = time.time() - start_time
        
        # Should be able to calculate 100 times in less than 0.1 seconds
        self.assertLess(calc_time, 0.1)
        
        # Benchmark evolution steps
        start_time = time.time()
        for _ in range(10):
            self.matrix.evolve_step()
        evolve_time = time.time() - start_time
        
        # Should be able to evolve 10 times in less than 5 seconds
        self.assertLess(evolve_time, 5.0)
        
    def test_matrix_memory_efficiency(self):
        """Test memory efficiency during large-scale operations."""
        import sys
        
        # Get initial memory usage
        initial_memory = sys.getsizeof(self.matrix)
        
        # Add many nodes
        for i in range(1000):
            node = MatrixNode(id=f"memory_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Check memory growth is reasonable
        current_memory = sys.getsizeof(self.matrix)
        memory_growth = current_memory - initial_memory
        
        # Memory growth should be reasonable (less than 1MB for 1000 nodes)
        self.assertLess(memory_growth, 1024 * 1024)
        
        # Clean up half the nodes
        for i in range(0, 1000, 2):
            self.matrix.remove_node(f"memory_node_{i}")
            
        # Verify node count is correct
        self.assertEqual(len(self.matrix.nodes), 500)


class TestMatrixNodeAdvanced(unittest.TestCase):
    """Advanced test cases for MatrixNode functionality."""
    
    def test_node_consciousness_level_precision(self):
        """Test consciousness level handling with high precision values."""
        precision_levels = [
            0.0000001,
            0.1234567,
            0.9999999,
            0.5000000,
            0.7777777
        ]
        
        for level in precision_levels:
            node = MatrixNode(id=f"precision_node_{level}", consciousness_level=level)
            self.assertAlmostEqual(node.consciousness_level, level, places=7)
            
    def test_node_id_validation(self):
        """Test node ID validation with various edge cases."""
        invalid_ids = [
            "",           # Empty string
            None,         # None value
            123,          # Non-string type
            [],           # List type
            {},           # Dict type
        ]
        
        for invalid_id in invalid_ids:
            with self.assertRaises((ValueError, TypeError)):
                MatrixNode(id=invalid_id, consciousness_level=0.5)
                
    def test_node_consciousness_level_boundary_values(self):
        """Test consciousness level with exact boundary values."""
        boundary_values = [
            (0.0, True),    # Minimum valid value
            (1.0, True),    # Maximum valid value
            (-0.0000001, False),  # Just below minimum
            (1.0000001, False),   # Just above maximum
            (float('inf'), False),  # Infinity
            (float('-inf'), False), # Negative infinity
            (float('nan'), False),  # NaN
        ]
        
        for value, should_be_valid in boundary_values:
            if should_be_valid:
                node = MatrixNode(id=f"boundary_node_{value}", consciousness_level=value)
                self.assertEqual(node.consciousness_level, value)
            else:
                with self.assertRaises((ValueError, TypeError)):
                    MatrixNode(id=f"boundary_node_{value}", consciousness_level=value)
                    
    def test_node_update_consciousness_level_edge_cases(self):
        """Test updating consciousness level with edge cases."""
        node = MatrixNode(id="update_test", consciousness_level=0.5)
        
        # Test rapid successive updates
        for i in range(100):
            new_level = i / 100.0
            node.update_consciousness_level(new_level)
            self.assertEqual(node.consciousness_level, new_level)
            
        # Test updating to the same value
        node.update_consciousness_level(0.5)
        node.update_consciousness_level(0.5)
        self.assertEqual(node.consciousness_level, 0.5)
        
    def test_node_comparison_edge_cases(self):
        """Test node comparison with various edge cases."""
        node1 = MatrixNode(id="compare_test", consciousness_level=0.5)
        node2 = MatrixNode(id="compare_test", consciousness_level=0.7)
        node3 = MatrixNode(id="different_test", consciousness_level=0.5)
        
        # Test equality with different consciousness levels but same ID
        self.assertEqual(node1, node2)
        
        # Test inequality with different IDs
        self.assertNotEqual(node1, node3)
        
        # Test comparison with non-MatrixNode objects
        self.assertNotEqual(node1, "not_a_node")
        self.assertNotEqual(node1, 123)
        self.assertNotEqual(node1, None)
        
    def test_node_serialization_edge_cases(self):
        """Test node serialization with edge cases."""
        # Test with extreme consciousness levels
        extreme_nodes = [
            MatrixNode(id="min_node", consciousness_level=0.0),
            MatrixNode(id="max_node", consciousness_level=1.0),
            MatrixNode(id="precise_node", consciousness_level=0.123456789)
        ]
        
        for node in extreme_nodes:
            # Assuming nodes have a to_dict method
            if hasattr(node, 'to_dict'):
                node_dict = node.to_dict()
                self.assertIn('id', node_dict)
                self.assertIn('consciousness_level', node_dict)
                self.assertEqual(node_dict['id'], node.id)
                self.assertEqual(node_dict['consciousness_level'], node.consciousness_level)


class TestMatrixExceptionsAdvanced(unittest.TestCase):
    """Advanced test cases for custom matrix exceptions."""
    
    def test_exception_chaining(self):
        """Test exception chaining and cause tracking."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as original_error:
                raise MatrixException("Matrix error occurred") from original_error
        except MatrixException as e:
            self.assertEqual(str(e), "Matrix error occurred")
            self.assertIsInstance(e.__cause__, ValueError)
            self.assertEqual(str(e.__cause__), "Original error")
            
    def test_exception_context_preservation(self):
        """Test that exception context is preserved in custom exceptions."""
        try:
            matrix = GenesisConsciousnessMatrix()
            # Simulate a scenario that might cause an exception
            matrix.add_node(None)  # This should raise an exception
        except Exception as e:
            # The exception should contain useful context
            self.assertIsInstance(e, (MatrixException, ValueError, TypeError))
            
    def test_exception_hierarchy_validation(self):
        """Test the complete exception hierarchy."""
        # Test that all custom exceptions are properly structured
        exception_classes = [
            MatrixException,
            InvalidStateException,
            MatrixInitializationError
        ]
        
        for exc_class in exception_classes:
            # Test instantiation
            exc = exc_class("Test message")
            self.assertIsInstance(exc, Exception)
            self.assertEqual(str(exc), "Test message")
            
            # Test raising and catching
            with self.assertRaises(exc_class):
                raise exc_class("Test error")


class TestMatrixIntegrationAdvanced(unittest.TestCase):
    """Advanced integration tests for complex matrix scenarios."""
    
    def setUp(self):
        """Set up complex test scenario."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_large_scale_matrix_operations(self):
        """Test matrix operations at scale."""
        # Create a large matrix with 1000 nodes
        node_count = 1000
        for i in range(node_count):
            node = MatrixNode(id=f"scale_node_{i}", consciousness_level=0.1 + (i % 10) * 0.08)
            self.matrix.add_node(node)
            
        # Create a complex connection pattern
        for i in range(node_count - 1):
            # Connect each node to the next one
            self.matrix.connect_nodes(f"scale_node_{i}", f"scale_node_{i+1}", strength=0.5)
            
            # Create some random connections
            if i % 10 == 0 and i + 10 < node_count:
                self.matrix.connect_nodes(f"scale_node_{i}", f"scale_node_{i+10}", strength=0.3)
                
        # Perform complex operations
        initial_metrics = self.matrix.calculate_metrics()
        self.matrix.evolve_step()
        final_metrics = self.matrix.calculate_metrics()
        
        # Verify the operations completed successfully
        self.assertIsInstance(initial_metrics, dict)
        self.assertIsInstance(final_metrics, dict)
        self.assertEqual(len(self.matrix.nodes), node_count)
        
    def test_matrix_resilience_under_stress(self):
        """Test matrix resilience under various stress conditions."""
        import random
        
        # Add nodes with random properties
        for i in range(100):
            consciousness_level = random.uniform(0.0, 1.0)
            node = MatrixNode(id=f"stress_node_{i}", consciousness_level=consciousness_level)
            self.matrix.add_node(node)
            
        # Perform random operations
        for _ in range(500):
            operation = random.choice(['add', 'remove', 'update', 'connect', 'evolve'])
            
            try:
                if operation == 'add':
                    node_id = f"random_node_{random.randint(1000, 9999)}"
                    if node_id not in self.matrix.nodes:
                        node = MatrixNode(id=node_id, consciousness_level=random.uniform(0.0, 1.0))
                        self.matrix.add_node(node)
                        
                elif operation == 'remove':
                    if self.matrix.nodes:
                        node_id = random.choice(list(self.matrix.nodes.keys()))
                        self.matrix.remove_node(node_id)
                        
                elif operation == 'update':
                    if self.matrix.nodes:
                        node_id = random.choice(list(self.matrix.nodes.keys()))
                        node = self.matrix.nodes[node_id]
                        node.update_consciousness_level(random.uniform(0.0, 1.0))
                        
                elif operation == 'connect':
                    if len(self.matrix.nodes) >= 2:
                        node_ids = random.sample(list(self.matrix.nodes.keys()), 2)
                        self.matrix.connect_nodes(node_ids[0], node_ids[1], strength=random.uniform(0.1, 1.0))
                        
                elif operation == 'evolve':
                    self.matrix.evolve_step()
                    
            except Exception:
                # Some operations might fail due to random conditions, which is acceptable
                pass
                
        # Verify matrix is still in a valid state
        self.assertIsInstance(self.matrix.nodes, dict)
        self.assertIsInstance(self.matrix.calculate_consciousness_level(), float)
        
    def test_matrix_recovery_from_corruption(self):
        """Test matrix recovery capabilities from various corruption scenarios."""
        # Create a valid matrix state
        for i in range(10):
            node = MatrixNode(id=f"recovery_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Simulate various corruption scenarios and recovery
        original_node_count = len(self.matrix.nodes)
        
        # Test state recovery
        if hasattr(self.matrix, 'create_checkpoint'):
            self.matrix.create_checkpoint()
            
        # Simulate corruption by removing all nodes
        for node_id in list(self.matrix.nodes.keys()):
            self.matrix.remove_node(node_id)
            
        self.assertEqual(len(self.matrix.nodes), 0)
        
        # Test recovery mechanism
        if hasattr(self.matrix, 'restore_checkpoint'):
            self.matrix.restore_checkpoint()
            self.assertEqual(len(self.matrix.nodes), original_node_count)
        else:
            # Manual recovery by re-adding nodes
            for i in range(original_node_count):
                node = MatrixNode(id=f"recovery_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            self.assertEqual(len(self.matrix.nodes), original_node_count)


# Add async tests if the matrix supports asynchronous operations
class TestMatrixAsyncOperations(unittest.TestCase):
    """Test asynchronous operations if supported by the matrix."""
    
    def setUp(self):
        """Set up async test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_async_matrix_operations(self):
        """Test asynchronous matrix operations."""
        async def async_operations():
            # Add nodes asynchronously
            for i in range(10):
                node = MatrixNode(id=f"async_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Perform async evolution if supported
            if hasattr(self.matrix, 'evolve_step_async'):
                await self.matrix.evolve_step_async()
                
            # Calculate metrics
            metrics = self.matrix.calculate_metrics()
            self.assertIsInstance(metrics, dict)
            
        # Run the async test
        if hasattr(asyncio, 'run'):
            asyncio.run(async_operations())
        else:
            # Fallback for older Python versions
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_operations())


# Performance and profiling tests
class TestMatrixProfiling(unittest.TestCase):
    """Profiling and performance analysis tests."""
    
    def test_matrix_memory_profiling(self):
        """Profile memory usage during matrix operations."""
        import gc
        
        matrix = GenesisConsciousnessMatrix()
        
        # Force garbage collection
        gc.collect()
        
        # Add many nodes and measure memory
        for i in range(1000):
            node = MatrixNode(id=f"profile_node_{i}", consciousness_level=0.5)
            matrix.add_node(node)
            
        # Force garbage collection again
        gc.collect()
        
        # Remove all nodes
        for node_id in list(matrix.nodes.keys()):
            matrix.remove_node(node_id)
            
        # Force garbage collection
        gc.collect()
        
        # Verify cleanup
        self.assertEqual(len(matrix.nodes), 0)
        
    def test_matrix_cpu_profiling(self):
        """Profile CPU usage during intensive matrix operations."""
        import time
        
        matrix = GenesisConsciousnessMatrix()
        
        # Add nodes
        for i in range(100):
            node = MatrixNode(id=f"cpu_node_{i}", consciousness_level=0.5)
            matrix.add_node(node)
            
        # Measure CPU-intensive operations
        start_time = time.time()
        
        # Perform many evolution steps
        for _ in range(50):
            matrix.evolve_step()
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify performance is reasonable
        self.assertLess(execution_time, 10.0)  # Should complete in less than 10 seconds
        
        # Measure consciousness calculations
        start_time = time.time()
        
        for _ in range(1000):
            matrix.calculate_consciousness_level()
            
        end_time = time.time()
        calc_time = end_time - start_time
        
        # Should be very fast
        self.assertLess(calc_time, 1.0)


if __name__ == '__main__':
    # Run all tests including the new advanced ones
    unittest.main(verbosity=2, buffer=True)