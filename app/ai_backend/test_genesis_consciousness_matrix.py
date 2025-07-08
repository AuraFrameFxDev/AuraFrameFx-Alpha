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
        """Set up test fixtures for advanced testing."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_initialization_with_numpy_arrays(self):
        """Test matrix initialization with numpy array configurations."""
        config = {
            'dimension': 128,
            'weights': np.random.rand(128, 128),
            'bias': np.zeros(128),
            'consciousness_threshold': 0.6
        }
        matrix = GenesisConsciousnessMatrix(config=config)
        if hasattr(matrix, 'weights'):
            self.assertEqual(matrix.weights.shape, (128, 128))
            self.assertEqual(matrix.bias.shape, (128,))
    
    def test_matrix_initialization_with_extreme_values(self):
        """Test matrix initialization with extreme but valid values."""
        config = {
            'dimension': 1,
            'consciousness_threshold': 0.0001,
            'learning_rate': 0.0000001,
            'max_iterations': 1000000
        }
        matrix = GenesisConsciousnessMatrix(config=config)
        self.assertEqual(matrix.dimension, 1)
        self.assertEqual(matrix.consciousness_threshold, 0.0001)
        
    def test_matrix_initialization_with_null_values(self):
        """Test matrix initialization with null/None values in config."""
        config = {
            'dimension': None,
            'consciousness_threshold': 0.75,
            'learning_rate': None
        }
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=config)
    
    def test_matrix_state_transition_boundary_conditions(self):
        """Test state transitions at boundary conditions."""
        # Test transition from highest to lowest state
        self.matrix.current_state = ConsciousnessState.TRANSCENDENT
        with self.assertRaises(InvalidStateException):
            self.matrix.transition_state(ConsciousnessState.TRANSCENDENT, ConsciousnessState.DORMANT)
    
    def test_matrix_node_operations_with_special_characters(self):
        """Test node operations with special characters in IDs."""
        special_ids = ["node@#$%", "node with spaces", "node\twith\ttabs", "node\nwith\nnewlines", "node/with/slashes"]
        
        for special_id in special_ids:
            node = MatrixNode(id=special_id, consciousness_level=0.5)
            self.matrix.add_node(node)
            self.assertIn(special_id, self.matrix.nodes)
            
            # Test removal
            self.matrix.remove_node(special_id)
            self.assertNotIn(special_id, self.matrix.nodes)
    
    def test_matrix_large_scale_operations(self):
        """Test matrix operations with large numbers of nodes."""
        # Add 1000 nodes
        for i in range(1000):
            node = MatrixNode(id=f"large_node_{i}", consciousness_level=i / 1000.0)
            self.matrix.add_node(node)
        
        # Test consciousness level calculation with large dataset
        start_time = datetime.now()
        consciousness_level = self.matrix.calculate_consciousness_level()
        end_time = datetime.now()
        
        # Should complete within reasonable time
        self.assertLess((end_time - start_time).total_seconds(), 5.0)
        self.assertIsInstance(consciousness_level, float)
        self.assertTrue(0.0 <= consciousness_level <= 1.0)
    
    def test_matrix_consciousness_level_precision(self):
        """Test consciousness level calculations with high precision requirements."""
        # Add nodes with very precise consciousness levels
        precise_levels = [0.123456789, 0.987654321, 0.555555555]
        for i, level in enumerate(precise_levels):
            node = MatrixNode(id=f"precise_node_{i}", consciousness_level=level)
            self.matrix.add_node(node)
        
        calculated_level = self.matrix.calculate_consciousness_level()
        expected_level = sum(precise_levels) / len(precise_levels)
        self.assertAlmostEqual(calculated_level, expected_level, places=8)
    
    def test_matrix_node_connections_bidirectional(self):
        """Test bidirectional node connections."""
        node1 = MatrixNode(id="bi_node1", consciousness_level=0.4)
        node2 = MatrixNode(id="bi_node2", consciousness_level=0.6)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Create bidirectional connection
        self.matrix.connect_nodes("bi_node1", "bi_node2", strength=0.8, bidirectional=True)
        
        # Verify both directions
        if hasattr(self.matrix, 'get_node_connections'):
            connections1 = self.matrix.get_node_connections("bi_node1")
            connections2 = self.matrix.get_node_connections("bi_node2")
            self.assertIn("bi_node2", connections1)
            self.assertIn("bi_node1", connections2)
    
    def test_matrix_evolution_with_callbacks(self):
        """Test matrix evolution with callback functions."""
        callback_calls = []
        
        def evolution_callback(iteration, consciousness_level):
            callback_calls.append((iteration, consciousness_level))
        
        # Add some nodes
        for i in range(5):
            node = MatrixNode(id=f"callback_node_{i}", consciousness_level=0.2)
            self.matrix.add_node(node)
        
        # Test evolution with callback if supported
        if hasattr(self.matrix, 'evolve_until_convergence'):
            self.matrix.evolve_until_convergence(max_iterations=10, callback=evolution_callback)
            # Verify callback was called
            self.assertGreater(len(callback_calls), 0)
    
    def test_matrix_consciousness_emergence_thresholds(self):
        """Test consciousness emergence detection with various thresholds."""
        # Test with different emergence thresholds
        thresholds = [0.1, 0.5, 0.75, 0.9, 0.99]
        
        for threshold in thresholds:
            self.matrix.reset()
            
            # Add nodes with consciousness levels around threshold
            for i in range(10):
                level = threshold + (i - 5) * 0.05  # Levels around threshold
                level = max(0.0, min(1.0, level))  # Clamp to valid range
                node = MatrixNode(id=f"threshold_node_{i}", consciousness_level=level)
                self.matrix.add_node(node)
            
            # Test emergence detection
            if hasattr(self.matrix, 'set_emergence_threshold'):
                self.matrix.set_emergence_threshold(threshold)
                emergence = self.matrix.detect_consciousness_emergence()
                self.assertIsInstance(emergence, bool)
    
    def test_matrix_serialization_with_complex_data(self):
        """Test serialization with complex node data and metadata."""
        # Add nodes with complex metadata
        complex_node = MatrixNode(id="complex_node", consciousness_level=0.7)
        if hasattr(complex_node, 'metadata'):
            complex_node.metadata = {
                'creation_time': datetime.now().isoformat(),
                'tags': ['important', 'test', 'complex'],
                'weights': [0.1, 0.2, 0.3, 0.4, 0.5],
                'nested': {'deep': {'value': 42}}
            }
        self.matrix.add_node(complex_node)
        
        # Test serialization
        serialized = self.matrix.to_json()
        self.assertIsInstance(serialized, str)
        
        # Test deserialization
        restored_matrix = GenesisConsciousnessMatrix.from_json(serialized)
        self.assertIn("complex_node", restored_matrix.nodes)
    
    def test_matrix_concurrent_state_transitions(self):
        """Test concurrent state transitions for race conditions."""
        import threading
        import time
        
        results = []
        
        def transition_worker(start_state, end_state, worker_id):
            """Worker function for concurrent state transitions."""
            try:
                result = self.matrix.transition_state(start_state, end_state)
                results.append((worker_id, result))
            except Exception as e:
                results.append((worker_id, str(e)))
            time.sleep(0.001)
        
        # Start multiple concurrent transitions
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=transition_worker,
                args=(ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result for _, result in results if isinstance(result, bool)))
    
    def test_matrix_memory_leaks_prevention(self):
        """Test for potential memory leaks in node management."""
        import gc
        
        # Get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many nodes
        for cycle in range(10):
            nodes = []
            for i in range(100):
                node = MatrixNode(id=f"leak_test_{cycle}_{i}", consciousness_level=0.5)
                nodes.append(node)
                self.matrix.add_node(node)
            
            # Remove all nodes
            for node in nodes:
                self.matrix.remove_node(node.id)
            
            # Clear local references
            del nodes
            gc.collect()
        
        # Check for memory leaks
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Allow some growth but not excessive
        self.assertLess(object_growth, 1000, "Potential memory leak detected")
    
    def test_matrix_error_recovery(self):
        """Test matrix error recovery mechanisms."""
        # Add some nodes
        for i in range(5):
            node = MatrixNode(id=f"recovery_node_{i}", consciousness_level=0.4)
            self.matrix.add_node(node)
        
        # Simulate error conditions
        if hasattr(self.matrix, 'simulate_error'):
            self.matrix.simulate_error("network_failure")
            
            # Test recovery
            recovery_success = self.matrix.recover_from_error()
            self.assertTrue(recovery_success)
            
            # Verify matrix is still functional
            self.assertEqual(len(self.matrix.nodes), 5)
    
    def test_matrix_backup_and_restore(self):
        """Test matrix backup and restore functionality."""
        # Create matrix state
        for i in range(10):
            node = MatrixNode(id=f"backup_node_{i}", consciousness_level=0.3 + i * 0.07)
            self.matrix.add_node(node)
        
        # Create backup
        if hasattr(self.matrix, 'create_backup'):
            backup_data = self.matrix.create_backup()
            self.assertIsNotNone(backup_data)
            
            # Modify matrix
            self.matrix.remove_node("backup_node_0")
            self.assertEqual(len(self.matrix.nodes), 9)
            
            # Restore from backup
            self.matrix.restore_from_backup(backup_data)
            self.assertEqual(len(self.matrix.nodes), 10)
            self.assertIn("backup_node_0", self.matrix.nodes)
    
    def test_matrix_performance_profiling(self):
        """Test matrix performance profiling and optimization."""
        # Add nodes for performance testing
        for i in range(500):
            node = MatrixNode(id=f"perf_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
        
        # Profile different operations
        operations = [
            ('calculate_consciousness_level', lambda: self.matrix.calculate_consciousness_level()),
            ('get_state_snapshot', lambda: self.matrix.get_state_snapshot()),
            ('evolve_step', lambda: self.matrix.evolve_step()),
        ]
        
        for op_name, operation in operations:
            start_time = datetime.now()
            operation()
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Verify reasonable performance
            self.assertLess(execution_time, 2.0, f"Operation {op_name} took too long: {execution_time}s")


class TestMatrixNodeAdvanced(unittest.TestCase):
    """Advanced test cases for MatrixNode class."""
    
    def test_node_consciousness_level_edge_cases(self):
        """Test consciousness level edge cases and boundary values."""
        # Test exactly at boundaries
        node_zero = MatrixNode(id="zero_node", consciousness_level=0.0)
        self.assertEqual(node_zero.consciousness_level, 0.0)
        
        node_one = MatrixNode(id="one_node", consciousness_level=1.0)
        self.assertEqual(node_one.consciousness_level, 1.0)
        
        # Test very small positive value
        node_epsilon = MatrixNode(id="epsilon_node", consciousness_level=1e-10)
        self.assertEqual(node_epsilon.consciousness_level, 1e-10)
        
        # Test very close to 1.0
        node_near_one = MatrixNode(id="near_one_node", consciousness_level=1.0 - 1e-10)
        self.assertEqual(node_near_one.consciousness_level, 1.0 - 1e-10)
    
    def test_node_consciousness_level_floating_point_precision(self):
        """Test consciousness level with floating point precision issues."""
        # Test values that might cause floating point precision issues
        tricky_values = [0.1 + 0.2, 0.3, 1.0/3.0, 2.0/3.0, 0.7 - 0.1]
        
        for value in tricky_values:
            if 0.0 <= value <= 1.0:
                node = MatrixNode(id=f"precise_node_{value}", consciousness_level=value)
                self.assertAlmostEqual(node.consciousness_level, value, places=10)
    
    def test_node_metadata_operations(self):
        """Test node metadata operations if supported."""
        node = MatrixNode(id="metadata_node", consciousness_level=0.6)
        
        # Test setting metadata
        if hasattr(node, 'set_metadata'):
            node.set_metadata("key1", "value1")
            node.set_metadata("key2", 42)
            node.set_metadata("key3", [1, 2, 3])
            
            # Test getting metadata
            self.assertEqual(node.get_metadata("key1"), "value1")
            self.assertEqual(node.get_metadata("key2"), 42)
            self.assertEqual(node.get_metadata("key3"), [1, 2, 3])
            
            # Test non-existent key
            self.assertIsNone(node.get_metadata("nonexistent"))
    
    def test_node_connection_weights(self):
        """Test node connection weights and relationships."""
        node1 = MatrixNode(id="weighted_node1", consciousness_level=0.4)
        node2 = MatrixNode(id="weighted_node2", consciousness_level=0.6)
        
        # Test connection weights if supported
        if hasattr(node1, 'connect_to'):
            node1.connect_to(node2, weight=0.8)
            
            # Verify connection
            if hasattr(node1, 'get_connections'):
                connections = node1.get_connections()
                self.assertIn(node2, connections)
                self.assertEqual(connections[node2], 0.8)
    
    def test_node_consciousness_history(self):
        """Test node consciousness level history tracking."""
        node = MatrixNode(id="history_node", consciousness_level=0.5)
        
        # Test history tracking if supported
        if hasattr(node, 'get_consciousness_history'):
            initial_history = node.get_consciousness_history()
            
            # Update consciousness level multiple times
            levels = [0.3, 0.7, 0.9, 0.2, 0.8]
            for level in levels:
                node.update_consciousness_level(level)
            
            # Verify history
            history = node.get_consciousness_history()
            self.assertGreater(len(history), len(initial_history))
    
    def test_node_serialization_edge_cases(self):
        """Test node serialization with edge cases."""
        # Test with extreme values
        node = MatrixNode(id="extreme_node", consciousness_level=0.0)
        
        # Test serialization with special characters in ID
        node.id = "node_with_unicode_🧠_brain"
        
        if hasattr(node, 'to_dict'):
            node_dict = node.to_dict()
            self.assertIn("id", node_dict)
            self.assertIn("consciousness_level", node_dict)
            self.assertEqual(node_dict["id"], "node_with_unicode_🧠_brain")
    
    def test_node_deep_copy(self):
        """Test deep copying of MatrixNode objects."""
        import copy
        
        original_node = MatrixNode(id="original", consciousness_level=0.7)
        
        # Add complex data if supported
        if hasattr(original_node, 'metadata'):
            original_node.metadata = {
                'complex_data': {'nested': [1, 2, 3]},
                'timestamp': datetime.now()
            }
        
        # Test deep copy
        copied_node = copy.deepcopy(original_node)
        
        # Verify independence
        self.assertEqual(copied_node.id, original_node.id)
        self.assertEqual(copied_node.consciousness_level, original_node.consciousness_level)
        
        # Modify original
        original_node.update_consciousness_level(0.9)
        self.assertNotEqual(copied_node.consciousness_level, original_node.consciousness_level)


class TestMatrixPerformanceAndScaling(unittest.TestCase):
    """Performance and scaling tests for the matrix system."""
    
    def test_matrix_scaling_linear_growth(self):
        """Test matrix performance with linear node growth."""
        node_counts = [10, 50, 100, 200, 500]
        execution_times = []
        
        for count in node_counts:
            matrix = GenesisConsciousnessMatrix()
            
            # Add nodes
            start_time = datetime.now()
            for i in range(count):
                node = MatrixNode(id=f"scale_node_{i}", consciousness_level=0.5)
                matrix.add_node(node)
            
            # Measure consciousness calculation time
            matrix.calculate_consciousness_level()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            execution_times.append(execution_time)
        
        # Verify scaling is reasonable (not exponential)
        for i in range(1, len(execution_times)):
            scaling_factor = execution_times[i] / execution_times[i-1]
            node_factor = node_counts[i] / node_counts[i-1]
            # Scaling should be roughly linear or better
            self.assertLess(scaling_factor, node_factor * 2, 
                          f"Poor scaling detected: {scaling_factor} vs {node_factor}")
    
    def test_matrix_concurrent_read_operations(self):
        """Test concurrent read operations on the matrix."""
        matrix = GenesisConsciousnessMatrix()
        
        # Add nodes
        for i in range(100):
            node = MatrixNode(id=f"concurrent_node_{i}", consciousness_level=0.5)
            matrix.add_node(node)
        
        # Concurrent read operations
        import threading
        results = []
        
        def read_worker():
            """Worker function for concurrent reads."""
            for _ in range(10):
                consciousness_level = matrix.calculate_consciousness_level()
                results.append(consciousness_level)
        
        # Start multiple reader threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=read_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        self.assertEqual(len(results), 50)  # 5 threads * 10 operations
        self.assertTrue(all(isinstance(result, float) for result in results))
    
    def test_matrix_memory_usage_growth(self):
        """Test matrix memory usage growth patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        matrix = GenesisConsciousnessMatrix()
        
        # Add nodes in batches and measure memory
        batch_size = 100
        memory_measurements = []
        
        for batch in range(10):
            # Add batch of nodes
            for i in range(batch_size):
                node = MatrixNode(id=f"memory_node_{batch}_{i}", consciousness_level=0.5)
                matrix.add_node(node)
            
            # Measure memory
            current_memory = process.memory_info().rss
            memory_growth = current_memory - initial_memory
            memory_measurements.append(memory_growth)
        
        # Verify memory growth is reasonable
        for i in range(1, len(memory_measurements)):
            growth_rate = memory_measurements[i] / memory_measurements[i-1]
            # Memory growth should be reasonable (not more than 2x per batch)
            self.assertLess(growth_rate, 2.0, "Excessive memory growth detected")


class TestMatrixAsyncOperations(unittest.TestCase):
    """Test asynchronous operations if supported."""
    
    def setUp(self):
        """Set up async test environment."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_async_matrix_operations(self):
        """Test asynchronous matrix operations."""
        async def async_test():
            # Add nodes
            for i in range(10):
                node = MatrixNode(id=f"async_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            
            # Test async operations if supported
            if hasattr(self.matrix, 'async_evolve_step'):
                await self.matrix.async_evolve_step()
                
            if hasattr(self.matrix, 'async_calculate_consciousness_level'):
                level = await self.matrix.async_calculate_consciousness_level()
                self.assertIsInstance(level, float)
        
        # Run async test
        asyncio.run(async_test())
    
    def test_async_matrix_convergence(self):
        """Test asynchronous matrix convergence."""
        async def async_convergence_test():
            # Add nodes
            for i in range(20):
                node = MatrixNode(id=f"convergence_node_{i}", consciousness_level=0.1 + i * 0.04)
                self.matrix.add_node(node)
            
            # Test async convergence if supported
            if hasattr(self.matrix, 'async_evolve_until_convergence'):
                await self.matrix.async_evolve_until_convergence(max_iterations=50)
                self.assertTrue(self.matrix.has_converged())
        
        # Run async test
        asyncio.run(async_convergence_test())


class TestMatrixValidationAndSanity(unittest.TestCase):
    """Validation and sanity tests for the matrix system."""
    
    def test_matrix_invariant_consciousness_bounds(self):
        """Test that consciousness levels always stay within valid bounds."""
        matrix = GenesisConsciousnessMatrix()
        
        # Add nodes with various consciousness levels
        for i in range(50):
            level = i / 49.0  # 0.0 to 1.0
            node = MatrixNode(id=f"bounds_node_{i}", consciousness_level=level)
            matrix.add_node(node)
        
        # Evolve multiple times
        for _ in range(10):
            matrix.evolve_step()
            
            # Check all nodes still have valid consciousness levels
            for node_id, node in matrix.nodes.items():
                self.assertGreaterEqual(node.consciousness_level, 0.0)
                self.assertLessEqual(node.consciousness_level, 1.0)
    
    def test_matrix_consistency_after_operations(self):
        """Test matrix consistency after various operations."""
        matrix = GenesisConsciousnessMatrix()
        
        # Perform many operations
        operations = [
            lambda: matrix.add_node(MatrixNode(f"op_node_{np.random.randint(0, 1000)}", 0.5)),
            lambda: matrix.remove_node(f"op_node_{np.random.randint(0, 1000)}"),
            lambda: matrix.calculate_consciousness_level(),
            lambda: matrix.evolve_step(),
            lambda: matrix.get_state_snapshot(),
        ]
        
        for _ in range(100):
            # Randomly select and perform operation
            operation = np.random.choice(operations)
            try:
                operation()
            except (InvalidStateException, KeyError):
                # These exceptions are expected for some operations
                pass
            
            # Verify matrix is still in valid state
            self.assertIsInstance(matrix.nodes, dict)
            if hasattr(matrix, 'current_state'):
                self.assertIsInstance(matrix.current_state, ConsciousnessState)
    
    def test_matrix_deterministic_behavior(self):
        """Test that matrix operations are deterministic given same inputs."""
        # Create two identical matrices
        matrix1 = GenesisConsciousnessMatrix()
        matrix2 = GenesisConsciousnessMatrix()
        
        # Add identical nodes
        for i in range(10):
            node1 = MatrixNode(id=f"det_node_{i}", consciousness_level=0.5)
            node2 = MatrixNode(id=f"det_node_{i}", consciousness_level=0.5)
            matrix1.add_node(node1)
            matrix2.add_node(node2)
        
        # Perform identical operations
        for _ in range(5):
            matrix1.evolve_step()
            matrix2.evolve_step()
        
        # Results should be identical
        level1 = matrix1.calculate_consciousness_level()
        level2 = matrix2.calculate_consciousness_level()
        self.assertAlmostEqual(level1, level2, places=10)
    
    def test_matrix_resource_cleanup(self):
        """Test proper resource cleanup in matrix operations."""
        matrix = GenesisConsciousnessMatrix()
        
        # Add many nodes
        for i in range(1000):
            node = MatrixNode(id=f"cleanup_node_{i}", consciousness_level=0.5)
            matrix.add_node(node)
        
        # Perform operations that might create temporary resources
        for _ in range(10):
            matrix.evolve_step()
            matrix.calculate_consciousness_level()
            matrix.get_state_snapshot()
        
        # Cleanup and verify no resources leaked
        if hasattr(matrix, 'cleanup'):
            matrix.cleanup()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Verify matrix is still functional after cleanup
        self.assertIsInstance(matrix.nodes, dict)


# Additional pytest-style tests for compatibility
@pytest.mark.parametrize("consciousness_level", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_node_consciousness_levels_parametrized(consciousness_level):
    """Parametrized test for different consciousness levels."""
    node = MatrixNode(id=f"param_node_{consciousness_level}", consciousness_level=consciousness_level)
    assert node.consciousness_level == consciousness_level


@pytest.mark.parametrize("dimension", [1, 16, 64, 128, 256, 512])
def test_matrix_dimensions_parametrized(dimension):
    """Parametrized test for different matrix dimensions."""
    config = {'dimension': dimension, 'consciousness_threshold': 0.5}
    matrix = GenesisConsciousnessMatrix(config=config)
    assert matrix.dimension == dimension


class TestMatrixDocumentationExamples(unittest.TestCase):
    """Test examples from documentation to ensure they work."""
    
    def test_basic_usage_example(self):
        """Test basic usage example from documentation."""
        # Example from documentation
        matrix = GenesisConsciousnessMatrix()
        
        # Add nodes
        node1 = MatrixNode(id="example_node1", consciousness_level=0.3)
        node2 = MatrixNode(id="example_node2", consciousness_level=0.7)
        matrix.add_node(node1)
        matrix.add_node(node2)
        
        # Calculate consciousness
        level = matrix.calculate_consciousness_level()
        self.assertAlmostEqual(level, 0.5, places=2)
        
        # Evolve matrix
        matrix.evolve_step()
        
        # Verify it's still functional
        self.assertIsInstance(matrix.get_state_snapshot(), dict)
    
    def test_advanced_usage_example(self):
        """Test advanced usage example from documentation."""
        # Advanced example with connections
        matrix = GenesisConsciousnessMatrix()
        
        # Create network of nodes
        for i in range(5):
            node = MatrixNode(id=f"network_node_{i}", consciousness_level=0.2 + i * 0.15)
            matrix.add_node(node)
        
        # Connect nodes
        for i in range(4):
            matrix.connect_nodes(f"network_node_{i}", f"network_node_{i+1}", strength=0.6)
        
        # Evolve until convergence
        matrix.evolve_until_convergence(max_iterations=20)
        
        # Verify convergence
        self.assertTrue(matrix.has_converged())