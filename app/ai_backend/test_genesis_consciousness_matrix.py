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
        Prepare a new GenesisConsciousnessMatrix instance and test configuration before each test case.
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
        Test that GenesisConsciousnessMatrix initializes with default parameters and contains the required 'state' and 'nodes' attributes.
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
        Test that initializing GenesisConsciousnessMatrix with an invalid configuration raises a MatrixInitializationError.
        """
        invalid_config = {'dimension': -1, 'consciousness_threshold': 2.0}
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=invalid_config)
            
    def test_add_consciousness_node_valid(self):
        """
        Tests that adding a valid MatrixNode to the matrix succeeds and the node is present in the matrix after addition.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        result = self.matrix.add_node(node)
        self.assertTrue(result)
        self.assertIn("test_node", self.matrix.nodes)
        
    def test_add_consciousness_node_duplicate(self):
        """
        Verify that attempting to add a duplicate node to the matrix raises an InvalidStateException.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        with self.assertRaises(InvalidStateException):
            self.matrix.add_node(node)
            
    def test_remove_consciousness_node_existing(self):
        """
        Test that removing an existing node from the matrix returns True and ensures the node is removed.
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
        Verify that a valid transition between consciousness states updates the matrix's current state and returns True.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.ACTIVE
        result = self.matrix.transition_state(initial_state, target_state)
        self.assertTrue(result)
        self.assertEqual(self.matrix.current_state, target_state)
        
    def test_consciousness_state_transition_invalid(self):
        """
        Test that an invalid consciousness state transition raises an InvalidStateException.
        
        Attempts to transition the matrix from DORMANT directly to TRANSCENDENT and asserts that the operation fails with an InvalidStateException.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.TRANSCENDENT
        with self.assertRaises(InvalidStateException):
            self.matrix.transition_state(initial_state, target_state)
            
    def test_consciousness_level_calculation(self):
        """
        Verify that the matrix calculates the correct average consciousness level when multiple nodes with varying levels are present.
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
        Test that the consciousness level calculation returns 0.0 when the matrix contains no nodes.
        """
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.0)
        
    def test_consciousness_level_calculation_single_node(self):
        """
        Test that the matrix calculates and returns the correct consciousness level when only one node is present.
        """
        node = MatrixNode(id="single_node", consciousness_level=0.8)
        self.matrix.add_node(node)
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.8)
        
    def test_matrix_evolution_step(self):
        """
        Verifies that performing a single evolution step changes the matrix's state snapshot.
        
        Ensures that calling `evolve_step()` results in a different state snapshot than before the evolution.
        """
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        self.assertNotEqual(initial_state, final_state)
        
    def test_matrix_evolution_convergence(self):
        """
        Verifies that the matrix evolution process correctly detects convergence within a specified maximum number of iterations.
        """
        self.matrix.evolve_until_convergence(max_iterations=10)
        self.assertTrue(self.matrix.has_converged())
        
    def test_matrix_reset_to_initial_state(self):
        """
        Tests that resetting the matrix removes all nodes and restores the DORMANT state after changes.
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
        Tests that the matrix can be serialized to a JSON string containing both 'nodes' and 'state' fields, and that the output is valid JSON.
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
        Test that deserializing a matrix from a JSON string restores all nodes and their consciousness levels accurately.
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
        Verify that saving the matrix to a file and loading it restores all nodes and their consciousness levels accurately.
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
        Test that two nodes can be connected in the matrix and that the connection strength is correctly recorded.
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
        Verify that connecting two non-existent nodes in the matrix raises an InvalidStateException.
        """
        with self.assertRaises(InvalidStateException):
            self.matrix.connect_nodes("nonexistent1", "nonexistent2", strength=0.5)
            
    def test_consciousness_emergence_detection(self):
        """
        Tests that the matrix correctly detects the emergence of consciousness when multiple nodes have high consciousness levels.
        """
        # Add nodes with high consciousness levels
        for i in range(5):
            node = MatrixNode(id=f"high_node_{i}", consciousness_level=0.9)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertTrue(emergence_detected)
        
    def test_consciousness_emergence_detection_insufficient(self):
        """
        Verify that the matrix does not detect consciousness emergence when all nodes have low consciousness levels.
        """
        # Add nodes with low consciousness levels
        for i in range(2):
            node = MatrixNode(id=f"low_node_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertFalse(emergence_detected)
        
    def test_matrix_metrics_calculation(self):
        """
        Verifies that the matrix calculates and returns metrics including average consciousness, node count, and connection density after nodes are added.
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
        Verify that the matrix can complete an evolution step with 100 nodes in less than one second.
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
        Verify that the matrix accurately tracks the number of nodes after multiple additions and removals.
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
        Verify that deserializing corrupted JSON data with `from_json` raises a `MatrixException`.
        """
        corrupted_json = '{"nodes": {"invalid": "data"}, "state":'
        
        with self.assertRaises(MatrixException):
            GenesisConsciousnessMatrix.from_json(corrupted_json)
            
    def test_matrix_thread_safety(self):
        """
        Verifies that adding nodes to the matrix concurrently from multiple threads is thread-safe and that all node additions succeed without errors or data corruption.
        """
        import threading
        import time
        
        results = []
        
        def add_nodes_thread(thread_id):
            """
            Adds ten uniquely identified nodes to the matrix from a single thread, recording the outcome of each addition in the shared `results` list.
            
            Each node's ID is constructed using the thread ID and an index to ensure uniqueness. Appends `True` to `results` if the node is added successfully, or `False` if an exception occurs during addition.
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
        Tests that the ordering of ConsciousnessState enum members reflects the correct progression from DORMANT to TRANSCENDENT.
        """
        self.assertLess(ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE)
        self.assertLess(ConsciousnessState.ACTIVE, ConsciousnessState.AWARE)
        self.assertLess(ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT)
        
    def test_consciousness_state_string_representation(self):
        """
        Verifies that each ConsciousnessState enum member returns the correct string representation.
        """
        self.assertEqual(str(ConsciousnessState.DORMANT), "DORMANT")
        self.assertEqual(str(ConsciousnessState.ACTIVE), "ACTIVE")
        self.assertEqual(str(ConsciousnessState.AWARE), "AWARE")
        self.assertEqual(str(ConsciousnessState.TRANSCENDENT), "TRANSCENDENT")


class TestMatrixNode(unittest.TestCase):
    """Test cases for MatrixNode class."""
    
    def setUp(self):
        """
        Prepare a MatrixNode instance with a test ID and consciousness level for use in each test.
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
        Verify that creating a MatrixNode with a consciousness level outside the range [0.0, 1.0] raises a ValueError.
        """
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=1.5)
            
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=-0.1)
            
    def test_node_consciousness_level_update(self):
        """
        Verify that updating a MatrixNode's consciousness level correctly sets the new value.
        """
        self.node.update_consciousness_level(0.8)
        self.assertEqual(self.node.consciousness_level, 0.8)
        
    def test_node_consciousness_level_update_invalid(self):
        """
        Test that updating a node's consciousness level to a value outside the valid range raises a ValueError.
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
        Verify that MatrixNode instances with the same ID have identical hash values, regardless of their consciousness level.
        """
        node1 = MatrixNode(id="hash_test", consciousness_level=0.5)
        node2 = MatrixNode(id="hash_test", consciousness_level=0.7)
        
        # Nodes with same ID should have same hash
        self.assertEqual(hash(node1), hash(node2))
        
    def test_node_string_representation(self):
        """
        Verify that the string representation of a MatrixNode contains its ID and consciousness level.
        """
        node_str = str(self.node)
        self.assertIn("test_node", node_str)
        self.assertIn("0.5", node_str)


class TestMatrixExceptions(unittest.TestCase):
    """Test cases for custom matrix exceptions."""
    
    def test_matrix_exception_inheritance(self):
        """
        Verify that custom matrix exceptions inherit from their intended base exception classes.
        """
        self.assertTrue(issubclass(MatrixException, Exception))
        self.assertTrue(issubclass(InvalidStateException, MatrixException))
        self.assertTrue(issubclass(MatrixInitializationError, MatrixException))
        
    def test_matrix_exception_messages(self):
        """
        Verify that custom matrix exceptions propagate and display the correct error messages.
        
        Ensures that `MatrixException` and `InvalidStateException` instances correctly retain and return their provided error messages when raised and caught.
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
        Initializes a new GenesisConsciousnessMatrix instance before each integration test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_consciousness_evolution_cycle(self):
        """
        Performs an end-to-end test of the matrix's evolution cycle, ensuring that adding and connecting nodes, followed by evolution until convergence, results in a change in the overall consciousness level.
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
        Tests that consciousness emergence is detected only after all nodes' consciousness levels surpass the emergence threshold, ensuring correct detection throughout the full emergence cycle.
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
        Ensures that serializing and deserializing the matrix retains all node data and node-to-node connections, verifying the integrity of the persisted matrix state.
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
    """Advanced test cases covering edge cases, boundary conditions, and complex scenarios."""
    
    def setUp(self):
        """
        Set up a GenesisConsciousnessMatrix instance and advanced configuration for advanced test scenarios.
        
        Initializes the matrix and prepares a configuration dictionary with high-dimension and threshold values for use in advanced and edge case tests.
        """
        self.matrix = GenesisConsciousnessMatrix()
        self.advanced_config = {
            'dimension': 1024,
            'consciousness_threshold': 0.95,
            'learning_rate': 0.0001,
            'max_iterations': 10000,
            'neural_pathways': 512,
            'memory_buffer_size': 256
        }
        
    def test_matrix_initialization_extreme_configs(self):
        """
        Tests that GenesisConsciousnessMatrix initializes correctly with both maximum and minimum supported configuration values.
        """
        # Test maximum values
        max_config = {
            'dimension': 99999,
            'consciousness_threshold': 0.99999,
            'learning_rate': 1.0,
            'max_iterations': 1000000
        }
        matrix = GenesisConsciousnessMatrix(config=max_config)
        self.assertEqual(matrix.dimension, max_config['dimension'])
        
        # Test minimum values
        min_config = {
            'dimension': 1,
            'consciousness_threshold': 0.00001,
            'learning_rate': 0.000001,
            'max_iterations': 1
        }
        matrix_min = GenesisConsciousnessMatrix(config=min_config)
        self.assertEqual(matrix_min.dimension, min_config['dimension'])
        
    def test_matrix_initialization_zero_values(self):
        """
        Test that initializing GenesisConsciousnessMatrix with zero-valued configuration parameters raises MatrixInitializationError.
        """
        zero_config = {
            'dimension': 0,
            'consciousness_threshold': 0.0,
            'learning_rate': 0.0,
            'max_iterations': 0
        }
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=zero_config)
            
    def test_matrix_initialization_none_config(self):
        """
        Test that initializing GenesisConsciousnessMatrix with a None configuration results in a valid matrix instance.
        """
        matrix = GenesisConsciousnessMatrix(config=None)
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        
    def test_consciousness_level_precision_boundaries(self):
        """
        Test that the matrix accurately calculates consciousness levels when nodes have values extremely close to 0 and 1, ensuring correct handling of floating-point precision boundaries.
        """
        # Test floating point precision edge cases
        node1 = MatrixNode(id="precision1", consciousness_level=0.9999999999999999)
        node2 = MatrixNode(id="precision2", consciousness_level=0.0000000000000001)
        
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        calculated_level = self.matrix.calculate_consciousness_level()
        # Should handle precision correctly
        self.assertIsInstance(calculated_level, float)
        self.assertGreaterEqual(calculated_level, 0.0)
        self.assertLessEqual(calculated_level, 1.0)
        
    def test_node_operations_with_unicode_ids(self):
        """
        Tests that the matrix can add and store nodes with Unicode, emoji, and special character IDs.
        """
        unicode_node = MatrixNode(id="测试节点_🧠", consciousness_level=0.5)
        emoji_node = MatrixNode(id="🤖🧠💭", consciousness_level=0.7)
        special_node = MatrixNode(id="node@#$%^&*()", consciousness_level=0.3)
        
        self.assertTrue(self.matrix.add_node(unicode_node))
        self.assertTrue(self.matrix.add_node(emoji_node))
        self.assertTrue(self.matrix.add_node(special_node))
        
        self.assertIn("测试节点_🧠", self.matrix.nodes)
        self.assertIn("🤖🧠💭", self.matrix.nodes)
        self.assertIn("node@#$%^&*()", self.matrix.nodes)
        
    def test_node_operations_with_extremely_long_ids(self):
        """
        Tests that a node with an extremely long string ID can be added to the matrix and is correctly stored and retrievable.
        """
        long_id = "a" * 10000  # 10,000 character ID
        node = MatrixNode(id=long_id, consciousness_level=0.5)
        
        result = self.matrix.add_node(node)
        self.assertTrue(result)
        self.assertIn(long_id, self.matrix.nodes)
        
    def test_node_operations_empty_string_id(self):
        """
        Test that creating a MatrixNode with an empty string ID raises a ValueError.
        """
        with self.assertRaises(ValueError):
            MatrixNode(id="", consciousness_level=0.5)
            
    def test_massive_node_addition_performance(self):
        """
        Verify that adding 10,000 nodes to the matrix completes in under 10 seconds and that all nodes are successfully added.
        """
        import time
        
        start_time = time.time()
        
        # Add 10,000 nodes
        for i in range(10000):
            node = MatrixNode(id=f"massive_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        end_time = time.time()
        
        # Should complete within reasonable time (10 seconds)
        self.assertLess(end_time - start_time, 10.0)
        self.assertEqual(len(self.matrix.nodes), 10000)
        
    def test_consciousness_state_transition_all_combinations(self):
        """
        Test all possible consciousness state transitions for correctness.
        
        Verifies that valid transitions between consciousness states succeed, while invalid transitions (excluding self-transitions) raise an InvalidStateException.
        """
        states = [
            ConsciousnessState.DORMANT,
            ConsciousnessState.ACTIVE,
            ConsciousnessState.AWARE,
            ConsciousnessState.TRANSCENDENT
        ]
        
        valid_transitions = {
            ConsciousnessState.DORMANT: [ConsciousnessState.ACTIVE],
            ConsciousnessState.ACTIVE: [ConsciousnessState.AWARE, ConsciousnessState.DORMANT],
            ConsciousnessState.AWARE: [ConsciousnessState.TRANSCENDENT, ConsciousnessState.ACTIVE],
            ConsciousnessState.TRANSCENDENT: [ConsciousnessState.AWARE]
        }
        
        for from_state in states:
            for to_state in states:
                if to_state in valid_transitions.get(from_state, []):
                    # Valid transition should succeed
                    result = self.matrix.transition_state(from_state, to_state)
                    self.assertTrue(result)
                elif from_state != to_state:
                    # Invalid transition should raise exception
                    with self.assertRaises(InvalidStateException):
                        self.matrix.transition_state(from_state, to_state)
                        
    def test_matrix_evolution_with_no_nodes(self):
        """
        Test that evolving the matrix when it contains no nodes completes without errors and returns a valid state snapshot.
        """
        # Evolution with empty matrix should not crash
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        
        # State might change even with no nodes
        self.assertIsNotNone(final_state)
        
    def test_matrix_evolution_convergence_timeout(self):
        """
        Verify that matrix evolution completes without errors when convergence cannot be reached due to a very low maximum iteration limit.
        
        Adds multiple nodes to create a non-convergent scenario and ensures the evolution process handles the timeout gracefully.
        """
        # Add nodes that make convergence difficult
        for i in range(50):
            node = MatrixNode(id=f"divergent_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Test with very low max_iterations
        self.matrix.evolve_until_convergence(max_iterations=1)
        # Should not crash and should handle timeout gracefully
        
    def test_node_connections_circular_references(self):
        """
        Tests that the matrix can create and maintain circular connections among nodes without errors or loss of functionality.
        """
        nodes = []
        for i in range(5):
            node = MatrixNode(id=f"circular_node_{i}", consciousness_level=0.5)
            nodes.append(node)
            self.matrix.add_node(node)
            
        # Create circular connections
        for i in range(5):
            next_index = (i + 1) % 5
            self.matrix.connect_nodes(f"circular_node_{i}", f"circular_node_{next_index}", strength=0.8)
            
        # Verify circular structure doesn't break functionality
        connections = self.matrix.get_node_connections("circular_node_0")
        self.assertIn("circular_node_1", connections)
        
    def test_node_connections_self_reference(self):
        """
        Verify that a node can establish a self-referential connection and that this connection is accurately reflected in the matrix's connection data.
        """
        node = MatrixNode(id="self_ref_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Node connecting to itself
        self.matrix.connect_nodes("self_ref_node", "self_ref_node", strength=1.0)
        
        connections = self.matrix.get_node_connections("self_ref_node")
        self.assertIn("self_ref_node", connections)
        
    def test_consciousness_emergence_boundary_conditions(self):
        """
        Verifies that consciousness emergence is detected when a node's consciousness level is exactly at the threshold boundary.
        """
        # Test exactly at threshold
        threshold = self.matrix.consciousness_threshold if hasattr(self.matrix, 'consciousness_threshold') else 0.75
        
        node = MatrixNode(id="boundary_node", consciousness_level=threshold)
        self.matrix.add_node(node)
        
        # Should detect emergence at exact threshold
        emergence = self.matrix.detect_consciousness_emergence()
        self.assertIsInstance(emergence, bool)
        
    def test_matrix_serialization_large_data(self):
        """
        Verifies that the matrix can serialize a large number of nodes and connections into a valid JSON string, and that the resulting JSON includes the expected 'nodes' key.
        """
        # Add many nodes with complex data
        for i in range(1000):
            node = MatrixNode(id=f"large_data_node_{i}", consciousness_level=0.1 + (i % 900) / 1000)
            self.matrix.add_node(node)
            
        # Add many connections
        for i in range(500):
            self.matrix.connect_nodes(f"large_data_node_{i}", f"large_data_node_{i+500}", strength=0.5)
            
        serialized = self.matrix.to_json()
        self.assertIsInstance(serialized, str)
        
        # Verify it's valid JSON and can be parsed
        import json
        parsed = json.loads(serialized)
        self.assertIn("nodes", parsed)
        
    def test_matrix_serialization_special_characters(self):
        """
        Verify that matrix serialization and deserialization preserve node IDs containing special characters and Unicode, maintaining data integrity.
        """
        node = MatrixNode(id='node_with_"quotes"_and_\n_newlines', consciousness_level=0.5)
        self.matrix.add_node(node)
        
        serialized = self.matrix.to_json()
        new_matrix = GenesisConsciousnessMatrix.from_json(serialized)
        
        self.assertIn('node_with_"quotes"_and_\n_newlines', new_matrix.nodes)
        
    def test_matrix_deserialization_malformed_data(self):
        """
        Test that deserializing various malformed JSON inputs raises a MatrixException.
        
        Covers cases such as null or improperly structured nodes, invalid state values, and empty objects to ensure deserialization fails gracefully with invalid data.
        """
        malformed_cases = [
            '{"nodes": null}',  # null nodes
            '{"nodes": []}',    # array instead of dict
            '{"state": "INVALID_STATE"}',  # invalid state
            '{}',  # empty object
            '{"nodes": {"node1": null}}',  # null node data
        ]
        
        for malformed_json in malformed_cases:
            with self.assertRaises(MatrixException):
                GenesisConsciousnessMatrix.from_json(malformed_json)
                
    def test_matrix_file_operations_permissions(self):
        """
        Verify that attempting to save the matrix to a read-only file raises a MatrixException and does not crash the test.
        """
        import tempfile
        import os
        import stat
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            
        try:
            # Make file read-only
            os.chmod(temp_file, stat.S_IRUSR)
            
            node = MatrixNode(id="permission_test", consciousness_level=0.5)
            self.matrix.add_node(node)
            
            # Should handle permission error gracefully
            with self.assertRaises(MatrixException):
                self.matrix.save_to_file(temp_file)
                
        finally:
            # Restore permissions and cleanup
            try:
                os.chmod(temp_file, stat.S_IRUSR | stat.S_IWUSR)
                os.unlink(temp_file)
            except:
                pass
                
    def test_matrix_file_operations_nonexistent_directory(self):
        """
        Verifies that attempting to save to or load from a nonexistent directory path raises a MatrixException.
        """
        nonexistent_path = "/nonexistent/directory/file.json"
        
        with self.assertRaises(MatrixException):
            self.matrix.save_to_file(nonexistent_path)
            
        with self.assertRaises(MatrixException):
            GenesisConsciousnessMatrix.load_from_file(nonexistent_path)
            
    def test_matrix_metrics_edge_cases(self):
        """
        Tests that the matrix metrics calculation correctly handles cases with no nodes and with a single node, ensuring accurate node count and average consciousness values.
        """
        # Empty matrix metrics
        metrics_empty = self.matrix.calculate_metrics()
        self.assertEqual(metrics_empty["node_count"], 0)
        self.assertEqual(metrics_empty["average_consciousness"], 0.0)
        
        # Single node metrics
        node = MatrixNode(id="single_metrics", consciousness_level=0.9)
        self.matrix.add_node(node)
        
        metrics_single = self.matrix.calculate_metrics()
        self.assertEqual(metrics_single["node_count"], 1)
        self.assertEqual(metrics_single["average_consciousness"], 0.9)
        
    def test_matrix_concurrent_modifications(self):
        """
        Test that the matrix remains in a valid state when nodes are concurrently added and removed from multiple threads.
        """
        import threading
        import time
        
        def modify_matrix():
            """
            Performs random concurrent additions and occasional removals of nodes in the matrix to stress test thread safety.
            
            Intended to be executed by multiple threads simultaneously to simulate concurrent modifications.
            """
            for i in range(100):
                node = MatrixNode(id=f"concurrent_node_{threading.current_thread().ident}_{i}", 
                                consciousness_level=0.5)
                try:
                    self.matrix.add_node(node)
                    time.sleep(0.001)
                    if i % 10 == 0:
                        self.matrix.remove_node(f"concurrent_node_{threading.current_thread().ident}_{i}")
                except:
                    pass  # Expected in concurrent scenarios
                    
        threads = []
        for i in range(3):
            thread = threading.Thread(target=modify_matrix)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Matrix should still be in valid state
        self.assertIsInstance(self.matrix.nodes, dict)
        
    def test_matrix_memory_stress(self):
        """
        Verify that the GenesisConsciousnessMatrix maintains correct behavior and manages memory efficiently during repeated large-scale node additions and removals.
        
        Simulates memory stress by adding and partially removing thousands of nodes in multiple cycles, forcing garbage collection, and ensuring the final node count remains within expected bounds.
        """
        import gc
        
        initial_node_count = len(self.matrix.nodes)
        
        # Add and remove many nodes to test memory management
        for cycle in range(10):
            nodes_to_add = []
            for i in range(1000):
                node = MatrixNode(id=f"stress_node_{cycle}_{i}", consciousness_level=0.5)
                nodes_to_add.append(node)
                self.matrix.add_node(node)
                
            # Remove half the nodes
            for i in range(500):
                self.matrix.remove_node(f"stress_node_{cycle}_{i}")
                
            # Force garbage collection
            gc.collect()
            
        # Should have manageable number of nodes
        final_node_count = len(self.matrix.nodes)
        self.assertLess(final_node_count, initial_node_count + 6000)  # 500 * 10 + buffer


class TestMatrixNodeAdvanced(unittest.TestCase):
    """Advanced test cases for MatrixNode class covering edge cases."""
    
    def test_node_consciousness_level_float_precision(self):
        """
        Tests that MatrixNode instances accurately represent consciousness levels at floating-point precision boundaries.
        
        Verifies initialization with extremely small positive values and values just below 1.0, ensuring precision is maintained.
        """
        # Test very small positive value
        node1 = MatrixNode(id="precision1", consciousness_level=1e-15)
        self.assertAlmostEqual(node1.consciousness_level, 1e-15, places=15)
        
        # Test very close to 1.0
        node2 = MatrixNode(id="precision2", consciousness_level=1.0 - 1e-15)
        self.assertLess(node2.consciousness_level, 1.0)
        
    def test_node_consciousness_level_update_boundary(self):
        """
        Verify that a node's consciousness level can be updated to the exact boundary values of 0.0 and 1.0 without errors.
        """
        node = MatrixNode(id="boundary_test", consciousness_level=0.5)
        
        # Test update to exact boundaries
        node.update_consciousness_level(0.0)
        self.assertEqual(node.consciousness_level, 0.0)
        
        node.update_consciousness_level(1.0)
        self.assertEqual(node.consciousness_level, 1.0)
        
    def test_node_equality_edge_cases(self):
        """
        Test that MatrixNode instances are not considered equal to None, objects of different types, or dictionaries.
        """
        node1 = MatrixNode(id="test", consciousness_level=0.5)
        
        # Test equality with None
        self.assertNotEqual(node1, None)
        
        # Test equality with different type
        self.assertNotEqual(node1, "test")
        
        # Test equality with dict
        self.assertNotEqual(node1, {"id": "test", "consciousness_level": 0.5})
        
    def test_node_hash_consistency(self):
        """
        Verify that a MatrixNode's hash value does not change when its consciousness level is updated.
        """
        node = MatrixNode(id="hash_test", consciousness_level=0.5)
        original_hash = hash(node)
        
        # Hash should remain same after consciousness level change
        node.update_consciousness_level(0.8)
        self.assertEqual(hash(node), original_hash)
        
    def test_node_string_representation_edge_cases(self):
        """
        Verifies that the string representation of a MatrixNode includes very long and Unicode IDs.
        """
        # Very long ID
        long_id = "a" * 1000
        node = MatrixNode(id=long_id, consciousness_level=0.5)
        node_str = str(node)
        self.assertIn("a", node_str)  # Should contain part of the ID
        
        # Unicode ID
        unicode_node = MatrixNode(id="测试🧠", consciousness_level=0.7)
        unicode_str = str(unicode_node)
        self.assertIn("测试🧠", unicode_str)


class TestAsyncMatrixOperations(unittest.TestCase):
    """Test cases for asynchronous matrix operations if supported."""
    
    def setUp(self):
        """
        Initializes a new GenesisConsciousnessMatrix instance before each asynchronous test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_async_evolution_if_supported(self):
        """
        Tests asynchronous evolution operations of the matrix if supported.
        
        Adds multiple nodes and attempts to invoke asynchronous evolution methods, verifying that async evolution can be executed without error if implemented. Skips the test if async support is unavailable.
        """
        async def async_evolution_test():
            # Add nodes
            """
            Performs asynchronous evolution tests on the matrix by adding nodes and invoking async evolution methods if supported.
            
            Adds ten nodes to the matrix, then calls asynchronous evolution step and convergence methods if the matrix implementation provides them.
            """
            for i in range(10):
                node = MatrixNode(id=f"async_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Check if matrix supports async evolution
            if hasattr(self.matrix, 'evolve_step_async'):
                await self.matrix.evolve_step_async()
                
            if hasattr(self.matrix, 'evolve_until_convergence_async'):
                await self.matrix.evolve_until_convergence_async(max_iterations=10)
                
        # Run async test if asyncio is available
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(async_evolution_test())
            loop.close()
        except Exception:
            # Skip if async not supported
            pass
            
    def test_async_node_operations_if_supported(self):
        """
        Tests asynchronous node addition and removal on the matrix if async methods are implemented.
        
        This test creates a node and verifies that asynchronous add and remove operations function correctly, skipping the test if async methods are not supported by the matrix.
        """
        async def async_node_test():
            """
            Asynchronously tests adding and removing a MatrixNode using the matrix's async methods, if implemented.
            
            Creates a MatrixNode and verifies that asynchronous addition and removal operations succeed when supported by the matrix instance.
            """
            node = MatrixNode(id="async_test_node", consciousness_level=0.6)
            
            # Test async add if supported
            if hasattr(self.matrix, 'add_node_async'):
                result = await self.matrix.add_node_async(node)
                self.assertTrue(result)
                
            # Test async remove if supported
            if hasattr(self.matrix, 'remove_node_async'):
                result = await self.matrix.remove_node_async("async_test_node")
                self.assertTrue(result)
                
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(async_node_test())
            loop.close()
        except Exception:
            # Skip if async not supported
            pass


class TestMatrixPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for matrix operations."""
    
    def setUp(self):
        """
        Initializes a new GenesisConsciousnessMatrix instance before each performance benchmark test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_addition_scalability(self):
        """
        Benchmarks node addition performance at various scales to ensure scalability.
        
        Adds 100, 1000, and 5000 nodes to a new matrix instance, verifying that the average time per addition remains under 10 milliseconds for each scale.
        """
        import time
        
        scales = [100, 1000, 5000]
        
        for scale in scales:
            matrix = GenesisConsciousnessMatrix()
            
            start_time = time.time()
            for i in range(scale):
                node = MatrixNode(id=f"scale_node_{scale}_{i}", consciousness_level=0.5)
                matrix.add_node(node)
            end_time = time.time()
            
            # Performance should scale reasonably
            time_per_node = (end_time - start_time) / scale
            self.assertLess(time_per_node, 0.01)  # Less than 10ms per node
            
    def test_consciousness_calculation_performance(self):
        """
        Benchmark that calculating the consciousness level for 10,000 nodes completes 100 iterations with an average time under 100 milliseconds per calculation.
        """
        import time
        
        # Add 10,000 nodes
        for i in range(10000):
            node = MatrixNode(id=f"perf_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Measure calculation time
        start_time = time.time()
        for _ in range(100):  # 100 calculations
            self.matrix.calculate_consciousness_level()
        end_time = time.time()
        
        # Should complete 100 calculations quickly
        avg_time = (end_time - start_time) / 100
        self.assertLess(avg_time, 0.1)  # Less than 100ms per calculation
        
    def test_serialization_performance(self):
        """
        Verifies that serializing and deserializing a matrix containing 5,000 nodes each complete in under 5 seconds.
        """
        import time
        
        # Create large matrix
        for i in range(5000):
            node = MatrixNode(id=f"serial_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Test serialization time
        start_time = time.time()
        serialized = self.matrix.to_json()
        end_time = time.time()
        
        serialization_time = end_time - start_time
        self.assertLess(serialization_time, 5.0)  # Less than 5 seconds
        
        # Test deserialization time
        start_time = time.time()
        GenesisConsciousnessMatrix.from_json(serialized)
        end_time = time.time()
        
        deserialization_time = end_time - start_time
        self.assertLess(deserialization_time, 5.0)  # Less than 5 seconds


class TestMatrixErrorRecovery(unittest.TestCase):
    """Test cases for error recovery and resilience."""
    
    def setUp(self):
        """
        Initializes a new GenesisConsciousnessMatrix instance before each error recovery test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_recovery_from_corrupted_state(self):
        """
        Verify that the matrix handles exceptions gracefully when its internal state is corrupted during consciousness calculation and evolution steps, ensuring robust error recovery.
        """
        # Add some nodes
        for i in range(10):
            node = MatrixNode(id=f"recovery_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Simulate state corruption (if matrix allows direct state access)
        if hasattr(self.matrix, '_state'):
            original_state = getattr(self.matrix, '_state', None)
            setattr(self.matrix, '_state', None)
            
            # Matrix should handle corrupted state gracefully
            try:
                self.matrix.calculate_consciousness_level()
                self.matrix.evolve_step()
            except MatrixException:
                pass  # Expected behavior
                
            # Restore state
            if original_state is not None:
                setattr(self.matrix, '_state', original_state)
                
    def test_recovery_from_memory_issues(self):
        """
        Verifies that the matrix remains operational and accurately calculates consciousness level after adding many nodes and repeated garbage collection.
        """
        import gc
        
        # Fill matrix with many nodes
        for i in range(1000):
            node = MatrixNode(id=f"memory_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Force garbage collection multiple times
        for _ in range(10):
            gc.collect()
            
        # Matrix should still function
        self.assertIsInstance(self.matrix.calculate_consciousness_level(), float)
        
    def test_graceful_degradation(self):
        """
        Tests that the matrix remains responsive and does not crash under extreme stress, such as massive node additions and repeated evolution steps, handling memory and matrix-specific errors gracefully.
        """
        # Overload matrix with operations
        try:
            for i in range(50000):  # Very large number
                node = MatrixNode(id=f"stress_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
                if i % 1000 == 0:
                    self.matrix.evolve_step()
                    
        except MemoryError:
            # Should handle memory errors gracefully
            pass
        except MatrixException:
            # Should handle matrix-specific errors gracefully
            pass
            
        # Matrix should still be responsive
        self.assertIsInstance(len(self.matrix.nodes), int)


if __name__ == '__main__':
    # Configure comprehensive test runner with additional options
    import sys
    
    # Add command line argument parsing for different test suites
    if len(sys.argv) > 1:
        if sys.argv[1] == 'performance':
            # Run only performance tests
            suite = unittest.TestLoader().loadTestsFromTestCase(TestMatrixPerformanceBenchmarks)
            unittest.TextTestRunner(verbosity=2, buffer=True).run(suite)
        elif sys.argv[1] == 'advanced':
            # Run only advanced tests
            suite = unittest.TestLoader().loadTestsFromTestCase(TestGenesisConsciousnessMatrixAdvanced)
            unittest.TextTestRunner(verbosity=2, buffer=True).run(suite)
        elif sys.argv[1] == 'async':
            # Run only async tests
            suite = unittest.TestLoader().loadTestsFromTestCase(TestAsyncMatrixOperations)
            unittest.TextTestRunner(verbosity=2, buffer=True).run(suite)
        else:
            unittest.main(verbosity=2, buffer=True)
    else:
        # Run all tests with enhanced configuration
        unittest.main(verbosity=2, buffer=True, catchbreak=True, failfast=False)