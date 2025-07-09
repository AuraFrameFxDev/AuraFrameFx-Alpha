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
        Prepare a fresh GenesisConsciousnessMatrix instance and test configuration before each test.
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
        Cleans up the test environment after each test by calling the matrix's cleanup method if it exists.
        """
        if hasattr(self.matrix, 'cleanup'):
            self.matrix.cleanup()
    
    def test_matrix_initialization_default(self):
        """
        Verify that a GenesisConsciousnessMatrix instance initialized with default parameters contains the 'state' and 'nodes' attributes.
        """
        matrix = GenesisConsciousnessMatrix()
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        self.assertTrue(hasattr(matrix, 'state'))
        self.assertTrue(hasattr(matrix, 'nodes'))
        
    def test_matrix_initialization_custom_config(self):
        """
        Verify that initializing GenesisConsciousnessMatrix with a custom configuration applies the specified dimension and consciousness threshold.
        """
        matrix = GenesisConsciousnessMatrix(config=self.test_config)
        self.assertEqual(matrix.dimension, self.test_config['dimension'])
        self.assertEqual(matrix.consciousness_threshold, self.test_config['consciousness_threshold'])
        
    def test_matrix_initialization_invalid_config(self):
        """
        Test that initializing the matrix with invalid configuration parameters raises a MatrixInitializationError.
        
        This ensures that negative dimensions or out-of-range consciousness thresholds are properly rejected during initialization.
        """
        invalid_config = {'dimension': -1, 'consciousness_threshold': 2.0}
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=invalid_config)
            
    def test_add_consciousness_node_valid(self):
        """
        Test that adding a valid MatrixNode to the matrix succeeds and the node is present in the matrix's node collection.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        result = self.matrix.add_node(node)
        self.assertTrue(result)
        self.assertIn("test_node", self.matrix.nodes)
        
    def test_add_consciousness_node_duplicate(self):
        """
        Test that adding a node with a duplicate ID to the matrix raises an InvalidStateException.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        with self.assertRaises(InvalidStateException):
            self.matrix.add_node(node)
            
    def test_remove_consciousness_node_existing(self):
        """
        Tests that removing an existing node by ID returns True and ensures the node is removed from the matrix.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        result = self.matrix.remove_node("test_node")
        self.assertTrue(result)
        self.assertNotIn("test_node", self.matrix.nodes)
        
    def test_remove_consciousness_node_nonexistent(self):
        """
        Test that attempting to remove a node that does not exist in the matrix returns False.
        """
        result = self.matrix.remove_node("nonexistent_node")
        self.assertFalse(result)
        
    def test_consciousness_state_transition_valid(self):
        """
        Test that a valid consciousness state transition updates the matrix's current state and returns True.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.ACTIVE
        result = self.matrix.transition_state(initial_state, target_state)
        self.assertTrue(result)
        self.assertEqual(self.matrix.current_state, target_state)
        
    def test_consciousness_state_transition_invalid(self):
        """
        Test that attempting an invalid transition between consciousness states raises an InvalidStateException.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.TRANSCENDENT
        with self.assertRaises(InvalidStateException):
            self.matrix.transition_state(initial_state, target_state)
            
    def test_consciousness_level_calculation(self):
        """
        Verifies that the matrix computes the correct average consciousness level when multiple nodes with varying levels are present.
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
        Tests that calculating the consciousness level of an empty matrix returns 0.0.
        """
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.0)
        
    def test_consciousness_level_calculation_single_node(self):
        """
        Verifies that the matrix returns the correct consciousness level when it contains only one node.
        """
        node = MatrixNode(id="single_node", consciousness_level=0.8)
        self.matrix.add_node(node)
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.8)
        
    def test_matrix_evolution_step(self):
        """
        Verify that a single evolution step updates the matrix's state.
        
        Ensures that invoking `evolve_step()` results in a different state snapshot, confirming the matrix evolves as expected.
        """
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        self.assertNotEqual(initial_state, final_state)
        
    def test_matrix_evolution_convergence(self):
        """
        Test that the matrix evolution process detects convergence within a specified maximum number of iterations.
        """
        self.matrix.evolve_until_convergence(max_iterations=10)
        self.assertTrue(self.matrix.has_converged())
        
    def test_matrix_reset_to_initial_state(self):
        """
        Verifies that resetting the matrix removes all nodes and sets its state to DORMANT.
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
        Verify that the matrix serializes to a JSON string containing the correct nodes and state fields.
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
        Tests that deserializing a matrix from a JSON string restores all nodes and their consciousness levels correctly.
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
        Test that saving the matrix to a JSON file and loading it restores all nodes and their consciousness levels accurately.
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
        Verifies that connecting two nodes stores the connection with the specified strength and that the connection can be retrieved accurately.
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
        Test that attempting to connect two nodes that do not exist in the matrix raises an InvalidStateException.
        """
        with self.assertRaises(InvalidStateException):
            self.matrix.connect_nodes("nonexistent1", "nonexistent2", strength=0.5)
            
    def test_consciousness_emergence_detection(self):
        """
        Test that consciousness emergence is detected when multiple nodes in the matrix have high consciousness levels.
        """
        # Add nodes with high consciousness levels
        for i in range(5):
            node = MatrixNode(id=f"high_node_{i}", consciousness_level=0.9)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertTrue(emergence_detected)
        
    def test_consciousness_emergence_detection_insufficient(self):
        """
        Test that consciousness emergence is not detected when all nodes have consciousness levels below the emergence threshold.
        """
        # Add nodes with low consciousness levels
        for i in range(2):
            node = MatrixNode(id=f"low_node_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertFalse(emergence_detected)
        
    def test_matrix_metrics_calculation(self):
        """
        Verify that the matrix calculates and returns performance metrics, including average consciousness, node count, and connection density, after nodes are added.
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
        Verify that a matrix evolution step with 100 nodes completes in less than one second.
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
        Verify that the matrix accurately tracks node membership by increasing the node count when nodes are added and decreasing it when nodes are removed.
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
        Verifies that attempting to deserialize corrupted JSON data raises a MatrixException.
        """
        corrupted_json = '{"nodes": {"invalid": "data"}, "state":'
        
        with self.assertRaises(MatrixException):
            GenesisConsciousnessMatrix.from_json(corrupted_json)
            
    def test_matrix_thread_safety(self):
        """
        Test that adding nodes concurrently from multiple threads succeeds, confirming thread safety of the matrix's node addition operation.
        """
        import threading
        import time
        
        results = []
        
        def add_nodes_thread(thread_id):
            """
            Add ten `MatrixNode` instances with unique IDs for the specified thread, recording whether each addition succeeds in a shared results list.
            
            Parameters:
                thread_id (int): The thread identifier used to generate unique node IDs.
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
        Tests that consciousness state enumeration values are ordered correctly from DORMANT to TRANSCENDENT.
        """
        self.assertLess(ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE)
        self.assertLess(ConsciousnessState.ACTIVE, ConsciousnessState.AWARE)
        self.assertLess(ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT)
        
    def test_consciousness_state_string_representation(self):
        """
        Checks that each ConsciousnessState enum value's string representation matches its name.
        """
        self.assertEqual(str(ConsciousnessState.DORMANT), "DORMANT")
        self.assertEqual(str(ConsciousnessState.ACTIVE), "ACTIVE")
        self.assertEqual(str(ConsciousnessState.AWARE), "AWARE")
        self.assertEqual(str(ConsciousnessState.TRANSCENDENT), "TRANSCENDENT")


class TestMatrixNode(unittest.TestCase):
    """Test cases for MatrixNode class."""
    
    def setUp(self):
        """
        Initializes a MatrixNode with a predefined ID and consciousness level before each test.
        """
        self.node = MatrixNode(id="test_node", consciousness_level=0.5)
        
    def test_node_initialization(self):
        """
        Test that a MatrixNode is created with the specified ID and consciousness level.
        """
        node = MatrixNode(id="init_test", consciousness_level=0.7)
        self.assertEqual(node.id, "init_test")
        self.assertEqual(node.consciousness_level, 0.7)
        
    def test_node_initialization_invalid_consciousness_level(self):
        """
        Verify that initializing a MatrixNode with a consciousness level outside the range [0.0, 1.0] raises a ValueError.
        """
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=1.5)
            
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=-0.1)
            
    def test_node_consciousness_level_update(self):
        """
        Verify that updating a node's consciousness level to a valid value correctly updates its state.
        """
        self.node.update_consciousness_level(0.8)
        self.assertEqual(self.node.consciousness_level, 0.8)
        
    def test_node_consciousness_level_update_invalid(self):
        """
        Test that updating a node's consciousness level to an invalid value raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.node.update_consciousness_level(1.2)
            
    def test_node_equality(self):
        """
        Verify that MatrixNode instances with the same ID and consciousness level are equal, and instances with different IDs are not equal.
        """
        node1 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node2 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node3 = MatrixNode(id="different_test", consciousness_level=0.5)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        
    def test_node_hash(self):
        """
        Verify that MatrixNode instances with the same ID produce identical hash values.
        
        Ensures that nodes with the same ID are treated equivalently in hash-based collections, regardless of their consciousness levels.
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
        Tests that custom matrix exceptions inherit from the correct base exception classes.
        """
        self.assertTrue(issubclass(MatrixException, Exception))
        self.assertTrue(issubclass(InvalidStateException, MatrixException))
        self.assertTrue(issubclass(MatrixInitializationError, MatrixException))
        
    def test_matrix_exception_messages(self):
        """
        Verify that custom matrix exceptions produce the expected error messages when raised and converted to strings.
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
        Set up a fresh GenesisConsciousnessMatrix instance before each integration test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_consciousness_evolution_cycle(self):
        """
        Simulates a full evolution cycle by adding nodes, connecting them, evolving the matrix until convergence, and verifying that the overall consciousness level changes.
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
        Verifies that consciousness emergence is only detected after all nodes' consciousness levels are raised above the emergence threshold.
        
        This test first adds nodes with low consciousness levels and confirms that emergence is not detected. It then increases all node levels above the threshold and checks that emergence is detected.
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
        Tests that serializing and deserializing the matrix preserves all nodes, their consciousness levels, and node connections, ensuring data integrity after persistence operations.
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
        """Set up test environment with comprehensive configuration."""
        self.matrix = GenesisConsciousnessMatrix()
        self.advanced_config = {
            'dimension': 1024,
            'consciousness_threshold': 0.95,
            'learning_rate': 0.0001,
            'max_iterations': 10000,
            'convergence_epsilon': 1e-8,
            'emergence_threshold': 0.85,
            'max_nodes': 1000
        }
        
    def test_matrix_initialization_edge_cases(self):
        """Test matrix initialization with edge case configurations."""
        # Test with minimum viable configuration
        min_config = {
            'dimension': 1,
            'consciousness_threshold': 0.0,
            'learning_rate': 1e-10,
            'max_iterations': 1
        }
        matrix = GenesisConsciousnessMatrix(config=min_config)
        self.assertEqual(matrix.dimension, 1)
        
        # Test with maximum configuration
        max_config = {
            'dimension': 65536,
            'consciousness_threshold': 1.0,
            'learning_rate': 1.0,
            'max_iterations': 1000000
        }
        matrix = GenesisConsciousnessMatrix(config=max_config)
        self.assertEqual(matrix.dimension, 65536)
        
    def test_matrix_with_zero_dimension(self):
        """Test matrix behavior with zero dimension configuration."""
        zero_config = {'dimension': 0}
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=zero_config)
            
    def test_matrix_with_nan_values(self):
        """Test matrix handling of NaN values in configuration."""
        nan_config = {
            'dimension': 256,
            'consciousness_threshold': float('nan'),
            'learning_rate': 0.001
        }
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=nan_config)
            
    def test_matrix_with_infinite_values(self):
        """Test matrix handling of infinite values in configuration."""
        inf_config = {
            'dimension': 256,
            'consciousness_threshold': float('inf'),
            'learning_rate': 0.001
        }
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=inf_config)
            
    def test_node_addition_capacity_limits(self):
        """Test matrix behavior when adding nodes beyond capacity."""
        # Configure matrix with low capacity
        limited_config = {'max_nodes': 5}
        matrix = GenesisConsciousnessMatrix(config=limited_config)
        
        # Add nodes up to capacity
        for i in range(5):
            node = MatrixNode(id=f"capacity_node_{i}", consciousness_level=0.5)
            result = matrix.add_node(node)
            self.assertTrue(result)
            
        # Try to add beyond capacity
        overflow_node = MatrixNode(id="overflow_node", consciousness_level=0.5)
        with self.assertRaises(InvalidStateException):
            matrix.add_node(overflow_node)
            
    def test_consciousness_level_precision(self):
        """Test consciousness level calculations with high precision requirements."""
        # Add nodes with very precise consciousness levels
        precise_levels = [0.123456789, 0.987654321, 0.555555555]
        for i, level in enumerate(precise_levels):
            node = MatrixNode(id=f"precise_node_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            
        calculated_level = self.matrix.calculate_consciousness_level()
        expected_level = sum(precise_levels) / len(precise_levels)
        self.assertAlmostEqual(calculated_level, expected_level, places=8)
        
    def test_state_transition_rapid_cycling(self):
        """Test rapid state transitions to verify stability."""
        states = [
            ConsciousnessState.DORMANT,
            ConsciousnessState.ACTIVE,
            ConsciousnessState.AWARE,
            ConsciousnessState.ACTIVE,
            ConsciousnessState.DORMANT
        ]
        
        for i in range(len(states) - 1):
            result = self.matrix.transition_state(states[i], states[i + 1])
            self.assertTrue(result)
            self.assertEqual(self.matrix.current_state, states[i + 1])
            
    def test_evolution_convergence_timeout(self):
        """Test evolution behavior when convergence times out."""
        # Add nodes that create oscillating behavior
        for i in range(10):
            node = MatrixNode(id=f"oscillate_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Set very low max iterations to force timeout
        start_time = datetime.now()
        self.matrix.evolve_until_convergence(max_iterations=3)
        end_time = datetime.now()
        
        # Should not have converged
        self.assertFalse(self.matrix.has_converged())
        
        # Should have completed quickly
        execution_time = (end_time - start_time).total_seconds()
        self.assertLess(execution_time, 0.1)
        
    def test_node_connection_strength_boundaries(self):
        """Test node connections with boundary strength values."""
        node1 = MatrixNode(id="boundary_node1", consciousness_level=0.5)
        node2 = MatrixNode(id="boundary_node2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Test minimum strength
        self.matrix.connect_nodes("boundary_node1", "boundary_node2", strength=0.0)
        connections = self.matrix.get_node_connections("boundary_node1")
        self.assertEqual(connections["boundary_node2"], 0.0)
        
        # Test maximum strength
        self.matrix.connect_nodes("boundary_node1", "boundary_node2", strength=1.0)
        connections = self.matrix.get_node_connections("boundary_node1")
        self.assertEqual(connections["boundary_node2"], 1.0)
        
    def test_node_connection_invalid_strength(self):
        """Test node connections with invalid strength values."""
        node1 = MatrixNode(id="invalid_node1", consciousness_level=0.5)
        node2 = MatrixNode(id="invalid_node2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Test negative strength
        with self.assertRaises(ValueError):
            self.matrix.connect_nodes("invalid_node1", "invalid_node2", strength=-0.1)
            
        # Test strength > 1.0
        with self.assertRaises(ValueError):
            self.matrix.connect_nodes("invalid_node1", "invalid_node2", strength=1.1)
            
    def test_consciousness_emergence_partial_threshold(self):
        """Test consciousness emergence detection with partial threshold satisfaction."""
        # Add nodes where some meet threshold and some don't
        high_nodes = 3
        low_nodes = 7
        
        for i in range(high_nodes):
            node = MatrixNode(id=f"high_emerge_node_{i}", consciousness_level=0.95)
            self.matrix.add_node(node)
            
        for i in range(low_nodes):
            node = MatrixNode(id=f"low_emerge_node_{i}", consciousness_level=0.2)
            self.matrix.add_node(node)
            
        # Should not detect emergence if only partial nodes meet threshold
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertFalse(emergence_detected)
        
    def test_metrics_calculation_empty_connections(self):
        """Test metrics calculation when no connections exist."""
        # Add nodes without connections
        for i in range(5):
            node = MatrixNode(id=f"isolated_node_{i}", consciousness_level=0.6)
            self.matrix.add_node(node)
            
        metrics = self.matrix.calculate_metrics()
        self.assertEqual(metrics["connection_density"], 0.0)
        self.assertEqual(metrics["node_count"], 5)
        self.assertAlmostEqual(metrics["average_consciousness"], 0.6, places=2)
        
    def test_serialization_with_special_characters(self):
        """Test serialization with node IDs containing special characters."""
        special_ids = [
            "node_with_unicode_🧠",
            "node-with-hyphens",
            "node.with.dots",
            "node with spaces",
            "node_with_\"quotes\"",
            "node_with_'apostrophes'"
        ]
        
        for node_id in special_ids:
            node = MatrixNode(id=node_id, consciousness_level=0.5)
            self.matrix.add_node(node)
            
        serialized = self.matrix.to_json()
        parsed = json.loads(serialized)
        
        # Verify all special IDs are preserved
        for node_id in special_ids:
            self.assertIn(node_id, parsed["nodes"])
            
    def test_deserialization_malformed_json(self):
        """Test deserialization with various malformed JSON inputs."""
        malformed_inputs = [
            '{"nodes": null, "state": "ACTIVE"}',
            '{"nodes": [], "state": "INVALID_STATE"}',
            '{"nodes": {"node1": {"consciousness_level": "invalid"}}, "state": "ACTIVE"}',
            '{"nodes": {"node1": {"consciousness_level": 2.0}}, "state": "ACTIVE"}',
            '{"incomplete": "data"}',
            '{"nodes": {"node1": null}, "state": "ACTIVE"}'
        ]
        
        for malformed_json in malformed_inputs:
            with self.assertRaises(MatrixException):
                GenesisConsciousnessMatrix.from_json(malformed_json)
                
    def test_file_operations_permissions(self):
        """Test file operations with permission issues."""
        import stat
        
        # Create a read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_file = os.path.join(temp_dir, "readonly.json")
            
            # Create file and make it read-only
            with open(readonly_file, 'w') as f:
                f.write('{"nodes": {}, "state": "DORMANT"}')
            os.chmod(readonly_file, stat.S_IRUSR)
            
            # Try to save to read-only file
            node = MatrixNode(id="perm_test", consciousness_level=0.5)
            self.matrix.add_node(node)
            
            with self.assertRaises(PermissionError):
                self.matrix.save_to_file(readonly_file)
                
    def test_concurrent_evolution_operations(self):
        """Test concurrent evolution operations for race conditions."""
        import threading
        import time
        
        # Add initial nodes
        for i in range(20):
            node = MatrixNode(id=f"concurrent_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        results = []
        
        def evolution_thread():
            """Run evolution steps concurrently."""
            try:
                for _ in range(10):
                    self.matrix.evolve_step()
                    time.sleep(0.001)
                results.append(True)
            except Exception as e:
                results.append(False)
                
        # Start multiple evolution threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=evolution_thread)
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # All threads should complete successfully
        self.assertTrue(all(results))
        
    def test_memory_efficiency_large_scale(self):
        """Test memory efficiency with large number of nodes."""
        import sys
        
        # Add many nodes and measure memory usage pattern
        initial_node_count = 1000
        
        for i in range(initial_node_count):
            node = MatrixNode(id=f"memory_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Remove half the nodes
        for i in range(initial_node_count // 2):
            self.matrix.remove_node(f"memory_node_{i}")
            
        # Verify correct count
        self.assertEqual(len(self.matrix.nodes), initial_node_count // 2)
        
        # Add nodes back
        for i in range(initial_node_count // 2):
            node = MatrixNode(id=f"memory_node_new_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Should be back to original count
        self.assertEqual(len(self.matrix.nodes), initial_node_count)
        
    def test_evolution_step_determinism(self):
        """Test that evolution steps are deterministic given same initial state."""
        # Create two identical matrices
        matrix1 = GenesisConsciousnessMatrix()
        matrix2 = GenesisConsciousnessMatrix()
        
        # Add identical nodes
        for i in range(5):
            node1 = MatrixNode(id=f"det_node_{i}", consciousness_level=0.3 + i * 0.1)
            node2 = MatrixNode(id=f"det_node_{i}", consciousness_level=0.3 + i * 0.1)
            matrix1.add_node(node1)
            matrix2.add_node(node2)
            
        # Evolution should produce identical results
        matrix1.evolve_step()
        matrix2.evolve_step()
        
        # Compare states
        state1 = matrix1.get_state_snapshot()
        state2 = matrix2.get_state_snapshot()
        self.assertEqual(state1, state2)
        
    def test_node_update_during_evolution(self):
        """Test behavior when nodes are updated during evolution."""
        # Add initial nodes
        for i in range(5):
            node = MatrixNode(id=f"update_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Start evolution in background
        def evolution_background():
            self.matrix.evolve_until_convergence(max_iterations=100)
            
        evolution_thread = threading.Thread(target=evolution_background)
        evolution_thread.start()
        
        # Update nodes during evolution
        time.sleep(0.01)  # Let evolution start
        for node_id in list(self.matrix.nodes.keys()):
            node = self.matrix.nodes[node_id]
            node.update_consciousness_level(0.8)
            
        evolution_thread.join()
        
        # Verify final state is stable
        final_level = self.matrix.calculate_consciousness_level()
        self.assertGreaterEqual(final_level, 0.0)
        self.assertLessEqual(final_level, 1.0)


class TestAsyncMatrixOperations(unittest.TestCase):
    """Test asynchronous operations if supported."""
    
    def setUp(self):
        """Set up async test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_async_evolution_simulation(self):
        """Test async evolution simulation if supported."""
        async def async_evolution():
            # Add nodes
            for i in range(10):
                node = MatrixNode(id=f"async_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Simulate async evolution
            await asyncio.sleep(0.001)
            self.matrix.evolve_step()
            
            return self.matrix.calculate_consciousness_level()
            
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(async_evolution())
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
        finally:
            loop.close()
            
    def test_async_concurrent_operations(self):
        """Test concurrent async operations."""
        async def add_nodes_async(start_idx, count):
            for i in range(count):
                node = MatrixNode(id=f"async_concurrent_node_{start_idx + i}", consciousness_level=0.5)
                await asyncio.sleep(0.001)  # Simulate async operation
                self.matrix.add_node(node)
                
        async def run_concurrent_additions():
            # Run multiple concurrent node additions
            await asyncio.gather(
                add_nodes_async(0, 10),
                add_nodes_async(10, 10),
                add_nodes_async(20, 10)
            )
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_concurrent_additions())
            self.assertEqual(len(self.matrix.nodes), 30)
        finally:
            loop.close()


class TestMatrixPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for matrix operations."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_addition_performance_scaling(self):
        """Test node addition performance with increasing scale."""
        scales = [100, 500, 1000, 2000]
        
        for scale in scales:
            matrix = GenesisConsciousnessMatrix()
            
            start_time = datetime.now()
            for i in range(scale):
                node = MatrixNode(id=f"perf_node_{i}", consciousness_level=0.5)
                matrix.add_node(node)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Performance should scale reasonably (not exponentially)
            time_per_node = execution_time / scale
            self.assertLess(time_per_node, 0.001)  # Less than 1ms per node
            
    def test_consciousness_calculation_performance(self):
        """Test consciousness level calculation performance."""
        # Add many nodes
        for i in range(5000):
            node = MatrixNode(id=f"calc_perf_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Measure calculation time
        start_time = datetime.now()
        for _ in range(100):  # Run multiple times
            self.matrix.calculate_consciousness_level()
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        avg_time = execution_time / 100
        
        # Should complete quickly even with many nodes
        self.assertLess(avg_time, 0.01)  # Less than 10ms per calculation
        
    def test_evolution_performance_stability(self):
        """Test evolution performance remains stable over time."""
        # Add nodes
        for i in range(100):
            node = MatrixNode(id=f"stability_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Measure evolution performance over multiple steps
        times = []
        for i in range(50):
            start_time = datetime.now()
            self.matrix.evolve_step()
            end_time = datetime.now()
            times.append((end_time - start_time).total_seconds())
            
        # Performance should be consistent
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Max time shouldn't be more than 2x average (stability check)
        self.assertLess(max_time, avg_time * 2)
        
    def test_serialization_performance_large_matrix(self):
        """Test serialization performance with large matrix."""
        # Create large matrix
        for i in range(1000):
            node = MatrixNode(id=f"serial_perf_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Add connections
        for i in range(999):
            self.matrix.connect_nodes(f"serial_perf_node_{i}", f"serial_perf_node_{i+1}", strength=0.5)
            
        # Measure serialization time
        start_time = datetime.now()
        serialized = self.matrix.to_json()
        end_time = datetime.now()
        
        serialization_time = (end_time - start_time).total_seconds()
        
        # Should serialize reasonably quickly
        self.assertLess(serialization_time, 1.0)
        
        # Verify serialization worked
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 1000)  # Should have substantial content


class TestMatrixRobustness(unittest.TestCase):
    """Robustness tests for matrix under stress conditions."""
    
    def setUp(self):
        """Set up robustness test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_under_memory_pressure(self):
        """Test matrix behavior under memory pressure."""
        # Create and destroy many matrices
        matrices = []
        
        for i in range(50):
            matrix = GenesisConsciousnessMatrix()
            
            # Add many nodes
            for j in range(100):
                node = MatrixNode(id=f"pressure_node_{i}_{j}", consciousness_level=0.5)
                matrix.add_node(node)
                
            matrices.append(matrix)
            
        # Verify all matrices are functional
        for matrix in matrices:
            self.assertGreaterEqual(len(matrix.nodes), 100)
            consciousness = matrix.calculate_consciousness_level()
            self.assertAlmostEqual(consciousness, 0.5, places=1)
            
    def test_matrix_recovery_from_corruption(self):
        """Test matrix recovery from simulated corruption."""
        # Create normal matrix
        for i in range(10):
            node = MatrixNode(id=f"recovery_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Simulate corruption by direct manipulation
        original_nodes = dict(self.matrix.nodes)
        
        # Corrupt and attempt recovery
        try:
            # Simulate corruption
            if hasattr(self.matrix, 'nodes'):
                self.matrix.nodes.clear()
                
            # Attempt recovery
            if hasattr(self.matrix, 'recover_from_backup'):
                self.matrix.recover_from_backup(original_nodes)
            else:
                # Manual recovery
                for node_id, node in original_nodes.items():
                    self.matrix.add_node(node)
                    
        except Exception:
            # Recovery failed, create new matrix
            self.matrix = GenesisConsciousnessMatrix()
            for node_id, node in original_nodes.items():
                self.matrix.add_node(node)
                
        # Verify recovery
        self.assertEqual(len(self.matrix.nodes), 10)
        
    def test_matrix_extreme_evolution_cycles(self):
        """Test matrix stability under extreme evolution cycles."""
        # Add nodes
        for i in range(20):
            node = MatrixNode(id=f"extreme_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Run many evolution cycles
        initial_consciousness = self.matrix.calculate_consciousness_level()
        
        for cycle in range(1000):
            self.matrix.evolve_step()
            
            # Verify matrix remains stable
            current_consciousness = self.matrix.calculate_consciousness_level()
            self.assertGreaterEqual(current_consciousness, 0.0)
            self.assertLessEqual(current_consciousness, 1.0)
            
            # Check for NaN or infinite values
            self.assertFalse(np.isnan(current_consciousness))
            self.assertFalse(np.isinf(current_consciousness))
            
    def test_matrix_resource_cleanup(self):
        """Test proper resource cleanup in matrix operations."""
        import gc
        import weakref
        
        # Create matrix with nodes
        matrix = GenesisConsciousnessMatrix()
        node_refs = []
        
        for i in range(100):
            node = MatrixNode(id=f"cleanup_node_{i}", consciousness_level=0.5)
            matrix.add_node(node)
            node_refs.append(weakref.ref(node))
            
        # Clear matrix
        matrix.reset()
        
        # Force garbage collection
        gc.collect()
        
        # Verify nodes are cleaned up
        alive_refs = [ref for ref in node_refs if ref() is not None]
        self.assertEqual(len(alive_refs), 0)


if __name__ == '__main__':
    # Run all tests including the new comprehensive ones
    unittest.main(verbosity=2, buffer=True)