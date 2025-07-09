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
    """Advanced test cases for additional edge cases and scenarios."""
    
    def setUp(self):
        """Set up advanced test configurations."""
        self.matrix = GenesisConsciousnessMatrix()
        self.extreme_config = {
            'dimension': 1024,
            'consciousness_threshold': 0.95,
            'learning_rate': 0.0001,
            'max_iterations': 10000,
            'emergence_threshold': 0.85,
            'connection_decay': 0.01
        }
    
    def test_matrix_initialization_edge_cases(self):
        """Test matrix initialization with edge case configurations."""
        # Test with minimal configuration
        minimal_config = {'dimension': 1, 'consciousness_threshold': 0.0}
        matrix = GenesisConsciousnessMatrix(config=minimal_config)
        self.assertEqual(matrix.dimension, 1)
        self.assertEqual(matrix.consciousness_threshold, 0.0)
        
        # Test with maximum configuration
        max_config = {'dimension': 10000, 'consciousness_threshold': 1.0}
        matrix = GenesisConsciousnessMatrix(config=max_config)
        self.assertEqual(matrix.dimension, 10000)
        self.assertEqual(matrix.consciousness_threshold, 1.0)
    
    def test_matrix_initialization_boundary_values(self):
        """Test matrix initialization with boundary values."""
        boundary_configs = [
            {'dimension': 1, 'consciousness_threshold': 0.0},
            {'dimension': 1, 'consciousness_threshold': 1.0},
            {'dimension': 2, 'consciousness_threshold': 0.5},
        ]
        
        for config in boundary_configs:
            with self.subTest(config=config):
                matrix = GenesisConsciousnessMatrix(config=config)
                self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
    
    def test_matrix_initialization_invalid_types(self):
        """Test matrix initialization with invalid parameter types."""
        invalid_configs = [
            {'dimension': 'invalid', 'consciousness_threshold': 0.5},
            {'dimension': 256, 'consciousness_threshold': 'invalid'},
            {'dimension': None, 'consciousness_threshold': 0.5},
            {'dimension': 256, 'consciousness_threshold': None},
        ]
        
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, MatrixInitializationError)):
                    GenesisConsciousnessMatrix(config=config)
    
    def test_add_node_with_none_values(self):
        """Test adding nodes with None values."""
        with self.assertRaises(ValueError):
            self.matrix.add_node(None)
    
    def test_add_node_with_invalid_node_structure(self):
        """Test adding nodes with invalid structure."""
        # Test with mock object that doesn't have required attributes
        invalid_node = Mock()
        invalid_node.id = None
        invalid_node.consciousness_level = 0.5
        
        with self.assertRaises(ValueError):
            self.matrix.add_node(invalid_node)
    
    def test_consciousness_level_precision(self):
        """Test consciousness level calculations with high precision values."""
        # Add nodes with precise decimal values
        precise_levels = [0.123456789, 0.987654321, 0.555555555]
        for i, level in enumerate(precise_levels):
            node = MatrixNode(id=f"precise_node_{i}", consciousness_level=level)
            self.matrix.add_node(node)
        
        calculated_level = self.matrix.calculate_consciousness_level()
        expected_level = sum(precise_levels) / len(precise_levels)
        self.assertAlmostEqual(calculated_level, expected_level, places=8)
    
    def test_consciousness_level_with_zero_values(self):
        """Test consciousness level calculation with zero values."""
        zero_nodes = [
            MatrixNode(id="zero_node_1", consciousness_level=0.0),
            MatrixNode(id="zero_node_2", consciousness_level=0.0),
            MatrixNode(id="zero_node_3", consciousness_level=0.0)
        ]
        
        for node in zero_nodes:
            self.matrix.add_node(node)
        
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.0)
    
    def test_consciousness_level_with_maximum_values(self):
        """Test consciousness level calculation with maximum values."""
        max_nodes = [
            MatrixNode(id="max_node_1", consciousness_level=1.0),
            MatrixNode(id="max_node_2", consciousness_level=1.0),
            MatrixNode(id="max_node_3", consciousness_level=1.0)
        ]
        
        for node in max_nodes:
            self.matrix.add_node(node)
        
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 1.0)
    
    def test_matrix_evolution_with_no_nodes(self):
        """Test matrix evolution with no nodes added."""
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        
        # State should remain unchanged or handle gracefully
        self.assertIsNotNone(final_state)
    
    def test_matrix_evolution_convergence_timeout(self):
        """Test matrix evolution convergence with timeout."""
        # Add nodes that might not converge quickly
        for i in range(20):
            node = MatrixNode(id=f"slow_node_{i}", consciousness_level=0.5 + (i % 2) * 0.1)
            self.matrix.add_node(node)
        
        # Test with very low max_iterations
        result = self.matrix.evolve_until_convergence(max_iterations=2)
        self.assertIsNotNone(result)
    
    def test_matrix_node_removal_edge_cases(self):
        """Test node removal edge cases."""
        # Test removing None
        result = self.matrix.remove_node(None)
        self.assertFalse(result)
        
        # Test removing empty string
        result = self.matrix.remove_node("")
        self.assertFalse(result)
        
        # Test removing with invalid type
        result = self.matrix.remove_node(123)
        self.assertFalse(result)
    
    def test_matrix_connections_with_same_node(self):
        """Test connecting a node to itself."""
        node = MatrixNode(id="self_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Self-connection should either be allowed or raise specific exception
        try:
            self.matrix.connect_nodes("self_node", "self_node", strength=0.8)
            # If allowed, verify connection exists
            connections = self.matrix.get_node_connections("self_node")
            self.assertIn("self_node", connections)
        except InvalidStateException:
            # If not allowed, that's also valid behavior
            pass
    
    def test_matrix_connections_with_invalid_strength(self):
        """Test node connections with invalid strength values."""
        node1 = MatrixNode(id="node1", consciousness_level=0.5)
        node2 = MatrixNode(id="node2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        invalid_strengths = [-0.1, 1.1, float('inf'), float('-inf'), float('nan')]
        
        for strength in invalid_strengths:
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    self.matrix.connect_nodes("node1", "node2", strength=strength)
    
    def test_matrix_serialization_edge_cases(self):
        """Test matrix serialization with edge cases."""
        # Test serialization with no nodes
        empty_serialized = self.matrix.to_json()
        self.assertIsInstance(empty_serialized, str)
        
        # Test serialization with extreme values
        extreme_node = MatrixNode(id="extreme_node", consciousness_level=1.0)
        self.matrix.add_node(extreme_node)
        
        serialized = self.matrix.to_json()
        parsed = json.loads(serialized)
        self.assertIn("nodes", parsed)
        self.assertIn("extreme_node", parsed["nodes"])
    
    def test_matrix_deserialization_malformed_json(self):
        """Test matrix deserialization with malformed JSON."""
        malformed_jsons = [
            '{"nodes": {}, "state": }',  # Missing value
            '{"nodes": {}, "state": "invalid"}',  # Invalid state
            '{"nodes": {"test": {"consciousness_level": "invalid"}}}',  # Invalid level
            '{"incomplete": "data"}',  # Missing required fields
        ]
        
        for malformed_json in malformed_jsons:
            with self.subTest(json_data=malformed_json):
                with self.assertRaises(MatrixException):
                    GenesisConsciousnessMatrix.from_json(malformed_json)
    
    def test_matrix_file_operations_permissions(self):
        """Test matrix file operations with permission issues."""
        import tempfile
        import os
        import stat
        
        # Create a temporary file and make it read-only
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name
        
        try:
            # Make file read-only
            os.chmod(temp_file, stat.S_IRUSR)
            
            # Try to save to read-only file
            with self.assertRaises(OSError):
                self.matrix.save_to_file(temp_file)
                
        finally:
            # Clean up - restore permissions and remove file
            os.chmod(temp_file, stat.S_IRUSR | stat.S_IWUSR)
            os.unlink(temp_file)
    
    def test_matrix_consciousness_emergence_edge_cases(self):
        """Test consciousness emergence detection edge cases."""
        # Test with exactly threshold values
        threshold_node = MatrixNode(id="threshold_node", consciousness_level=0.75)
        self.matrix.add_node(threshold_node)
        
        emergence = self.matrix.detect_consciousness_emergence()
        self.assertIsInstance(emergence, bool)
        
        # Test with mixed values around threshold
        mixed_levels = [0.74, 0.75, 0.76]
        for i, level in enumerate(mixed_levels):
            node = MatrixNode(id=f"mixed_node_{i}", consciousness_level=level)
            self.matrix.add_node(node)
        
        emergence = self.matrix.detect_consciousness_emergence()
        self.assertIsInstance(emergence, bool)
    
    def test_matrix_metrics_edge_cases(self):
        """Test matrix metrics calculation edge cases."""
        # Test metrics with empty matrix
        empty_metrics = self.matrix.calculate_metrics()
        self.assertIn("node_count", empty_metrics)
        self.assertEqual(empty_metrics["node_count"], 0)
        
        # Test metrics with single node
        single_node = MatrixNode(id="single_metrics", consciousness_level=0.5)
        self.matrix.add_node(single_node)
        
        single_metrics = self.matrix.calculate_metrics()
        self.assertEqual(single_metrics["node_count"], 1)
        self.assertEqual(single_metrics["average_consciousness"], 0.5)
    
    def test_matrix_large_scale_operations(self):
        """Test matrix operations with large numbers of nodes."""
        # Add 1000 nodes
        large_node_count = 1000
        for i in range(large_node_count):
            node = MatrixNode(id=f"large_node_{i}", consciousness_level=i / large_node_count)
            self.matrix.add_node(node)
        
        # Test operations still work correctly
        self.assertEqual(len(self.matrix.nodes), large_node_count)
        
        # Test metrics calculation
        metrics = self.matrix.calculate_metrics()
        self.assertEqual(metrics["node_count"], large_node_count)
        
        # Test consciousness level calculation
        consciousness = self.matrix.calculate_consciousness_level()
        self.assertIsInstance(consciousness, float)
        self.assertGreaterEqual(consciousness, 0.0)
        self.assertLessEqual(consciousness, 1.0)


class TestMatrixNodeAdvanced(unittest.TestCase):
    """Advanced test cases for MatrixNode class."""
    
    def test_node_consciousness_level_boundary_values(self):
        """Test node consciousness level with exact boundary values."""
        # Test exactly 0.0
        node_zero = MatrixNode(id="zero_node", consciousness_level=0.0)
        self.assertEqual(node_zero.consciousness_level, 0.0)
        
        # Test exactly 1.0
        node_one = MatrixNode(id="one_node", consciousness_level=1.0)
        self.assertEqual(node_one.consciousness_level, 1.0)
    
    def test_node_consciousness_level_precision(self):
        """Test node consciousness level with high precision values."""
        precise_level = 0.123456789012345
        node = MatrixNode(id="precise_node", consciousness_level=precise_level)
        self.assertEqual(node.consciousness_level, precise_level)
    
    def test_node_id_edge_cases(self):
        """Test node ID with edge cases."""
        # Test with empty string
        with self.assertRaises(ValueError):
            MatrixNode(id="", consciousness_level=0.5)
        
        # Test with very long ID
        long_id = "a" * 1000
        node = MatrixNode(id=long_id, consciousness_level=0.5)
        self.assertEqual(node.id, long_id)
        
        # Test with special characters
        special_id = "node_with_@#$%^&*()_special_chars"
        node = MatrixNode(id=special_id, consciousness_level=0.5)
        self.assertEqual(node.id, special_id)
    
    def test_node_update_consciousness_level_precision(self):
        """Test node consciousness level updates with precision."""
        node = MatrixNode(id="update_node", consciousness_level=0.5)
        
        # Update with high precision
        precise_level = 0.987654321098765
        node.update_consciousness_level(precise_level)
        self.assertEqual(node.consciousness_level, precise_level)
    
    def test_node_equality_edge_cases(self):
        """Test node equality with edge cases."""
        node1 = MatrixNode(id="test_node", consciousness_level=0.5)
        
        # Test equality with None
        self.assertNotEqual(node1, None)
        
        # Test equality with different type
        self.assertNotEqual(node1, "not_a_node")
        
        # Test equality with different consciousness level but same ID
        node2 = MatrixNode(id="test_node", consciousness_level=0.7)
        self.assertEqual(node1, node2)  # Should be equal based on ID only
    
    def test_node_hash_consistency(self):
        """Test node hash consistency across operations."""
        node = MatrixNode(id="hash_node", consciousness_level=0.5)
        initial_hash = hash(node)
        
        # Hash should remain consistent after consciousness level update
        node.update_consciousness_level(0.8)
        updated_hash = hash(node)
        self.assertEqual(initial_hash, updated_hash)
    
    def test_node_string_representation_edge_cases(self):
        """Test node string representation with edge cases."""
        # Test with extreme values
        node_zero = MatrixNode(id="zero_node", consciousness_level=0.0)
        str_repr = str(node_zero)
        self.assertIn("zero_node", str_repr)
        self.assertIn("0.0", str_repr)
        
        node_one = MatrixNode(id="one_node", consciousness_level=1.0)
        str_repr = str(node_one)
        self.assertIn("one_node", str_repr)
        self.assertIn("1.0", str_repr)


class TestMatrixAsyncOperations(unittest.TestCase):
    """Test cases for asynchronous matrix operations."""
    
    def setUp(self):
        """Set up async test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test environment."""
        self.loop.close()
    
    def test_async_matrix_evolution(self):
        """Test asynchronous matrix evolution."""
        async def async_evolve():
            # Add nodes
            for i in range(10):
                node = MatrixNode(id=f"async_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            
            # Simulate async evolution
            await asyncio.sleep(0.01)  # Simulate async operation
            self.matrix.evolve_step()
            return self.matrix.get_state_snapshot()
        
        # Run async test
        result = self.loop.run_until_complete(async_evolve())
        self.assertIsNotNone(result)
    
    def test_async_concurrent_node_operations(self):
        """Test concurrent node operations."""
        async def add_nodes_concurrently():
            tasks = []
            for i in range(50):
                async def add_node(node_id):
                    await asyncio.sleep(0.001)  # Simulate async delay
                    node = MatrixNode(id=f"concurrent_node_{node_id}", consciousness_level=0.5)
                    return self.matrix.add_node(node)
                
                tasks.append(add_node(i))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        results = self.loop.run_until_complete(add_nodes_concurrently())
        successful_adds = [r for r in results if r is True]
        self.assertGreater(len(successful_adds), 0)


class TestMatrixErrorHandlingAdvanced(unittest.TestCase):
    """Advanced error handling test cases."""
    
    def setUp(self):
        """Set up error handling test environment."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_matrix_memory_leak_prevention(self):
        """Test matrix memory leak prevention."""
        import gc
        
        # Create and destroy many nodes
        for cycle in range(10):
            for i in range(100):
                node = MatrixNode(id=f"leak_test_{cycle}_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            
            # Clear nodes
            self.matrix.reset()
            
            # Force garbage collection
            gc.collect()
        
        # Memory usage should be reasonable
        self.assertEqual(len(self.matrix.nodes), 0)
    
    def test_matrix_exception_chaining(self):
        """Test exception chaining in matrix operations."""
        try:
            # Simulate a chained exception scenario
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise MatrixException("Matrix error") from e
        except MatrixException as e:
            self.assertIsInstance(e.__cause__, ValueError)
            self.assertEqual(str(e.__cause__), "Original error")
    
    def test_matrix_resource_cleanup(self):
        """Test proper resource cleanup in matrix operations."""
        # Create matrix with resources
        node = MatrixNode(id="cleanup_test", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Simulate resource-intensive operation
        try:
            # Force cleanup
            if hasattr(self.matrix, 'cleanup'):
                self.matrix.cleanup()
        except Exception as e:
            self.fail(f"Cleanup should not raise exception: {e}")
    
    def test_matrix_state_recovery(self):
        """Test matrix state recovery after errors."""
        # Set up initial state
        node = MatrixNode(id="recovery_test", consciousness_level=0.5)
        self.matrix.add_node(node)
        initial_state = self.matrix.get_state_snapshot()
        
        # Simulate error during operation
        try:
            # Force an error condition
            self.matrix.nodes["recovery_test"].consciousness_level = "invalid"
            self.matrix.calculate_consciousness_level()
        except Exception:
            pass  # Expected to fail
        
        # Matrix should be able to recover
        node.consciousness_level = 0.7  # Fix the error
        recovered_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(recovered_level, 0.7)


class TestMatrixPerformanceAdvanced(unittest.TestCase):
    """Advanced performance test cases."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_matrix_scalability_stress(self):
        """Test matrix scalability under stress."""
        import time
        
        # Test with increasing node counts
        node_counts = [100, 500, 1000, 2000]
        performance_results = []
        
        for count in node_counts:
            # Reset matrix
            self.matrix.reset()
            
            # Add nodes and measure time
            start_time = time.time()
            for i in range(count):
                node = MatrixNode(id=f"stress_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            
            # Perform operations
            self.matrix.calculate_consciousness_level()
            self.matrix.calculate_metrics()
            
            end_time = time.time()
            performance_results.append((count, end_time - start_time))
        
        # Performance should scale reasonably
        for count, duration in performance_results:
            self.assertLess(duration, 5.0, f"Operation with {count} nodes took too long: {duration}s")
    
    def test_matrix_memory_efficiency(self):
        """Test matrix memory efficiency."""
        import sys
        
        # Measure memory usage
        initial_size = sys.getsizeof(self.matrix)
        
        # Add nodes
        for i in range(1000):
            node = MatrixNode(id=f"memory_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
        
        # Memory usage should be reasonable
        final_size = sys.getsizeof(self.matrix)
        memory_increase = final_size - initial_size
        
        # Should not use excessive memory per node
        self.assertLess(memory_increase / 1000, 1000, "Memory usage per node is too high")
    
    def test_matrix_operation_complexity(self):
        """Test complexity of matrix operations."""
        import time
        
        # Test with different sizes to verify complexity
        sizes = [100, 200, 400]
        times = []
        
        for size in sizes:
            self.matrix.reset()
            
            # Add nodes
            for i in range(size):
                node = MatrixNode(id=f"complexity_node_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            
            # Measure operation time
            start_time = time.time()
            self.matrix.calculate_consciousness_level()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Time complexity should be reasonable (not exponential)
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            self.assertLess(ratio, 10, f"Time complexity appears too high: {ratio}")


# Add pytest-style tests for compatibility
@pytest.mark.parametrize("consciousness_level", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_node_consciousness_levels_parametrized(consciousness_level):
    """Parametrized test for various consciousness levels."""
    node = MatrixNode(id="param_test", consciousness_level=consciousness_level)
    assert node.consciousness_level == consciousness_level

@pytest.mark.parametrize("dimension,threshold", [
    (1, 0.0),
    (256, 0.5),
    (1024, 0.95),
    (2048, 1.0)
])
def test_matrix_initialization_parametrized(dimension, threshold):
    """Parametrized test for matrix initialization."""
    config = {'dimension': dimension, 'consciousness_threshold': threshold}
    matrix = GenesisConsciousnessMatrix(config=config)
    assert matrix.dimension == dimension
    assert matrix.consciousness_threshold == threshold

@pytest.mark.asyncio
async def test_async_matrix_operations():
    """Async test for matrix operations."""
    matrix = GenesisConsciousnessMatrix()
    
    # Add nodes asynchronously
    for i in range(10):
        await asyncio.sleep(0.001)
        node = MatrixNode(id=f"async_test_{i}", consciousness_level=0.5)
        matrix.add_node(node)
    
    assert len(matrix.nodes) == 10

if __name__ == '__main__':
    # Run both unittest and pytest
    unittest.main(verbosity=2, exit=False)
    pytest.main([__file__, '-v'])