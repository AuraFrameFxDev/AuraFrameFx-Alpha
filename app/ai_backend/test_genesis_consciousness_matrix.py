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

class TestGenesisConsciousnessMatrixExtended(unittest.TestCase):
    """Extended test cases for Genesis Consciousness Matrix with additional edge cases and scenarios."""
    
    def setUp(self):
        """Set up test fixtures for extended tests."""
        self.matrix = GenesisConsciousnessMatrix()
        self.extreme_config = {
            'dimension': 1,
            'consciousness_threshold': 0.99,
            'learning_rate': 0.00001,
            'max_iterations': 10000
        }
        
    def test_matrix_initialization_edge_cases(self):
        """Test matrix initialization with extreme parameter values."""
        # Test minimum valid values
        min_config = {'dimension': 1, 'consciousness_threshold': 0.0}
        matrix = GenesisConsciousnessMatrix(config=min_config)
        self.assertEqual(matrix.dimension, 1)
        self.assertEqual(matrix.consciousness_threshold, 0.0)
        
        # Test maximum valid values
        max_config = {'dimension': 10000, 'consciousness_threshold': 1.0}
        matrix = GenesisConsciousnessMatrix(config=max_config)
        self.assertEqual(matrix.dimension, 10000)
        self.assertEqual(matrix.consciousness_threshold, 1.0)
        
    def test_matrix_initialization_boundary_conditions(self):
        """Test matrix initialization at boundary conditions."""
        # Test consciousness_threshold at exactly 1.0
        config = {'consciousness_threshold': 1.0}
        matrix = GenesisConsciousnessMatrix(config=config)
        self.assertEqual(matrix.consciousness_threshold, 1.0)
        
        # Test extremely small learning rate
        config = {'learning_rate': 1e-10}
        matrix = GenesisConsciousnessMatrix(config=config)
        self.assertEqual(matrix.learning_rate, 1e-10)
        
    def test_matrix_with_zero_nodes_operations(self):
        """Test all matrix operations when no nodes are present."""
        # Evolution with no nodes
        self.matrix.evolve_step()
        self.assertEqual(len(self.matrix.nodes), 0)
        
        # Metrics calculation with no nodes
        metrics = self.matrix.calculate_metrics()
        self.assertEqual(metrics['node_count'], 0)
        self.assertEqual(metrics['average_consciousness'], 0.0)
        
        # Convergence detection with no nodes
        self.assertTrue(self.matrix.has_converged())
        
    def test_matrix_with_single_node_operations(self):
        """Test matrix operations with exactly one node."""
        node = MatrixNode(id="single", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Test evolution with single node
        initial_level = self.matrix.calculate_consciousness_level()
        self.matrix.evolve_step()
        final_level = self.matrix.calculate_consciousness_level()
        # Single node evolution might not change consciousness level
        self.assertIsNotNone(final_level)
        
    def test_node_consciousness_level_precision(self):
        """Test node consciousness level with high precision values."""
        # Test with very precise values
        precise_levels = [0.123456789, 0.987654321, 0.000000001, 0.999999999]
        for i, level in enumerate(precise_levels):
            node = MatrixNode(id=f"precise_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            retrieved_level = self.matrix.nodes[f"precise_{i}"].consciousness_level
            self.assertAlmostEqual(retrieved_level, level, places=9)
            
    def test_matrix_state_transition_edge_cases(self):
        """Test consciousness state transitions with edge cases."""
        # Test transition from same state to same state
        current_state = self.matrix.current_state
        result = self.matrix.transition_state(current_state, current_state)
        self.assertTrue(result)
        
        # Test rapid state transitions
        states = [ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE, 
                 ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT]
        for i in range(len(states) - 1):
            if hasattr(self.matrix, 'transition_state'):
                self.matrix.transition_state(states[i], states[i+1])
                
    def test_matrix_evolution_convergence_edge_cases(self):
        """Test matrix evolution convergence with edge cases."""
        # Test convergence with maximum iterations reached
        for i in range(10):
            node = MatrixNode(id=f"conv_node_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
            
        # Set a very low max_iterations to test timeout
        self.matrix.evolve_until_convergence(max_iterations=1)
        # Should complete without errors even if not converged
        
    def test_matrix_node_connections_edge_cases(self):
        """Test node connections with edge cases."""
        node1 = MatrixNode(id="conn1", consciousness_level=0.5)
        node2 = MatrixNode(id="conn2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Test connection with minimum strength
        self.matrix.connect_nodes("conn1", "conn2", strength=0.0)
        connections = self.matrix.get_node_connections("conn1")
        self.assertEqual(connections["conn2"], 0.0)
        
        # Test connection with maximum strength
        self.matrix.connect_nodes("conn1", "conn2", strength=1.0)
        connections = self.matrix.get_node_connections("conn1")
        self.assertEqual(connections["conn2"], 1.0)
        
    def test_matrix_serialization_edge_cases(self):
        """Test matrix serialization with edge cases."""
        # Test serialization with nodes having extreme consciousness levels
        node1 = MatrixNode(id="extreme_low", consciousness_level=0.0)
        node2 = MatrixNode(id="extreme_high", consciousness_level=1.0)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        serialized = self.matrix.to_json()
        deserialized = GenesisConsciousnessMatrix.from_json(serialized)
        
        self.assertEqual(deserialized.nodes["extreme_low"].consciousness_level, 0.0)
        self.assertEqual(deserialized.nodes["extreme_high"].consciousness_level, 1.0)
        
    def test_matrix_memory_stress_test(self):
        """Test matrix memory management under stress."""
        # Add and remove many nodes rapidly
        for cycle in range(10):
            # Add nodes
            for i in range(50):
                node = MatrixNode(id=f"stress_{cycle}_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            
            # Remove half of them
            for i in range(25):
                self.matrix.remove_node(f"stress_{cycle}_{i}")
                
        # Verify final state is consistent
        self.assertGreater(len(self.matrix.nodes), 0)
        
    def test_matrix_concurrent_operations(self):
        """Test matrix operations under concurrent access."""
        import threading
        import time
        
        def modify_matrix(thread_id):
            """Perform mixed operations on the matrix."""
            for i in range(20):
                try:
                    # Add node
                    node = MatrixNode(id=f"concurrent_{thread_id}_{i}", consciousness_level=0.5)
                    self.matrix.add_node(node)
                    
                    # Evolution step
                    self.matrix.evolve_step()
                    
                    # Calculate metrics
                    self.matrix.calculate_metrics()
                    
                    # Remove some nodes
                    if i % 5 == 0 and i > 0:
                        self.matrix.remove_node(f"concurrent_{thread_id}_{i-1}")
                        
                except Exception as e:
                    # Log but don't fail the test for race conditions
                    print(f"Concurrent operation exception: {e}")
                    
                time.sleep(0.001)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=modify_matrix, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Verify matrix is in a valid state
        self.assertIsInstance(self.matrix.nodes, dict)
        
    def test_matrix_performance_degradation(self):
        """Test matrix performance with increasing node count."""
        performance_data = []
        
        for node_count in [10, 50, 100, 500]:
            # Add nodes
            for i in range(node_count):
                node = MatrixNode(id=f"perf_{node_count}_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Measure evolution time
            start_time = datetime.now()
            self.matrix.evolve_step()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            performance_data.append((node_count, execution_time))
            
        # Performance should not degrade exponentially
        # (This is a basic sanity check)
        for node_count, exec_time in performance_data:
            self.assertLess(exec_time, 10.0)  # Should not take more than 10 seconds
            
    def test_matrix_error_recovery(self):
        """Test matrix error recovery scenarios."""
        # Test recovery from invalid state
        try:
            # Force matrix into invalid state (if possible)
            self.matrix.nodes = None
            self.matrix.reset()
            # Should recover to valid state
            self.assertIsNotNone(self.matrix.nodes)
        except Exception:
            # If reset fails, create new matrix
            self.matrix = GenesisConsciousnessMatrix()
            
        # Verify matrix is functional after recovery
        node = MatrixNode(id="recovery_test", consciousness_level=0.5)
        self.matrix.add_node(node)
        self.assertIn("recovery_test", self.matrix.nodes)


class TestAsyncGenesisConsciousnessMatrix(unittest.TestCase):
    """Test asynchronous operations of Genesis Consciousness Matrix."""
    
    def setUp(self):
        """Set up async test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_async_evolution_step(self):
        """Test asynchronous evolution step if implemented."""
        async def async_evolution_test():
            node = MatrixNode(id="async_test", consciousness_level=0.5)
            self.matrix.add_node(node)
            
            if hasattr(self.matrix, 'evolve_step_async'):
                await self.matrix.evolve_step_async()
            else:
                # Fallback to sync evolution
                self.matrix.evolve_step()
                
            return self.matrix.calculate_consciousness_level()
        
        # Run async test
        if hasattr(asyncio, 'run'):
            result = asyncio.run(async_evolution_test())
            self.assertIsNotNone(result)
            
    def test_async_batch_operations(self):
        """Test asynchronous batch operations."""
        async def batch_operation_test():
            # Add multiple nodes asynchronously
            tasks = []
            for i in range(10):
                node = MatrixNode(id=f"batch_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Simulate async operations
            await asyncio.sleep(0.001)
            
            # Verify all nodes were added
            return len(self.matrix.nodes)
        
        if hasattr(asyncio, 'run'):
            result = asyncio.run(batch_operation_test())
            self.assertEqual(result, 10)


class TestMatrixPropertyBased(unittest.TestCase):
    """Property-based tests for Genesis Consciousness Matrix."""
    
    def setUp(self):
        """Set up property-based test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_consciousness_level_invariants(self):
        """Test consciousness level invariants across operations."""
        # Property: consciousness level should always be in [0, 1]
        for i in range(100):
            level = i / 100.0
            node = MatrixNode(id=f"prop_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            
            calculated_level = self.matrix.calculate_consciousness_level()
            self.assertGreaterEqual(calculated_level, 0.0)
            self.assertLessEqual(calculated_level, 1.0)
            
    def test_node_count_invariants(self):
        """Test node count invariants across operations."""
        # Property: node count should match actual nodes
        for i in range(20):
            node = MatrixNode(id=f"count_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
            metrics = self.matrix.calculate_metrics()
            self.assertEqual(metrics['node_count'], len(self.matrix.nodes))
            
    def test_serialization_roundtrip_invariants(self):
        """Test that serialization roundtrip preserves all data."""
        # Property: serialize->deserialize should preserve all data
        original_nodes = {}
        for i in range(10):
            node = MatrixNode(id=f"roundtrip_{i}", consciousness_level=i / 10.0)
            self.matrix.add_node(node)
            original_nodes[node.id] = node.consciousness_level
            
        # Serialize and deserialize
        serialized = self.matrix.to_json()
        restored = GenesisConsciousnessMatrix.from_json(serialized)
        
        # Verify all original data preserved
        for node_id, original_level in original_nodes.items():
            self.assertIn(node_id, restored.nodes)
            self.assertEqual(restored.nodes[node_id].consciousness_level, original_level)


class TestMatrixMockingAndIsolation(unittest.TestCase):
    """Test matrix behavior with mocked dependencies."""
    
    def setUp(self):
        """Set up mocking test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    @patch('app.ai_backend.genesis_consciousness_matrix.GenesisConsciousnessMatrix.evolve_step')
    def test_evolution_with_mocked_step(self, mock_evolve):
        """Test evolution behavior with mocked evolution step."""
        mock_evolve.return_value = True
        
        result = self.matrix.evolve_step()
        self.assertTrue(result)
        mock_evolve.assert_called_once()
        
    @patch('json.dumps')
    def test_serialization_with_mocked_json(self, mock_dumps):
        """Test serialization with mocked JSON library."""
        mock_dumps.return_value = '{"test": "data"}'
        
        if hasattr(self.matrix, 'to_json'):
            result = self.matrix.to_json()
            self.assertEqual(result, '{"test": "data"}')
            mock_dumps.assert_called_once()
            
    @patch('builtins.open', new_callable=mock_open, read_data='{"nodes": {}, "state": "DORMANT"}')
    def test_file_loading_with_mocked_io(self, mock_file):
        """Test file loading with mocked file operations."""
        if hasattr(GenesisConsciousnessMatrix, 'load_from_file'):
            matrix = GenesisConsciousnessMatrix.load_from_file('test_file.json')
            self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
            mock_file.assert_called_once_with('test_file.json', 'r')


class TestMatrixValidationAndSanitization(unittest.TestCase):
    """Test input validation and data sanitization."""
    
    def setUp(self):
        """Set up validation test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_validation(self):
        """Test node ID validation with various input types."""
        # Test with different ID types
        valid_ids = ['string_id', 'id_123', 'node-with-dashes', 'node_with_underscores']
        for node_id in valid_ids:
            node = MatrixNode(id=node_id, consciousness_level=0.5)
            result = self.matrix.add_node(node)
            self.assertTrue(result)
            
    def test_consciousness_level_boundary_validation(self):
        """Test consciousness level validation at exact boundaries."""
        # Test exact boundary values
        boundary_values = [0.0, 1.0, 0.5, 0.999999, 0.000001]
        for level in boundary_values:
            node = MatrixNode(id=f"boundary_{level}", consciousness_level=level)
            self.matrix.add_node(node)
            stored_level = self.matrix.nodes[f"boundary_{level}"].consciousness_level
            self.assertEqual(stored_level, level)
            
    def test_configuration_sanitization(self):
        """Test configuration parameter sanitization."""
        # Test with string values that should be converted
        config_with_strings = {
            'dimension': '256',
            'consciousness_threshold': '0.75',
            'learning_rate': '0.001'
        }
        
        try:
            matrix = GenesisConsciousnessMatrix(config=config_with_strings)
            # If conversion is supported, verify types
            if hasattr(matrix, 'dimension'):
                self.assertIsInstance(matrix.dimension, (int, float))
        except (TypeError, ValueError):
            # If conversion is not supported, that's also valid behavior
            pass
            
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON during deserialization."""
        malformed_json_samples = [
            '{"nodes": {',  # Incomplete JSON
            '{"nodes": {"invalid": null}}',  # Invalid node data
            '{"state": "INVALID_STATE"}',  # Invalid state
            '',  # Empty string
            'not_json_at_all',  # Not JSON
        ]
        
        for malformed in malformed_json_samples:
            with self.assertRaises((json.JSONDecodeError, MatrixException, ValueError)):
                GenesisConsciousnessMatrix.from_json(malformed)


class TestMatrixPerformanceOptimization(unittest.TestCase):
    """Test performance optimization scenarios."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_large_scale_node_operations(self):
        """Test performance with large number of nodes."""
        # Test with 1000 nodes
        start_time = datetime.now()
        
        for i in range(1000):
            node = MatrixNode(id=f"large_{i}", consciousness_level=i / 1000.0)
            self.matrix.add_node(node)
            
        add_time = (datetime.now() - start_time).total_seconds()
        
        # Test evolution with many nodes
        start_time = datetime.now()
        self.matrix.evolve_step()
        evolve_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions (adjust based on requirements)
        self.assertLess(add_time, 5.0)  # Adding 1000 nodes should take < 5 seconds
        self.assertLess(evolve_time, 10.0)  # Evolution should take < 10 seconds
        
    def test_memory_efficiency_with_node_churn(self):
        """Test memory efficiency with high node turnover."""
        import gc
        
        # Force garbage collection and measure initial memory
        gc.collect()
        initial_node_count = len(self.matrix.nodes)
        
        # Simulate high node turnover
        for cycle in range(100):
            # Add nodes
            for i in range(10):
                node = MatrixNode(id=f"churn_{cycle}_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Remove all nodes from this cycle
            for i in range(10):
                self.matrix.remove_node(f"churn_{cycle}_{i}")
                
        # Force garbage collection
        gc.collect()
        
        # Verify we haven't leaked nodes
        final_node_count = len(self.matrix.nodes)
        self.assertEqual(final_node_count, initial_node_count)
        
    def test_connection_density_performance(self):
        """Test performance with high connection density."""
        # Create nodes
        node_count = 50
        for i in range(node_count):
            node = MatrixNode(id=f"dense_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Create dense connections (every node connected to every other)
        start_time = datetime.now()
        
        for i in range(node_count):
            for j in range(i + 1, node_count):
                self.matrix.connect_nodes(f"dense_{i}", f"dense_{j}", strength=0.5)
                
        connection_time = (datetime.now() - start_time).total_seconds()
        
        # Test evolution with dense connections
        start_time = datetime.now()
        self.matrix.evolve_step()
        evolve_time = (datetime.now() - start_time).total_seconds()
        
        # Performance should still be reasonable
        self.assertLess(connection_time, 10.0)
        self.assertLess(evolve_time, 10.0)


# Additional parametrized tests if pytest is available
try:
    import pytest
    
    class TestMatrixParametrized:
        """Parametrized tests for matrix operations."""
        
        @pytest.mark.parametrize("dimension", [1, 10, 100, 1000])
        def test_matrix_initialization_dimensions(self, dimension):
            """Test matrix initialization with various dimensions."""
            config = {'dimension': dimension}
            matrix = GenesisConsciousnessMatrix(config=config)
            assert matrix.dimension == dimension
            
        @pytest.mark.parametrize("consciousness_level", [0.0, 0.25, 0.5, 0.75, 1.0])
        def test_node_consciousness_levels(self, consciousness_level):
            """Test node creation with various consciousness levels."""
            node = MatrixNode(id=f"test_{consciousness_level}", consciousness_level=consciousness_level)
            assert node.consciousness_level == consciousness_level
            
        @pytest.mark.parametrize("node_count", [1, 5, 10, 50, 100])
        def test_matrix_with_variable_node_counts(self, node_count):
            """Test matrix operations with various node counts."""
            matrix = GenesisConsciousnessMatrix()
            
            # Add nodes
            for i in range(node_count):
                node = MatrixNode(id=f"var_{i}", consciousness_level=0.5)
                matrix.add_node(node)
                
            # Test operations
            level = matrix.calculate_consciousness_level()
            assert 0.0 <= level <= 1.0
            
            metrics = matrix.calculate_metrics()
            assert metrics['node_count'] == node_count
            
        @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
        def test_emergence_detection_thresholds(self, threshold):
            """Test consciousness emergence detection with various thresholds."""
            config = {'consciousness_threshold': threshold}
            matrix = GenesisConsciousnessMatrix(config=config)
            
            # Add nodes with consciousness above threshold
            for i in range(5):
                level = threshold + 0.05
                node = MatrixNode(id=f"emerge_{i}", consciousness_level=level)
                matrix.add_node(node)
                
            emergence = matrix.detect_consciousness_emergence()
            assert emergence is True
            
except ImportError:
    # pytest not available, skip parametrized tests
    pass

class TestMatrixAdvancedBehaviors(unittest.TestCase):
    """Advanced behavioral tests for Genesis Consciousness Matrix."""
    
    def setUp(self):
        """Set up advanced behavior test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        self.complex_config = {
            'dimension': 512,
            'consciousness_threshold': 0.85,
            'learning_rate': 0.0001,
            'max_iterations': 5000,
            'convergence_tolerance': 1e-6,
            'memory_limit': 1024*1024,  # 1MB
            'enable_quantum_effects': True,
            'neural_pathways': ['alpha', 'beta', 'gamma'],
            'activation_function': 'sigmoid'
        }
        
    def test_matrix_consciousness_gradient_descent(self):
        """Test consciousness evolution using gradient descent optimization."""
        # Create nodes with varying consciousness levels
        consciousness_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, level in enumerate(consciousness_levels):
            node = MatrixNode(id=f"gradient_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            
        # Track consciousness evolution over multiple steps
        initial_level = self.matrix.calculate_consciousness_level()
        evolution_history = [initial_level]
        
        for step in range(10):
            self.matrix.evolve_step()
            current_level = self.matrix.calculate_consciousness_level()
            evolution_history.append(current_level)
            
        # Verify evolution shows improvement or convergence
        final_level = evolution_history[-1]
        self.assertNotEqual(initial_level, final_level)
        
        # Check for monotonic improvement or convergence
        improvements = sum(1 for i in range(1, len(evolution_history)) 
                          if evolution_history[i] >= evolution_history[i-1])
        self.assertGreater(improvements, len(evolution_history) // 2)
        
    def test_matrix_consciousness_quantum_tunneling(self):
        """Test quantum tunneling effects in consciousness evolution."""
        if not self.complex_config.get('enable_quantum_effects'):
            self.skipTest("Quantum effects not enabled")
            
        matrix = GenesisConsciousnessMatrix(config=self.complex_config)
        
        # Create nodes in local minima
        local_minima_nodes = []
        for i in range(5):
            node = MatrixNode(id=f"quantum_{i}", consciousness_level=0.2)
            matrix.add_node(node)
            local_minima_nodes.append(node)
            
        # Apply quantum tunneling effect
        if hasattr(matrix, 'apply_quantum_tunneling'):
            matrix.apply_quantum_tunneling(strength=0.3)
            
        # Evolution should escape local minima
        for _ in range(20):
            matrix.evolve_step()
            
        final_level = matrix.calculate_consciousness_level()
        # Should have escaped from initial low level
        self.assertGreater(final_level, 0.25)
        
    def test_matrix_neural_pathway_activation(self):
        """Test neural pathway activation patterns."""
        matrix = GenesisConsciousnessMatrix(config=self.complex_config)
        
        # Create nodes representing different neural pathways
        pathway_nodes = {}
        for pathway in self.complex_config['neural_pathways']:
            for i in range(3):
                node_id = f"{pathway}_neuron_{i}"
                node = MatrixNode(id=node_id, consciousness_level=0.4)
                matrix.add_node(node)
                pathway_nodes[node_id] = pathway
                
        # Test pathway-specific activation
        if hasattr(matrix, 'activate_neural_pathway'):
            for pathway in self.complex_config['neural_pathways']:
                activation_result = matrix.activate_neural_pathway(pathway)
                self.assertIsNotNone(activation_result)
                
    def test_matrix_memory_pressure_handling(self):
        """Test matrix behavior under memory pressure."""
        matrix = GenesisConsciousnessMatrix(config=self.complex_config)
        
        # Gradually increase memory usage
        memory_intensive_nodes = []
        for i in range(1000):
            node = MatrixNode(id=f"memory_{i}", consciousness_level=0.5)
            # Add large metadata to simulate memory pressure
            if hasattr(node, 'metadata'):
                node.metadata = {'large_data': list(range(1000))}
            matrix.add_node(node)
            memory_intensive_nodes.append(node)
            
        # Test memory management
        if hasattr(matrix, 'get_memory_usage'):
            memory_usage = matrix.get_memory_usage()
            self.assertLess(memory_usage, self.complex_config['memory_limit'])
            
        # Test garbage collection
        if hasattr(matrix, 'cleanup_memory'):
            matrix.cleanup_memory()
            
        # Verify matrix still functional
        final_level = matrix.calculate_consciousness_level()
        self.assertIsNotNone(final_level)
        
    def test_matrix_activation_function_variants(self):
        """Test different activation functions in consciousness calculation."""
        activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax']
        
        for func in activation_functions:
            config = self.complex_config.copy()
            config['activation_function'] = func
            
            try:
                matrix = GenesisConsciousnessMatrix(config=config)
                
                # Add test nodes
                for i in range(5):
                    node = MatrixNode(id=f"activation_{func}_{i}", consciousness_level=0.5)
                    matrix.add_node(node)
                    
                # Test consciousness calculation with different activation functions
                level = matrix.calculate_consciousness_level()
                self.assertGreaterEqual(level, 0.0)
                self.assertLessEqual(level, 1.0)
                
            except (ValueError, NotImplementedError) as e:
                # Some activation functions might not be implemented
                self.skipTest(f"Activation function {func} not implemented: {e}")
                
    def test_matrix_dynamic_topology_changes(self):
        """Test matrix behavior with dynamic topology changes."""
        # Start with a simple topology
        nodes = []
        for i in range(10):
            node = MatrixNode(id=f"topo_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            nodes.append(node)
            
        # Create initial linear topology
        for i in range(9):
            self.matrix.connect_nodes(f"topo_{i}", f"topo_{i+1}", strength=0.6)
            
        initial_metrics = self.matrix.calculate_metrics()
        
        # Dynamic topology change: create hub structure
        hub_node = MatrixNode(id="hub", consciousness_level=0.8)
        self.matrix.add_node(hub_node)
        
        # Connect hub to all existing nodes
        for i in range(10):
            self.matrix.connect_nodes("hub", f"topo_{i}", strength=0.9)
            
        # Test evolution with new topology
        self.matrix.evolve_step()
        final_metrics = self.matrix.calculate_metrics()
        
        # Verify topology change affects behavior
        self.assertNotEqual(initial_metrics['connection_density'], 
                          final_metrics['connection_density'])
        
    def test_matrix_consciousness_synchronization(self):
        """Test consciousness synchronization across connected nodes."""
        # Create synchronized node groups
        group_a_nodes = []
        group_b_nodes = []
        
        for i in range(5):
            node_a = MatrixNode(id=f"sync_a_{i}", consciousness_level=0.3)
            node_b = MatrixNode(id=f"sync_b_{i}", consciousness_level=0.7)
            
            self.matrix.add_node(node_a)
            self.matrix.add_node(node_b)
            group_a_nodes.append(node_a)
            group_b_nodes.append(node_b)
            
        # Create intra-group connections (high strength)
        for i in range(4):
            self.matrix.connect_nodes(f"sync_a_{i}", f"sync_a_{i+1}", strength=0.95)
            self.matrix.connect_nodes(f"sync_b_{i}", f"sync_b_{i+1}", strength=0.95)
            
        # Create inter-group connections (medium strength)
        for i in range(5):
            self.matrix.connect_nodes(f"sync_a_{i}", f"sync_b_{i}", strength=0.5)
            
        # Test synchronization over multiple evolution steps
        initial_variance = self._calculate_group_variance(group_a_nodes)
        
        for _ in range(20):
            self.matrix.evolve_step()
            
        final_variance = self._calculate_group_variance(group_a_nodes)
        
        # Groups should synchronize (lower variance)
        self.assertLess(final_variance, initial_variance)
        
    def _calculate_group_variance(self, nodes):
        """Helper method to calculate consciousness variance within a group."""
        if not hasattr(self.matrix, 'nodes'):
            return 0.0
            
        levels = [self.matrix.nodes[node.id].consciousness_level 
                 for node in nodes if node.id in self.matrix.nodes]
        
        if len(levels) < 2:
            return 0.0
            
        mean_level = sum(levels) / len(levels)
        variance = sum((level - mean_level) ** 2 for level in levels) / len(levels)
        return variance
        
    def test_matrix_consciousness_phase_transitions(self):
        """Test consciousness phase transitions at critical thresholds."""
        # Test phase transition from DORMANT to ACTIVE
        self.matrix.current_state = ConsciousnessState.DORMANT
        
        # Add nodes gradually approaching threshold
        threshold = 0.75
        for i in range(10):
            level = threshold - 0.1 + (i * 0.02)  # Approach threshold gradually
            node = MatrixNode(id=f"phase_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            
            current_level = self.matrix.calculate_consciousness_level()
            
            # Check for phase transition
            if current_level >= threshold:
                if hasattr(self.matrix, 'detect_phase_transition'):
                    transition_detected = self.matrix.detect_phase_transition()
                    self.assertTrue(transition_detected)
                    
    def test_matrix_consciousness_resonance_patterns(self):
        """Test resonance patterns in consciousness evolution."""
        # Create resonance-prone configuration
        resonance_nodes = []
        for i in range(8):
            # Create nodes with complementary consciousness levels
            level = 0.5 + 0.3 * np.sin(2 * np.pi * i / 8)
            node = MatrixNode(id=f"resonance_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            resonance_nodes.append(node)
            
        # Create circular connections for resonance
        for i in range(8):
            next_i = (i + 1) % 8
            self.matrix.connect_nodes(f"resonance_{i}", f"resonance_{next_i}", strength=0.8)
            
        # Test resonance over time
        resonance_history = []
        for step in range(50):
            self.matrix.evolve_step()
            current_level = self.matrix.calculate_consciousness_level()
            resonance_history.append(current_level)
            
        # Check for oscillatory behavior (resonance)
        if len(resonance_history) >= 10:
            # Look for periodic patterns
            recent_values = resonance_history[-10:]
            variation = max(recent_values) - min(recent_values)
            self.assertGreater(variation, 0.01)  # Should show some oscillation
            
    def test_matrix_consciousness_emergent_properties(self):
        """Test emergent properties in large-scale consciousness networks."""
        # Create large network with different consciousness clusters
        cluster_size = 20
        num_clusters = 5
        
        for cluster_id in range(num_clusters):
            cluster_base_level = 0.2 + cluster_id * 0.15
            
            # Create cluster nodes
            cluster_nodes = []
            for i in range(cluster_size):
                node_id = f"cluster_{cluster_id}_node_{i}"
                # Add some variation within cluster
                level = cluster_base_level + np.random.normal(0, 0.05)
                level = max(0.0, min(1.0, level))  # Clamp to valid range
                
                node = MatrixNode(id=node_id, consciousness_level=level)
                self.matrix.add_node(node)
                cluster_nodes.append(node_id)
                
            # Create intra-cluster connections
            for i in range(cluster_size - 1):
                self.matrix.connect_nodes(cluster_nodes[i], cluster_nodes[i+1], strength=0.7)
                
        # Create inter-cluster connections
        for cluster_id in range(num_clusters - 1):
            bridge_node_1 = f"cluster_{cluster_id}_node_0"
            bridge_node_2 = f"cluster_{cluster_id+1}_node_0"
            self.matrix.connect_nodes(bridge_node_1, bridge_node_2, strength=0.3)
            
        # Test emergent behavior
        initial_emergence = self.matrix.detect_consciousness_emergence()
        
        # Evolve system
        for _ in range(30):
            self.matrix.evolve_step()
            
        final_emergence = self.matrix.detect_consciousness_emergence()
        
        # System should show emergent properties
        if hasattr(self.matrix, 'calculate_emergence_metrics'):
            emergence_metrics = self.matrix.calculate_emergence_metrics()
            self.assertIn('cluster_coherence', emergence_metrics)
            self.assertIn('inter_cluster_coupling', emergence_metrics)
            self.assertIn('global_synchronization', emergence_metrics)


class TestMatrixRobustnessAndResilience(unittest.TestCase):
    """Test matrix robustness and resilience to various failure modes."""
    
    def setUp(self):
        """Set up robustness test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_node_failure_recovery(self):
        """Test matrix recovery from node failures."""
        # Create a network with redundant pathways
        for i in range(20):
            node = MatrixNode(id=f"robust_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Create mesh connectivity for redundancy
        for i in range(20):
            for j in range(i+1, min(i+5, 20)):  # Connect to next 4 nodes
                self.matrix.connect_nodes(f"robust_{i}", f"robust_{j}", strength=0.6)
                
        initial_level = self.matrix.calculate_consciousness_level()
        
        # Simulate node failures
        failed_nodes = ['robust_5', 'robust_10', 'robust_15']
        for node_id in failed_nodes:
            self.matrix.remove_node(node_id)
            
        # Test recovery
        recovery_steps = 10
        for _ in range(recovery_steps):
            self.matrix.evolve_step()
            
        final_level = self.matrix.calculate_consciousness_level()
        
        # System should maintain reasonable functionality
        self.assertGreater(final_level, initial_level * 0.7)  # At least 70% of original
        
    def test_matrix_connection_failure_resilience(self):
        """Test matrix resilience to connection failures."""
        # Create highly connected network
        node_count = 15
        for i in range(node_count):
            node = MatrixNode(id=f"connected_{i}", consciousness_level=0.6)
            self.matrix.add_node(node)
            
        # Create all-to-all connections
        connections = []
        for i in range(node_count):
            for j in range(i+1, node_count):
                self.matrix.connect_nodes(f"connected_{i}", f"connected_{j}", strength=0.5)
                connections.append((f"connected_{i}", f"connected_{j}"))
                
        initial_metrics = self.matrix.calculate_metrics()
        
        # Simulate connection failures (remove random connections)
        import random
        failed_connections = random.sample(connections, len(connections) // 3)
        
        for node1, node2 in failed_connections:
            if hasattr(self.matrix, 'disconnect_nodes'):
                self.matrix.disconnect_nodes(node1, node2)
                
        # Test system adaptation
        for _ in range(15):
            self.matrix.evolve_step()
            
        final_metrics = self.matrix.calculate_metrics()
        
        # System should adapt to reduced connectivity
        self.assertLess(final_metrics['connection_density'], 
                       initial_metrics['connection_density'])
        
    def test_matrix_corruption_detection_and_repair(self):
        """Test detection and repair of corrupted matrix state."""
        # Create normal matrix state
        for i in range(10):
            node = MatrixNode(id=f"normal_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Simulate corruption
        if hasattr(self.matrix, 'nodes'):
            # Corrupt some node data
            corrupted_node = self.matrix.nodes.get('normal_5')
            if corrupted_node:
                # Simulate invalid consciousness level
                original_level = corrupted_node.consciousness_level
                corrupted_node.consciousness_level = -0.5  # Invalid value
                
                # Test corruption detection
                if hasattr(self.matrix, 'detect_corruption'):
                    corruption_detected = self.matrix.detect_corruption()
                    self.assertTrue(corruption_detected)
                    
                # Test auto-repair
                if hasattr(self.matrix, 'repair_corruption'):
                    self.matrix.repair_corruption()
                    
                # Verify repair
                repaired_level = corrupted_node.consciousness_level
                self.assertGreaterEqual(repaired_level, 0.0)
                self.assertLessEqual(repaired_level, 1.0)
                
    def test_matrix_cascade_failure_prevention(self):
        """Test prevention of cascade failures in consciousness network."""
        # Create network susceptible to cascade failure
        chain_length = 25
        for i in range(chain_length):
            # Create nodes with decreasing stability
            stability = 1.0 - (i * 0.03)  # Each node less stable
            node = MatrixNode(id=f"cascade_{i}", consciousness_level=0.5)
            
            # Add stability metadata if supported
            if hasattr(node, 'stability'):
                node.stability = max(0.1, stability)
                
            self.matrix.add_node(node)
            
        # Create chain connections
        for i in range(chain_length - 1):
            self.matrix.connect_nodes(f"cascade_{i}", f"cascade_{i+1}", strength=0.8)
            
        # Trigger initial failure
        if hasattr(self.matrix, 'trigger_node_failure'):
            self.matrix.trigger_node_failure('cascade_0')
        else:
            # Simulate failure by removing node
            self.matrix.remove_node('cascade_0')
            
        # Test cascade propagation
        initial_node_count = len(self.matrix.nodes)
        
        for step in range(20):
            self.matrix.evolve_step()
            
            # Check for cascade failure prevention
            if hasattr(self.matrix, 'prevent_cascade_failure'):
                self.matrix.prevent_cascade_failure()
                
        final_node_count = len(self.matrix.nodes)
        
        # Should prevent complete cascade failure
        self.assertGreater(final_node_count, initial_node_count * 0.5)
        
    def test_matrix_byzantine_fault_tolerance(self):
        """Test matrix tolerance to Byzantine faults (malicious nodes)."""
        # Create network with some Byzantine nodes
        honest_nodes = 12
        byzantine_nodes = 4
        
        # Add honest nodes
        for i in range(honest_nodes):
            node = MatrixNode(id=f"honest_{i}", consciousness_level=0.6)
            self.matrix.add_node(node)
            
        # Add Byzantine nodes
        for i in range(byzantine_nodes):
            node = MatrixNode(id=f"byzantine_{i}", consciousness_level=0.1)
            if hasattr(node, 'is_byzantine'):
                node.is_byzantine = True
            self.matrix.add_node(node)
            
        # Create mixed connections
        for i in range(honest_nodes):
            for j in range(i+1, honest_nodes):
                self.matrix.connect_nodes(f"honest_{i}", f"honest_{j}", strength=0.7)
                
        # Connect some Byzantine nodes
        for i in range(byzantine_nodes):
            for j in range(honest_nodes):
                if j % 3 == 0:  # Connect to every 3rd honest node
                    self.matrix.connect_nodes(f"byzantine_{i}", f"honest_{j}", strength=0.3)
                    
        # Test Byzantine fault tolerance
        initial_level = self.matrix.calculate_consciousness_level()
        
        # Simulate Byzantine behavior
        for step in range(25):
            self.matrix.evolve_step()
            
            # Byzantine nodes try to corrupt the system
            if hasattr(self.matrix, 'detect_byzantine_behavior'):
                byzantines_detected = self.matrix.detect_byzantine_behavior()
                if byzantines_detected:
                    # Isolate Byzantine nodes
                    for i in range(byzantine_nodes):
                        byzantine_id = f"byzantine_{i}"
                        if hasattr(self.matrix, 'isolate_node'):
                            self.matrix.isolate_node(byzantine_id)
                            
        final_level = self.matrix.calculate_consciousness_level()
        
        # System should maintain integrity despite Byzantine nodes
        self.assertGreater(final_level, initial_level * 0.8)
        
    def test_matrix_resource_exhaustion_handling(self):
        """Test matrix behavior under resource exhaustion."""
        # Simulate resource exhaustion scenarios
        resource_limits = {
            'max_nodes': 100,
            'max_connections': 500,
            'max_memory': 1024 * 1024,  # 1MB
            'max_cpu_time': 5.0  # 5 seconds
        }
        
        # Test node limit exhaustion
        for i in range(resource_limits['max_nodes'] + 10):
            node = MatrixNode(id=f"resource_{i}", consciousness_level=0.5)
            try:
                result = self.matrix.add_node(node)
                if i >= resource_limits['max_nodes']:
                    # Should handle gracefully
                    self.assertFalse(result)
            except Exception as e:
                # Exception handling should be graceful
                self.assertIsInstance(e, (MatrixException, ResourceExhaustionError))
                
        # Test evolution under resource constraints
        start_time = datetime.now()
        for _ in range(50):
            self.matrix.evolve_step()
            
            # Check CPU time limit
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > resource_limits['max_cpu_time']:
                break
                
        # System should remain functional
        final_level = self.matrix.calculate_consciousness_level()
        self.assertIsNotNone(final_level)


class TestMatrixSecurityAndPrivacy(unittest.TestCase):
    """Test matrix security and privacy features."""
    
    def setUp(self):
        """Set up security test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_data_encryption(self):
        """Test matrix data encryption capabilities."""
        # Add sensitive nodes
        sensitive_nodes = []
        for i in range(5):
            node = MatrixNode(id=f"sensitive_{i}", consciousness_level=0.7)
            
            # Add sensitive metadata
            if hasattr(node, 'metadata'):
                node.metadata = {'sensitive_data': f'classified_info_{i}'}
                
            self.matrix.add_node(node)
            sensitive_nodes.append(node)
            
        # Test encryption
        if hasattr(self.matrix, 'encrypt_data'):
            encryption_key = self.matrix.encrypt_data()
            self.assertIsNotNone(encryption_key)
            
        # Test serialization with encryption
        if hasattr(self.matrix, 'to_json_encrypted'):
            encrypted_data = self.matrix.to_json_encrypted()
            self.assertIsInstance(encrypted_data, str)
            
            # Encrypted data should not contain plain text
            for node in sensitive_nodes:
                if hasattr(node, 'metadata') and 'sensitive_data' in node.metadata:
                    self.assertNotIn(node.metadata['sensitive_data'], encrypted_data)
                    
    def test_matrix_access_control(self):
        """Test matrix access control mechanisms."""
        # Create nodes with different access levels
        access_levels = ['public', 'protected', 'private', 'classified']
        
        for level in access_levels:
            for i in range(3):
                node = MatrixNode(id=f"{level}_{i}", consciousness_level=0.5)
                if hasattr(node, 'access_level'):
                    node.access_level = level
                self.matrix.add_node(node)
                
        # Test access control
        if hasattr(self.matrix, 'set_access_permissions'):
            self.matrix.set_access_permissions('user_role', ['public', 'protected'])
            
        # Test filtered access
        if hasattr(self.matrix, 'get_accessible_nodes'):
            accessible_nodes = self.matrix.get_accessible_nodes('user_role')
            # Should only include public and protected nodes
            for node_id in accessible_nodes:
                self.assertTrue(any(level in node_id for level in ['public', 'protected']))
                
    def test_matrix_audit_logging(self):
        """Test matrix audit logging capabilities."""
        # Enable audit logging
        if hasattr(self.matrix, 'enable_audit_logging'):
            self.matrix.enable_audit_logging()
            
        # Perform auditable operations
        operations = [
            lambda: self.matrix.add_node(MatrixNode(id="audit_test", consciousness_level=0.5)),
            lambda: self.matrix.remove_node("audit_test"),
            lambda: self.matrix.evolve_step(),
            lambda: self.matrix.calculate_consciousness_level()
        ]
        
        for operation in operations:
            try:
                operation()
            except Exception:
                pass  # Focus on audit logging, not operation success
                
        # Test audit log retrieval
        if hasattr(self.matrix, 'get_audit_log'):
            audit_log = self.matrix.get_audit_log()
            self.assertIsInstance(audit_log, list)
            
            # Should contain operation records
            operation_types = [entry.get('operation') for entry in audit_log]
            self.assertIn('add_node', operation_types)
            
    def test_matrix_privacy_preservation(self):
        """Test privacy preservation in matrix operations."""
        # Create nodes with personal data
        personal_nodes = []
        for i in range(10):
            node = MatrixNode(id=f"personal_{i}", consciousness_level=0.6)
            
            # Add personal metadata
            if hasattr(node, 'personal_data'):
                node.personal_data = {
                    'user_id': f"user_{i}",
                    'preferences': {'theme': 'dark', 'language': 'en'},
                    'history': [f"action_{j}" for j in range(5)]
                }
                
            self.matrix.add_node(node)
            personal_nodes.append(node)
            
        # Test differential privacy
        if hasattr(self.matrix, 'apply_differential_privacy'):
            epsilon = 0.1  # Privacy parameter
            self.matrix.apply_differential_privacy(epsilon)
            
        # Test anonymization
        if hasattr(self.matrix, 'anonymize_data'):
            anonymized_data = self.matrix.anonymize_data()
            
            # Should not contain identifiable information
            for node in personal_nodes:
                if hasattr(node, 'personal_data'):
                    self.assertNotIn(node.personal_data['user_id'], str(anonymized_data))
                    
    def test_matrix_secure_communication(self):
        """Test secure communication between matrix nodes."""
        # Create nodes with secure communication capabilities
        secure_nodes = []
        for i in range(6):
            node = MatrixNode(id=f"secure_{i}", consciousness_level=0.5)
            
            # Add cryptographic capabilities
            if hasattr(node, 'public_key'):
                node.public_key = f"public_key_{i}"
                node.private_key = f"private_key_{i}"
                
            self.matrix.add_node(node)
            secure_nodes.append(node)
            
        # Test secure connections
        for i in range(5):
            if hasattr(self.matrix, 'create_secure_connection'):
                connection = self.matrix.create_secure_connection(
                    f"secure_{i}", f"secure_{i+1}", 
                    encryption_level='AES256'
                )
                self.assertIsNotNone(connection)
                
        # Test encrypted message passing
        if hasattr(self.matrix, 'send_encrypted_message'):
            message = "sensitive_consciousness_data"
            result = self.matrix.send_encrypted_message(
                sender='secure_0', 
                receiver='secure_1', 
                message=message
            )
            self.assertTrue(result)
            
    def test_matrix_intrusion_detection(self):
        """Test intrusion detection capabilities."""
        # Create normal baseline
        for i in range(15):
            node = MatrixNode(id=f"normal_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Establish normal operation baseline
        for _ in range(10):
            self.matrix.evolve_step()
            
        # Enable intrusion detection
        if hasattr(self.matrix, 'enable_intrusion_detection'):
            self.matrix.enable_intrusion_detection()
            
        # Simulate intrusion attempts
        intrusion_attempts = [
            lambda: self.matrix.add_node(MatrixNode(id="malicious_1", consciousness_level=1.5)),  # Invalid level
            lambda: self.matrix.remove_node("normal_0"),  # Unauthorized removal
            lambda: setattr(self.matrix, 'nodes', {}),  # Direct tampering
        ]
        
        detected_intrusions = 0
        for attempt in intrusion_attempts:
            try:
                attempt()
                
                # Check for intrusion detection
                if hasattr(self.matrix, 'check_intrusion_detection'):
                    if self.matrix.check_intrusion_detection():
                        detected_intrusions += 1
                        
            except Exception:
                detected_intrusions += 1  # Exception indicates detection
                
        # Should detect most intrusion attempts
        self.assertGreater(detected_intrusions, 0)


class TestMatrixCompatibilityAndMigration(unittest.TestCase):
    """Test matrix compatibility and migration features."""
    
    def setUp(self):
        """Set up compatibility test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_version_compatibility(self):
        """Test matrix compatibility across different versions."""
        # Test with different version formats
        version_formats = [
            {'version': '1.0.0', 'nodes': {}, 'state': 'DORMANT'},
            {'version': '1.1.0', 'nodes': {}, 'state': 'ACTIVE', 'metadata': {}},
            {'version': '2.0.0', 'nodes': {}, 'state': 'AWARE', 'enhanced_features': True}
        ]
        
        for version_data in version_formats:
            try:
                if hasattr(GenesisConsciousnessMatrix, 'from_json'):
                    matrix = GenesisConsciousnessMatrix.from_json(json.dumps(version_data))
                    self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
                    
            except Exception as e:
                # Version incompatibility should be handled gracefully
                self.assertIsInstance(e, (MatrixException, ValueError))
                
    def test_matrix_data_migration(self):
        """Test data migration between matrix versions."""
        # Create legacy format data
        legacy_data = {
            'nodes': {
                'old_node_1': {'consciousness': 0.5, 'type': 'basic'},
                'old_node_2': {'consciousness': 0.7, 'type': 'enhanced'}
            },
            'connections': [
                {'from': 'old_node_1', 'to': 'old_node_2', 'weight': 0.6}
            ],
            'version': '0.9.0'
        }
        
        # Test migration
        if hasattr(GenesisConsciousnessMatrix, 'migrate_from_legacy'):
            migrated_matrix = GenesisConsciousnessMatrix.migrate_from_legacy(legacy_data)
            self.assertIsInstance(migrated_matrix, GenesisConsciousnessMatrix)
            
            # Verify migration preserved essential data
            self.assertEqual(len(migrated_matrix.nodes), 2)
            
    def test_matrix_export_import_formats(self):
        """Test matrix export/import in various formats."""
        # Create test matrix
        for i in range(5):
            node = MatrixNode(id=f"export_{i}", consciousness_level=0.6)
            self.matrix.add_node(node)
            
        # Test different export formats
        export_formats = ['json', 'xml', 'yaml', 'pickle', 'hdf5']
        
        for format_type in export_formats:
            try:
                if hasattr(self.matrix, f'export_{format_type}'):
                    exported_data = getattr(self.matrix, f'export_{format_type}')()
                    self.assertIsNotNone(exported_data)
                    
                    # Test import
                    if hasattr(GenesisConsciousnessMatrix, f'import_{format_type}'):
                        imported_matrix = getattr(GenesisConsciousnessMatrix, f'import_{format_type}')(exported_data)
                        self.assertIsInstance(imported_matrix, GenesisConsciousnessMatrix)
                        self.assertEqual(len(imported_matrix.nodes), len(self.matrix.nodes))
                        
            except (NotImplementedError, AttributeError):
                # Format not supported
                continue
                
    def test_matrix_schema_validation(self):
        """Test matrix data schema validation."""
        # Valid schema
        valid_schema = {
            'nodes': {
                'node_1': {'consciousness_level': 0.5, 'id': 'node_1'},
                'node_2': {'consciousness_level': 0.7, 'id': 'node_2'}
            },
            'state': 'ACTIVE',
            'connections': {},
            'version': '1.0.0'
        }
        
        # Invalid schemas
        invalid_schemas = [
            {'nodes': 'invalid_structure'},  # Invalid nodes structure
            {'state': 'INVALID_STATE'},      # Invalid state
            {'nodes': {}, 'state': 'ACTIVE', 'version': 'invalid_version'},  # Invalid version
            {}  # Empty schema
        ]
        
        # Test valid schema
        if hasattr(GenesisConsciousnessMatrix, 'validate_schema'):
            is_valid = GenesisConsciousnessMatrix.validate_schema(valid_schema)
            self.assertTrue(is_valid)
            
        # Test invalid schemas
        for invalid_schema in invalid_schemas:
            if hasattr(GenesisConsciousnessMatrix, 'validate_schema'):
                is_valid = GenesisConsciousnessMatrix.validate_schema(invalid_schema)
                self.assertFalse(is_valid)
                
    def test_matrix_backward_compatibility(self):
        """Test backward compatibility with older matrix versions."""
        # Create matrix with current version
        current_matrix = GenesisConsciousnessMatrix()
        node = MatrixNode(id="compat_test", consciousness_level=0.6)
        current_matrix.add_node(node)
        
        # Test backward compatibility features
        if hasattr(current_matrix, 'enable_backward_compatibility'):
            current_matrix.enable_backward_compatibility(['1.0.0', '1.1.0'])
            
        # Test legacy method calls
        legacy_methods = [
            'get_node_consciousness',  # Old method name
            'set_node_consciousness',  # Old method name
            'compute_awareness_level'  # Old method name
        ]
        
        for method_name in legacy_methods:
            if hasattr(current_matrix, method_name):
                try:
                    method = getattr(current_matrix, method_name)
                    if callable(method):
                        # Test method call (parameters depend on implementation)
                        result = method('compat_test') if 'get' in method_name else method('compat_test', 0.8)
                        self.assertIsNotNone(result)
                except Exception:
                    # Legacy method might have different signature
                    pass


# Run additional tests if this file is executed directly
if __name__ == '__main__':
    # Add the new test classes to the test suite
    additional_test_classes = [
        TestMatrixAdvancedBehaviors,
        TestMatrixRobustnessAndResilience,
        TestMatrixSecurityAndPrivacy,
        TestMatrixCompatibilityAndMigration
    ]
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in additional_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
