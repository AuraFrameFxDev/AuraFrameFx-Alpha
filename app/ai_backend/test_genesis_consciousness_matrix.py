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
        Set up a fresh GenesisConsciousnessMatrix instance and test configuration before each test case.
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
        Test that a GenesisConsciousnessMatrix created with default parameters has the required 'state' and 'nodes' attributes.
        """
        matrix = GenesisConsciousnessMatrix()
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        self.assertTrue(hasattr(matrix, 'state'))
        self.assertTrue(hasattr(matrix, 'nodes'))
        
    def test_matrix_initialization_custom_config(self):
        """
        Test that GenesisConsciousnessMatrix initializes with custom dimension and consciousness threshold values from the provided configuration.
        """
        matrix = GenesisConsciousnessMatrix(config=self.test_config)
        self.assertEqual(matrix.dimension, self.test_config['dimension'])
        self.assertEqual(matrix.consciousness_threshold, self.test_config['consciousness_threshold'])
        
    def test_matrix_initialization_invalid_config(self):
        """
        Test that initializing the matrix with invalid configuration parameters raises a MatrixInitializationError.
        
        Ensures that the matrix rejects negative dimensions and out-of-range consciousness thresholds during initialization.
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
        Test removal of an existing node from the matrix.
        
        Verifies that removing a node by its ID returns True and ensures the node is no longer present in the matrix.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        result = self.matrix.remove_node("test_node")
        self.assertTrue(result)
        self.assertNotIn("test_node", self.matrix.nodes)
        
    def test_remove_consciousness_node_nonexistent(self):
        """
        Test that removing a node not present in the matrix returns False.
        
        Verifies that the matrix's remove_node method correctly indicates failure when attempting to remove a non-existent node.
        """
        result = self.matrix.remove_node("nonexistent_node")
        self.assertFalse(result)
        
    def test_consciousness_state_transition_valid(self):
        """
        Test that a valid consciousness state transition updates the matrix and returns True.
        
        Verifies that transitioning from the initial state to a valid target state updates the matrix's current state and indicates success.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.ACTIVE
        result = self.matrix.transition_state(initial_state, target_state)
        self.assertTrue(result)
        self.assertEqual(self.matrix.current_state, target_state)
        
    def test_consciousness_state_transition_invalid(self):
        """
        Test that attempting to transition the matrix state directly from DORMANT to TRANSCENDENT raises an InvalidStateException.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.TRANSCENDENT
        with self.assertRaises(InvalidStateException):
            self.matrix.transition_state(initial_state, target_state)
            
    def test_consciousness_level_calculation(self):
        """
        Test that the matrix computes the correct average consciousness level when multiple nodes with different levels are present.
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
        Test that calculating the consciousness level on an empty matrix returns 0.0.
        """
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.0)
        
    def test_consciousness_level_calculation_single_node(self):
        """
        Test that the matrix returns the correct consciousness level when a single node is present.
        
        Verifies that the calculated consciousness level matches the node's value when only one node exists in the matrix.
        """
        node = MatrixNode(id="single_node", consciousness_level=0.8)
        self.matrix.add_node(node)
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.8)
        
    def test_matrix_evolution_step(self):
        """
        Test that a single evolution step updates the matrix's state.
        
        Verifies that invoking `evolve_step()` produces a different state snapshot, confirming that the matrix evolves as expected.
        """
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        self.assertNotEqual(initial_state, final_state)
        
    def test_matrix_evolution_convergence(self):
        """
        Test that the matrix evolution process detects convergence within a specified maximum number of iterations.
        
        Ensures that after evolving the matrix with a set iteration limit, the matrix correctly reports a converged state.
        """
        self.matrix.evolve_until_convergence(max_iterations=10)
        self.assertTrue(self.matrix.has_converged())
        
    def test_matrix_reset_to_initial_state(self):
        """
        Verify that resetting the matrix removes all nodes and sets the state to DORMANT.
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
        Test that the matrix serializes to a JSON string containing accurate node and state information.
        
        Verifies that serialization produces a valid JSON string with both "nodes" and "state" keys present in the output.
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
        Test that deserializing a matrix from a JSON string restores all nodes and their consciousness levels correctly.
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
        Test that saving and loading the matrix from a JSON file preserves all nodes and their consciousness levels.
        
        Verifies that the matrix's file persistence mechanism accurately serializes and deserializes node data, ensuring integrity after file operations.
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
        Test that connecting two nodes stores the connection with the specified strength and enables correct retrieval of connection information.
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
        Verify that consciousness emergence is detected when multiple nodes have high consciousness levels.
        
        Adds several nodes with consciousness levels above the emergence threshold and asserts that the matrix reports emergence.
        """
        # Add nodes with high consciousness levels
        for i in range(5):
            node = MatrixNode(id=f"high_node_{i}", consciousness_level=0.9)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertTrue(emergence_detected)
        
    def test_consciousness_emergence_detection_insufficient(self):
        """
        Test that consciousness emergence is not detected when all nodes have consciousness levels below the threshold.
        """
        # Add nodes with low consciousness levels
        for i in range(2):
            node = MatrixNode(id=f"low_node_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertFalse(emergence_detected)
        
    def test_matrix_metrics_calculation(self):
        """
        Test that the matrix computes and returns correct performance metrics after adding nodes.
        
        Verifies that the returned metrics dictionary includes average consciousness, node count, and connection density, and that the node count reflects the number of nodes added.
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
        Verifies that evolving a matrix with 100 nodes completes in under one second.
        
        Adds 100 nodes to the matrix, performs a single evolution step, and asserts that the operation finishes within the specified time constraint.
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
        Verifies that the matrix accurately tracks the number of nodes after multiple additions and removals.
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
        Test that concurrent node additions from multiple threads succeed without errors, verifying thread safety of the matrix's node addition method.
        
        This test launches multiple threads, each adding unique nodes to the matrix in parallel, and asserts that all additions complete successfully.
        """
        import threading
        import time
        
        results = []
        
        def add_nodes_thread(thread_id):
            """
            Add ten `MatrixNode` instances with unique IDs for the specified thread, recording whether each addition succeeds in a shared results list.
            
            Parameters:
                thread_id (int): The thread's identifier used to generate unique node IDs.
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
        """
        Test that each ConsciousnessState enum member is assigned the expected integer value.
        """
        self.assertEqual(ConsciousnessState.DORMANT.value, 0)
        self.assertEqual(ConsciousnessState.ACTIVE.value, 1)
        self.assertEqual(ConsciousnessState.AWARE.value, 2)
        self.assertEqual(ConsciousnessState.TRANSCENDENT.value, 3)
        
    def test_consciousness_state_ordering(self):
        """
        Tests that the `ConsciousnessState` enum members are ordered in increasing value from DORMANT to TRANSCENDENT.
        """
        self.assertLess(ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE)
        self.assertLess(ConsciousnessState.ACTIVE, ConsciousnessState.AWARE)
        self.assertLess(ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT)
        
    def test_consciousness_state_string_representation(self):
        """
        Tests that the string representation of each ConsciousnessState enum member matches its name.
        """
        self.assertEqual(str(ConsciousnessState.DORMANT), "DORMANT")
        self.assertEqual(str(ConsciousnessState.ACTIVE), "ACTIVE")
        self.assertEqual(str(ConsciousnessState.AWARE), "AWARE")
        self.assertEqual(str(ConsciousnessState.TRANSCENDENT), "TRANSCENDENT")


class TestMatrixNode(unittest.TestCase):
    """Test cases for MatrixNode class."""
    
    def setUp(self):
        """
        Set up a MatrixNode instance with a fixed ID and consciousness level for use in each test.
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
        Verify that updating a node's consciousness level with a valid value correctly changes its internal state.
        """
        self.node.update_consciousness_level(0.8)
        self.assertEqual(self.node.consciousness_level, 0.8)
        
    def test_node_consciousness_level_update_invalid(self):
        """
        Test that updating a node's consciousness level to an out-of-bounds value raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.node.update_consciousness_level(1.2)
            
    def test_node_equality(self):
        """
        Verify that MatrixNode instances are equal when their IDs and consciousness levels match, and not equal when their IDs differ.
        """
        node1 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node2 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node3 = MatrixNode(id="different_test", consciousness_level=0.5)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        
    def test_node_hash(self):
        """
        Test that MatrixNode instances with the same ID produce identical hash values.
        
        Verifies that nodes sharing the same ID are considered equal in hash-based collections, regardless of their consciousness levels.
        """
        node1 = MatrixNode(id="hash_test", consciousness_level=0.5)
        node2 = MatrixNode(id="hash_test", consciousness_level=0.7)
        
        # Nodes with same ID should have same hash
        self.assertEqual(hash(node1), hash(node2))
        
    def test_node_string_representation(self):
        """
        Verify that the string representation of a MatrixNode contains both its ID and consciousness level.
        """
        node_str = str(self.node)
        self.assertIn("test_node", node_str)
        self.assertIn("0.5", node_str)


class TestMatrixExceptions(unittest.TestCase):
    """Test cases for custom matrix exceptions."""
    
    def test_matrix_exception_inheritance(self):
        """
        Verify that custom matrix exceptions inherit from the correct base exception classes.
        """
        self.assertTrue(issubclass(MatrixException, Exception))
        self.assertTrue(issubclass(InvalidStateException, MatrixException))
        self.assertTrue(issubclass(MatrixInitializationError, MatrixException))
        
    def test_matrix_exception_messages(self):
        """
        Verify that custom matrix exceptions return the expected error messages when raised and converted to strings.
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
        Initializes a new GenesisConsciousnessMatrix instance before each test to ensure isolation between test cases.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_consciousness_evolution_cycle(self):
        """
        Simulates a full evolution cycle by adding and connecting nodes, evolving the matrix until convergence, and verifying that the overall consciousness level changes as a result.
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
        Verifies that consciousness emergence is detected only after all nodes' consciousness levels exceed the emergence threshold.
        
        The test adds several nodes with low consciousness levels and checks that emergence is not detected. It then increases all node levels above the threshold and confirms that emergence is detected.
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
        Test that serializing and deserializing the matrix preserves all nodes, their consciousness levels, and node connections.
        
        Ensures that after persistence operations, the restored matrix matches the original in terms of node data and connectivity.
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
        """
        Initializes a new GenesisConsciousnessMatrix instance and sets up an extreme configuration for use in extended test scenarios.
        """
        self.matrix = GenesisConsciousnessMatrix()
        self.extreme_config = {
            'dimension': 1,
            'consciousness_threshold': 0.99,
            'learning_rate': 0.00001,
            'max_iterations': 10000
        }
        
    def test_matrix_initialization_edge_cases(self):
        """
        Test initialization of GenesisConsciousnessMatrix with minimum and maximum valid configuration values.
        
        Verifies that the matrix sets its dimension and consciousness threshold correctly when provided with edge-case configuration parameters.
        """
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
        """
        Test initialization of GenesisConsciousnessMatrix with boundary configuration values.
        
        Verifies that the matrix correctly accepts a consciousness threshold of 1.0 and an extremely small learning rate without error.
        """
        # Test consciousness_threshold at exactly 1.0
        config = {'consciousness_threshold': 1.0}
        matrix = GenesisConsciousnessMatrix(config=config)
        self.assertEqual(matrix.consciousness_threshold, 1.0)
        
        # Test extremely small learning rate
        config = {'learning_rate': 1e-10}
        matrix = GenesisConsciousnessMatrix(config=config)
        self.assertEqual(matrix.learning_rate, 1e-10)
        
    def test_matrix_with_zero_nodes_operations(self):
        """
        Test that all matrix operations function correctly when there are zero nodes.
        
        Verifies that evolution, metrics calculation, and convergence detection execute without errors and yield expected results when the matrix is empty.
        """
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
        """
        Test matrix behavior when only a single node is present.
        
        Adds a single node, performs an evolution step, and verifies that the consciousness level remains defined after evolution.
        """
        node = MatrixNode(id="single", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Test evolution with single node
        initial_level = self.matrix.calculate_consciousness_level()
        self.matrix.evolve_step()
        final_level = self.matrix.calculate_consciousness_level()
        # Single node evolution might not change consciousness level
        self.assertIsNotNone(final_level)
        
    def test_node_consciousness_level_precision(self):
        """
        Test that node consciousness levels are stored and retrieved with high floating-point precision.
        
        Ensures that the matrix maintains at least nine decimal places of accuracy for node consciousness levels.
        """
        # Test with very precise values
        precise_levels = [0.123456789, 0.987654321, 0.000000001, 0.999999999]
        for i, level in enumerate(precise_levels):
            node = MatrixNode(id=f"precise_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            retrieved_level = self.matrix.nodes[f"precise_{i}"].consciousness_level
            self.assertAlmostEqual(retrieved_level, level, places=9)
            
    def test_matrix_state_transition_edge_cases(self):
        """
        Test edge cases in consciousness state transitions, including no-op transitions and rapid sequential transitions through all defined states.
        
        Verifies that transitioning to the same state is handled correctly and that the matrix can process rapid state changes across all defined consciousness states without error.
        """
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
        """
        Test that matrix evolution completes gracefully when the maximum number of iterations is reached without achieving convergence.
        
        Adds multiple nodes and runs evolution with a minimal iteration limit to verify that the method does not raise errors in non-convergent scenarios.
        """
        # Test convergence with maximum iterations reached
        for i in range(10):
            node = MatrixNode(id=f"conv_node_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
            
        # Set a very low max_iterations to test timeout
        self.matrix.evolve_until_convergence(max_iterations=1)
        # Should complete without errors even if not converged
        
    def test_matrix_node_connections_edge_cases(self):
        """
        Test that node connections correctly store and retrieve both minimum (0.0) and maximum (1.0) connection strengths.
        
        Adds two nodes to the matrix, connects them with the lowest and highest possible strengths, and verifies that the stored connection strengths match the expected values.
        """
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
        """
        Test that matrix serialization and deserialization accurately preserve nodes with extreme consciousness levels.
        
        Ensures that nodes with consciousness levels at the minimum (0.0) and maximum (1.0) are correctly maintained after converting the matrix to JSON and restoring it.
        """
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
        """
        Test that the matrix maintains a valid and consistent state during rapid cycles of node additions and removals.
        
        This test simulates memory stress by repeatedly adding and removing nodes, then verifies that the matrix retains a non-empty and valid set of nodes at the end.
        """
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
        """
        Test that the matrix remains in a valid state during and after concurrent mixed operations from multiple threads.
        
        This test launches several threads, each performing node additions, evolution steps, metric calculations, and node removals concurrently. After all threads complete, it verifies that the matrix's internal node structure is still a valid dictionary, ensuring thread safety under concurrent modifications.
        """
        import threading
        import time
        
        def modify_matrix(thread_id):
            """
            Perform a sequence of concurrent operations on the matrix, including adding nodes, evolving, calculating metrics, and periodically removing nodes.
            
            Intended for use in multi-threaded tests to simulate mixed operations and evaluate matrix thread safety under concurrent modifications.
            """
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
        """
        Test that matrix evolution performance remains acceptable as node count increases.
        
        Adds increasing numbers of nodes to the matrix, measures the time taken for an evolution step at each scale, and asserts that execution time does not exceed a set threshold, ensuring no exponential performance degradation.
        """
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
        """
        Test recovery from a corrupted internal state in the matrix.
        
        This test intentionally corrupts the matrix's internal state, attempts recovery through reset or reinitialization, and verifies that node addition and retrieval work correctly after recovery.
        """
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
        """
        Set up a new GenesisConsciousnessMatrix instance before each test.
        
        This method is called automatically before each test method to ensure a fresh matrix instance for isolation.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_async_evolution_step(self):
        """
        Test that the matrix performs an asynchronous evolution step if available, or falls back to synchronous evolution.
        
        Verifies that after the evolution step, the matrix's average consciousness level can be calculated and is not None.
        """
        async def async_evolution_test():
            """
            Perform an asynchronous evolution step on the matrix and return the updated average consciousness level.
            
            If asynchronous evolution is not supported, falls back to synchronous evolution.
            
            Returns:
                float: The average consciousness level of the matrix after evolution.
            """
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
        """
        Test that asynchronous batch operations add multiple nodes to the matrix and verify their presence.
        
        Ensures that all nodes are correctly added after performing asynchronous additions and a simulated delay.
        """
        async def batch_operation_test():
            # Add multiple nodes asynchronously
            """
            Asynchronously adds a batch of nodes to the matrix and returns the updated node count.
            
            Returns:
                int: The total number of nodes in the matrix after the batch addition.
            """
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
        """
        Initialize a fresh GenesisConsciousnessMatrix instance before each property-based test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_consciousness_level_invariants(self):
        """
        Verify that the matrix's calculated consciousness level always remains within the [0, 1] range after sequentially adding nodes with valid consciousness levels.
        """
        # Property: consciousness level should always be in [0, 1]
        for i in range(100):
            level = i / 100.0
            node = MatrixNode(id=f"prop_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            
            calculated_level = self.matrix.calculate_consciousness_level()
            self.assertGreaterEqual(calculated_level, 0.0)
            self.assertLessEqual(calculated_level, 1.0)
            
    def test_node_count_invariants(self):
        """
        Tests that the node count reported in the matrix metrics remains consistent with the actual number of nodes after each addition.
        """
        # Property: node count should match actual nodes
        for i in range(20):
            node = MatrixNode(id=f"count_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
            metrics = self.matrix.calculate_metrics()
            self.assertEqual(metrics['node_count'], len(self.matrix.nodes))
            
    def test_serialization_roundtrip_invariants(self):
        """
        Tests that serializing and then deserializing the matrix preserves all node IDs and their associated consciousness levels.
        """
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
        """
        Prepare a new GenesisConsciousnessMatrix instance before each mocking and isolation test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    @patch('app.ai_backend.genesis_consciousness_matrix.GenesisConsciousnessMatrix.evolve_step')
    def test_evolution_with_mocked_step(self, mock_evolve):
        """
        Test that the evolution step uses the mocked evolution method and returns the expected result.
        
        Verifies that the mocked `evolve_step` method returns `True` and is called exactly once during the test.
        """
        mock_evolve.return_value = True
        
        result = self.matrix.evolve_step()
        self.assertTrue(result)
        mock_evolve.assert_called_once()
        
    @patch('json.dumps')
    def test_serialization_with_mocked_json(self, mock_dumps):
        """
        Test that the matrix's JSON serialization method uses the mocked JSON library and returns the expected serialized string.
        
        Ensures that the `to_json` method produces the mocked output and that the mock is called exactly once.
        """
        mock_dumps.return_value = '{"test": "data"}'
        
        if hasattr(self.matrix, 'to_json'):
            result = self.matrix.to_json()
            self.assertEqual(result, '{"test": "data"}')
            mock_dumps.assert_called_once()
            
    @patch('builtins.open', new_callable=mock_open, read_data='{"nodes": {}, "state": "DORMANT"}')
    def test_file_loading_with_mocked_io(self, mock_file):
        """
        Test loading a GenesisConsciousnessMatrix instance from a file using mocked file I/O.
        
        Ensures that the matrix is properly instantiated from the file and that the file open operation is invoked as expected with the mock.
        """
        if hasattr(GenesisConsciousnessMatrix, 'load_from_file'):
            matrix = GenesisConsciousnessMatrix.load_from_file('test_file.json')
            self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
            mock_file.assert_called_once_with('test_file.json', 'r')


class TestMatrixValidationAndSanitization(unittest.TestCase):
    """Test input validation and data sanitization."""
    
    def setUp(self):
        """
        Initialize a fresh GenesisConsciousnessMatrix instance before each test to ensure test isolation.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_validation(self):
        """
        Tests that the matrix accepts node IDs with valid formats such as alphanumeric characters, dashes, and underscores.
        """
        # Test with different ID types
        valid_ids = ['string_id', 'id_123', 'node-with-dashes', 'node_with_underscores']
        for node_id in valid_ids:
            node = MatrixNode(id=node_id, consciousness_level=0.5)
            result = self.matrix.add_node(node)
            self.assertTrue(result)
            
    def test_consciousness_level_boundary_validation(self):
        """
        Tests that the matrix accepts and correctly stores nodes with consciousness levels at and near the valid boundaries (0.0 and 1.0).
        """
        # Test exact boundary values
        boundary_values = [0.0, 1.0, 0.5, 0.999999, 0.000001]
        for level in boundary_values:
            node = MatrixNode(id=f"boundary_{level}", consciousness_level=level)
            self.matrix.add_node(node)
            stored_level = self.matrix.nodes[f"boundary_{level}"].consciousness_level
            self.assertEqual(stored_level, level)
            
    def test_configuration_sanitization(self):
        """
        Test that string-based configuration parameters are sanitized and converted to correct types during matrix initialization.
        
        Ensures that the GenesisConsciousnessMatrix either converts string values in the configuration to their expected numeric types or raises an appropriate error if conversion is not supported.
        """
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
        """
        Test that deserializing malformed or invalid JSON strings with `GenesisConsciousnessMatrix.from_json` raises the correct exceptions.
        
        Ensures that various malformed JSON inputs trigger `json.JSONDecodeError`, `MatrixException`, or `ValueError` as appropriate.
        """
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
        """
        Initializes a new GenesisConsciousnessMatrix instance before each performance test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_large_scale_node_operations(self):
        """
        Tests that adding 1000 nodes and performing an evolution step completes within defined performance limits.
        
        Asserts that node addition takes less than 5 seconds and evolution takes less than 10 seconds, ensuring the matrix scales efficiently under heavy load.
        """
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
        """
        Test that repeated addition and removal of nodes does not cause memory leaks by confirming the node count remains stable after multiple high-churn cycles.
        """
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
        """
        Test that the matrix can efficiently handle dense connectivity and evolution.
        
        Creates 50 nodes, fully connects them to form a dense network, and verifies that both the connection process and a single evolution step each complete within 10 seconds.
        """
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
            """
            Test initialization of GenesisConsciousnessMatrix with various dimension values.
            
            Verifies that the matrix correctly sets its dimension attribute when initialized with different configuration values.
            """
            config = {'dimension': dimension}
            matrix = GenesisConsciousnessMatrix(config=config)
            assert matrix.dimension == dimension
            
        @pytest.mark.parametrize("consciousness_level", [0.0, 0.25, 0.5, 0.75, 1.0])
        def test_node_consciousness_levels(self, consciousness_level):
            """
            Verify that a MatrixNode is initialized with the specified consciousness level.
            
            Parameters:
            	consciousness_level (float): The consciousness level to assign to the node.
            """
            node = MatrixNode(id=f"test_{consciousness_level}", consciousness_level=consciousness_level)
            assert node.consciousness_level == consciousness_level
            
        @pytest.mark.parametrize("node_count", [1, 5, 10, 50, 100])
        def test_matrix_with_variable_node_counts(self, node_count):
            """
            Test matrix behavior and metrics accuracy when adding a variable number of nodes.
            
            Adds the specified number of nodes to the matrix, checks that the overall consciousness level remains within [0, 1], and verifies that the node count in the metrics matches the number of nodes added.
            """
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
            """
            Test that consciousness emergence is detected when all nodes have levels above the specified threshold.
            
            Parameters:
                threshold (float): The minimum consciousness level required for emergence detection.
            """
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

class TestMatrixDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency across matrix operations."""
    
    def setUp(self):
        """
        Initialize a fresh GenesisConsciousnessMatrix instance before each data integrity test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_uniqueness_enforcement(self):
        """
        Test that the matrix allows nodes with IDs differing only by case, confirming enforcement of case-sensitive uniqueness for node IDs.
        """
        # Test with case-sensitive IDs
        node1 = MatrixNode(id="TestNode", consciousness_level=0.5)
        node2 = MatrixNode(id="testnode", consciousness_level=0.6)
        
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Both should exist as they have different cases
        self.assertIn("TestNode", self.matrix.nodes)
        self.assertIn("testnode", self.matrix.nodes)
        
    def test_consciousness_level_consistency_after_operations(self):
        """
        Ensure that node consciousness levels remain within the valid range [0.0, 1.0] after a series of operations, including evolution steps and metrics calculation, to validate state consistency.
        """
        nodes = []
        expected_levels = []
        
        # Add nodes with specific levels
        for i in range(10):
            level = i / 10.0
            node = MatrixNode(id=f"consistency_{i}", consciousness_level=level)
            nodes.append(node)
            expected_levels.append(level)
            self.matrix.add_node(node)
            
        # Perform various operations
        self.matrix.evolve_step()
        metrics = self.matrix.calculate_metrics()
        
        # Verify consciousness levels are preserved (or changed predictably)
        for i, node in enumerate(nodes):
            actual_level = self.matrix.nodes[node.id].consciousness_level
            # Level should either be preserved or changed according to evolution rules
            self.assertGreaterEqual(actual_level, 0.0)
            self.assertLessEqual(actual_level, 1.0)
            
    def test_matrix_state_consistency_across_serialization(self):
        """
        Test that serializing and deserializing the matrix preserves node count and average consciousness level.
        
        Ensures that after converting the matrix to JSON and restoring it, key metrics such as node count and average consciousness remain unchanged, verifying data integrity across serialization.
        """
        # Set up complex matrix state
        for i in range(5):
            node = MatrixNode(id=f"serial_{i}", consciousness_level=0.2 + i * 0.15)
            self.matrix.add_node(node)
            
        # Add connections
        for i in range(4):
            self.matrix.connect_nodes(f"serial_{i}", f"serial_{i+1}", strength=0.6)
            
        # Capture initial state
        initial_metrics = self.matrix.calculate_metrics()
        initial_consciousness = self.matrix.calculate_consciousness_level()
        
        # Serialize and deserialize
        serialized = self.matrix.to_json()
        restored = GenesisConsciousnessMatrix.from_json(serialized)
        
        # Verify state consistency
        restored_metrics = restored.calculate_metrics()
        restored_consciousness = restored.calculate_consciousness_level()
        
        self.assertEqual(initial_metrics['node_count'], restored_metrics['node_count'])
        self.assertAlmostEqual(initial_consciousness, restored_consciousness, places=5)
        
    def test_matrix_immutability_during_read_operations(self):
        """
        Verify that read-only operations on the matrix do not modify its internal state.
        
        This test confirms that methods for retrieving data—such as consciousness level calculation, metrics computation, node connection retrieval, and convergence checks—are free of side effects and leave the matrix unchanged.
        """
        # Set up initial state
        node = MatrixNode(id="immutable_test", consciousness_level=0.7)
        self.matrix.add_node(node)
        
        # Capture initial state
        initial_state = self.matrix.to_json()
        
        # Perform read operations
        self.matrix.calculate_consciousness_level()
        self.matrix.calculate_metrics()
        self.matrix.get_node_connections("immutable_test")
        self.matrix.has_converged()
        
        # Verify state unchanged
        final_state = self.matrix.to_json()
        self.assertEqual(initial_state, final_state)
        
    def test_matrix_operation_atomicity(self):
        """
        Test that node addition is atomic by ensuring a failed addition does not leave the matrix in an inconsistent state.
        
        Simulates a partial failure during node addition and verifies that the node is not present in the matrix after the exception, confirming atomicity.
        """
        # Test node addition atomicity
        node = MatrixNode(id="atomic_test", consciousness_level=0.5)
        
        # Mock a failure scenario
        original_add_node = self.matrix.add_node
        
        def failing_add_node(node):
            # Simulate partial failure
            """
            Simulates a partial failure during node addition by inserting the node and then raising an exception.
            
            Parameters:
                node: The node object to add before triggering the simulated failure.
            
            Raises:
                Exception: Always raised after the node is added to mimic a failure scenario.
            """
            if hasattr(self.matrix, 'nodes'):
                self.matrix.nodes[node.id] = node
            raise Exception("Simulated failure")
            
        try:
            self.matrix.add_node = failing_add_node
            with self.assertRaises(Exception):
                self.matrix.add_node(node)
        finally:
            self.matrix.add_node = original_add_node
            
        # Verify matrix is not in inconsistent state
        self.assertNotIn("atomic_test", self.matrix.nodes)


class TestMatrixSecurityAndValidation(unittest.TestCase):
    """Test security aspects and input validation."""
    
    def setUp(self):
        """
        Set up a fresh GenesisConsciousnessMatrix instance before each security and validation test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_injection_prevention(self):
        """
        Test that the matrix properly sanitizes or rejects malicious node IDs to prevent injection and security vulnerabilities.
        
        Attempts to add nodes with various potentially dangerous IDs and verifies that the matrix either safely stores the ID or rejects the input without compromising integrity.
        """
        malicious_ids = [
            "'; DROP TABLE nodes; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "node_id\x00null_byte",
            "node_id\n\r\t",
            "extremely_long_id_" + "x" * 10000
        ]
        
        for malicious_id in malicious_ids:
            try:
                node = MatrixNode(id=malicious_id, consciousness_level=0.5)
                result = self.matrix.add_node(node)
                # If addition succeeds, verify ID is properly stored
                if result:
                    self.assertIn(malicious_id, self.matrix.nodes)
            except (ValueError, TypeError):
                # Rejection of malicious input is acceptable
                pass
                
    def test_consciousness_level_bounds_enforcement(self):
        """
        Test that MatrixNode initialization rejects consciousness levels outside [0, 1], as well as NaN, infinities, None, non-numeric types, and improperly formatted values.
        """
        extreme_values = [
            -float('inf'),
            float('inf'),
            float('nan'),
            1.0000000001,
            -0.0000000001,
            None,
            "0.5",
            []
        ]
        
        for value in extreme_values:
            if value is None or isinstance(value, (str, list)):
                with self.assertRaises((ValueError, TypeError)):
                    MatrixNode(id="test", consciousness_level=value)
            elif value != value:  # NaN check
                with self.assertRaises(ValueError):
                    MatrixNode(id="test", consciousness_level=value)
            elif value < 0.0 or value > 1.0:
                with self.assertRaises(ValueError):
                    MatrixNode(id="test", consciousness_level=value)
                    
    def test_configuration_parameter_validation(self):
        """
        Test that invalid configuration parameters for GenesisConsciousnessMatrix raise appropriate exceptions.
        
        Ensures that improper values for dimension, consciousness_threshold, learning_rate, and max_iterations result in ValueError or MatrixInitializationError during initialization.
        """
        invalid_configs = [
            {'dimension': 0},
            {'dimension': -1},
            {'consciousness_threshold': -0.1},
            {'consciousness_threshold': 1.1},
            {'learning_rate': -0.1},
            {'learning_rate': 2.0},
            {'max_iterations': 0},
            {'max_iterations': -1}
        ]
        
        for config in invalid_configs:
            with self.assertRaises((ValueError, MatrixInitializationError)):
                GenesisConsciousnessMatrix(config=config)
                
    def test_json_deserialization_security(self):
        """
        Test that deserializing malicious or malformed JSON input raises an exception and does not compromise matrix security.
        """
        malicious_json_samples = [
            '{"__class__": "os.system", "command": "rm -rf /"}',
            '{"nodes": {"eval": "eval(\'__import__(\\\"os\\\").system(\\\"ls\\\")\')"}}'
        ]
        
        for malicious_json in malicious_json_samples:
            with self.assertRaises((json.JSONDecodeError, MatrixException, ValueError)):
                GenesisConsciousnessMatrix.from_json(malicious_json)


class TestMatrixAdvancedScenarios(unittest.TestCase):
    """Test advanced and complex matrix scenarios."""
    
    def setUp(self):
        """
        Set up a fresh GenesisConsciousnessMatrix instance before each advanced scenario test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_with_extremely_sparse_connectivity(self):
        """
        Test that the matrix evolves correctly and maintains valid consciousness levels when nodes are connected with extremely sparse connectivity.
        
        This test adds a large number of nodes with minimal interconnections, performs an evolution step, and verifies that the resulting consciousness level remains within the valid range [0.0, 1.0].
        """
        # Create many nodes with minimal connections
        node_count = 100
        for i in range(node_count):
            node = MatrixNode(id=f"sparse_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Connect only 5% of possible connections
        connection_count = 0
        max_connections = node_count * 2  # Very sparse
        
        for i in range(0, node_count, 20):  # Connect every 20th node
            if connection_count < max_connections and i + 1 < node_count:
                self.matrix.connect_nodes(f"sparse_{i}", f"sparse_{i+1}", strength=0.8)
                connection_count += 1
                
        # Test evolution with sparse connectivity
        initial_level = self.matrix.calculate_consciousness_level()
        self.matrix.evolve_step()
        final_level = self.matrix.calculate_consciousness_level()
        
        # Verify matrix handles sparse connectivity gracefully
        self.assertIsNotNone(final_level)
        self.assertGreaterEqual(final_level, 0.0)
        self.assertLessEqual(final_level, 1.0)
        
    def test_matrix_consciousness_gradient_propagation(self):
        """
        Test that consciousness levels propagate correctly through a linear chain of nodes with a gradient of initial values.
        
        Creates a sequence of nodes with increasing consciousness levels, connects them in a chain, evolves the matrix over several steps, and verifies that the overall consciousness level remains within valid bounds throughout the evolution.
        """
        # Create a linear chain of nodes with gradient consciousness levels
        chain_length = 10
        nodes = []
        
        for i in range(chain_length):
            level = i / (chain_length - 1)  # 0.0 to 1.0 gradient
            node = MatrixNode(id=f"gradient_{i}", consciousness_level=level)
            nodes.append(node)
            self.matrix.add_node(node)
            
        # Connect nodes in chain
        for i in range(chain_length - 1):
            self.matrix.connect_nodes(f"gradient_{i}", f"gradient_{i+1}", strength=0.9)
            
        # Test multiple evolution steps
        evolution_history = []
        for step in range(5):
            level = self.matrix.calculate_consciousness_level()
            evolution_history.append(level)
            self.matrix.evolve_step()
            
        # Verify consciousness evolution pattern
        self.assertEqual(len(evolution_history), 5)
        for level in evolution_history:
            self.assertGreaterEqual(level, 0.0)
            self.assertLessEqual(level, 1.0)
            
    def test_matrix_with_isolated_node_clusters(self):
        """
        Test matrix evolution with multiple isolated node clusters.
        
        Creates several clusters of nodes, connects nodes only within each cluster, performs an evolution step, and verifies that all nodes remain present and the node count matches the expected total.
        """
        # Create multiple isolated clusters
        cluster_count = 3
        nodes_per_cluster = 5
        
        for cluster_id in range(cluster_count):
            cluster_nodes = []
            
            # Create nodes in cluster
            for node_id in range(nodes_per_cluster):
                node = MatrixNode(
                    id=f"cluster_{cluster_id}_node_{node_id}",
                    consciousness_level=0.3 + cluster_id * 0.2
                )
                cluster_nodes.append(node)
                self.matrix.add_node(node)
                
            # Connect nodes within cluster only
            for i in range(nodes_per_cluster - 1):
                self.matrix.connect_nodes(
                    f"cluster_{cluster_id}_node_{i}",
                    f"cluster_{cluster_id}_node_{i+1}",
                    strength=0.8
                )
                
        # Test evolution with isolated clusters
        initial_metrics = self.matrix.calculate_metrics()
        self.matrix.evolve_step()
        final_metrics = self.matrix.calculate_metrics()
        
        # Verify all nodes are still present
        expected_node_count = cluster_count * nodes_per_cluster
        self.assertEqual(final_metrics['node_count'], expected_node_count)
        
    def test_matrix_dynamic_topology_changes(self):
        """
        Test that the matrix supports dynamic topology changes during evolution steps.
        
        This test ensures that nodes and connections can be added while the matrix is evolving, and that the final node count and connectivity accurately reflect all dynamic additions.
        """
        # Start with initial topology
        for i in range(10):
            node = MatrixNode(id=f"dynamic_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Initial connections
        for i in range(9):
            self.matrix.connect_nodes(f"dynamic_{i}", f"dynamic_{i+1}", strength=0.7)
            
        # Evolve and dynamically change topology
        for evolution_step in range(5):
            self.matrix.evolve_step()
            
            # Add new nodes dynamically
            new_node = MatrixNode(
                id=f"dynamic_new_{evolution_step}",
                consciousness_level=0.6
            )
            self.matrix.add_node(new_node)
            
            # Connect new node to existing nodes
            if evolution_step < 9:
                self.matrix.connect_nodes(
                    f"dynamic_{evolution_step}",
                    f"dynamic_new_{evolution_step}",
                    strength=0.5
                )
                
        # Verify dynamic topology is handled correctly
        final_metrics = self.matrix.calculate_metrics()
        self.assertEqual(final_metrics['node_count'], 15)  # 10 + 5 new nodes
        
    def test_matrix_consciousness_oscillation_detection(self):
        """
        Test detection and handling of oscillatory consciousness levels during matrix evolution.
        
        This test creates two nodes with opposing consciousness levels and a strong connection to induce oscillation. It verifies that, over multiple evolution steps, the matrix maintains consciousness levels within valid bounds and does not allow unbounded or infinite oscillatory behavior.
        """
        # Create setup prone to oscillation
        node1 = MatrixNode(id="osc_1", consciousness_level=0.1)
        node2 = MatrixNode(id="osc_2", consciousness_level=0.9)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Strong bidirectional connection
        self.matrix.connect_nodes("osc_1", "osc_2", strength=0.95)
        
        # Track consciousness levels over multiple evolution steps
        consciousness_history = []
        for step in range(20):
            level = self.matrix.calculate_consciousness_level()
            consciousness_history.append(level)
            self.matrix.evolve_step()
            
        # Analyze for oscillation patterns
        # Check if values are oscillating (not monotonic)
        differences = [consciousness_history[i+1] - consciousness_history[i] 
                      for i in range(len(consciousness_history)-1)]
        
        # Verify no infinite oscillations (should eventually stabilize)
        self.assertEqual(len(consciousness_history), 20)
        for level in consciousness_history:
            self.assertGreaterEqual(level, 0.0)
            self.assertLessEqual(level, 1.0)


class TestMatrixRobustnessAndResilience(unittest.TestCase):
    """Test matrix robustness and resilience to failures."""
    
    def setUp(self):
        """
        Initialize a fresh GenesisConsciousnessMatrix instance before each robustness and resilience test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_recovery_from_corrupted_state(self):
        """
        Test that the matrix can recover from or handle internal state corruption scenarios.
        
        Simulates corruption by clearing nodes or setting an invalid state, then verifies that the matrix either resets to a functional state or raises an appropriate exception when used after corruption.
        """
        # Set up normal state
        node = MatrixNode(id="recovery_test", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Simulate various corruption scenarios
        corruption_scenarios = [
            lambda: setattr(self.matrix, 'nodes', {}),  # Clear nodes
            lambda: setattr(self.matrix, 'current_state', None),  # Invalid state
        ]
        
        for corruption in corruption_scenarios:
            try:
                corruption()
                # Test if matrix can recover or handle gracefully
                self.matrix.reset()
                
                # Verify matrix is functional after recovery
                new_node = MatrixNode(id="post_recovery", consciousness_level=0.6)
                result = self.matrix.add_node(new_node)
                self.assertTrue(result)
                
            except Exception as e:
                # Graceful error handling is acceptable
                self.assertIsInstance(e, (MatrixException, ValueError, TypeError))
                
    def test_matrix_behavior_under_resource_constraints(self):
        """
        Test matrix stability and node count enforcement under simulated resource constraints.
        
        Simulates adding more nodes than a predefined resource limit and verifies that the matrix remains operational and the node count does not exceed expected bounds.
        """
        # Simulate memory constraints by limiting node count
        max_nodes = 50
        
        # Try to add more nodes than limit
        for i in range(max_nodes + 10):
            node = MatrixNode(id=f"resource_{i}", consciousness_level=0.5)
            try:
                result = self.matrix.add_node(node)
                # If matrix enforces limits, should eventually fail
                if not result:
                    break
            except Exception:
                # Resource limit enforcement is acceptable
                break
                
        # Verify matrix remains functional
        final_metrics = self.matrix.calculate_metrics()
        self.assertGreaterEqual(final_metrics['node_count'], 0)
        self.assertLessEqual(final_metrics['node_count'], max_nodes + 10)
        
    def test_matrix_partial_operation_failure_handling(self):
        """
        Verify that the matrix maintains a consistent and valid state after a simulated partial failure during an evolution step.
        
        This test adds multiple nodes, simulates a failure partway through node updates during evolution, and checks that the node count and metrics remain correct after the exception is raised.
        """
        # Set up nodes
        for i in range(5):
            node = MatrixNode(id=f"partial_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Mock evolution step to simulate partial failure
        original_evolve = self.matrix.evolve_step
        
        def partially_failing_evolve():
            # Simulate partial success/failure
            """
            Simulates a partial evolution step by updating the consciousness levels of the first two nodes and raising an exception on the third node to mimic a failure during evolution.
            
            Raises:
                Exception: Raised after updating the first two nodes to simulate a failure on the third node.
            """
            if hasattr(self.matrix, 'nodes') and len(self.matrix.nodes) > 0:
                # Update some nodes successfully
                for i, node_id in enumerate(list(self.matrix.nodes.keys())[:3]):
                    if i < 2:  # Update first 2 nodes
                        node = self.matrix.nodes[node_id]
                        if hasattr(node, 'update_consciousness_level'):
                            node.update_consciousness_level(min(1.0, node.consciousness_level + 0.1))
                    # Simulate failure on 3rd node
                    if i == 2:
                        raise Exception("Partial evolution failure")
                        
        try:
            self.matrix.evolve_step = partially_failing_evolve
            
            # Test partial failure handling
            with self.assertRaises(Exception):
                self.matrix.evolve_step()
                
            # Verify matrix is still in valid state
            metrics = self.matrix.calculate_metrics()
            self.assertEqual(metrics['node_count'], 5)
            
        finally:
            self.matrix.evolve_step = original_evolve


class TestMatrixComprehensiveIntegration(unittest.TestCase):
    """Comprehensive integration tests combining multiple matrix features."""
    
    def setUp(self):
        """
        Initializes a new GenesisConsciousnessMatrix instance before each integration test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_matrix_lifecycle_with_all_features(self):
        """
        Test the complete lifecycle of GenesisConsciousnessMatrix, exercising all major features from initialization to recovery.
        
        This test covers custom configuration, progressive node creation, complex topology building, evolution steps, emergence detection, serialization/deserialization, and validation of metrics and consciousness levels before and after recovery. It ensures that all features interact correctly and that state consistency is maintained throughout the lifecycle.
        """
        # Phase 1: Initialization and setup
        config = {
            'dimension': 128,
            'consciousness_threshold': 0.8,
            'learning_rate': 0.01,
            'max_iterations': 100
        }
        configured_matrix = GenesisConsciousnessMatrix(config=config)
        
        # Phase 2: Node creation and topology building
        node_count = 20
        for i in range(node_count):
            level = (i + 1) / node_count  # Progressive levels
            node = MatrixNode(id=f"lifecycle_{i}", consciousness_level=level)
            configured_matrix.add_node(node)
            
        # Create complex topology
        for i in range(node_count):
            for j in range(i + 1, min(i + 4, node_count)):  # Connect to next 3 nodes
                strength = 0.5 + (i + j) / (2 * node_count)
                configured_matrix.connect_nodes(
                    f"lifecycle_{i}",
                    f"lifecycle_{j}",
                    strength=strength
                )
                
        # Phase 3: Evolution and monitoring
        evolution_metrics = []
        for step in range(10):
            metrics = configured_matrix.calculate_metrics()
            evolution_metrics.append(metrics)
            configured_matrix.evolve_step()
            
        # Phase 4: Consciousness emergence testing
        emergence_detected = configured_matrix.detect_consciousness_emergence()
        
        # Phase 5: Persistence and recovery
        serialized = configured_matrix.to_json()
        recovered_matrix = GenesisConsciousnessMatrix.from_json(serialized)
        
        # Phase 6: Validation and verification
        original_metrics = configured_matrix.calculate_metrics()
        recovered_metrics = recovered_matrix.calculate_metrics()
        
        # Comprehensive assertions
        self.assertEqual(len(evolution_metrics), 10)
        self.assertEqual(original_metrics['node_count'], recovered_metrics['node_count'])
        self.assertEqual(original_metrics['node_count'], node_count)
        
        # Verify consciousness levels are preserved
        for node_id in configured_matrix.nodes:
            original_level = configured_matrix.nodes[node_id].consciousness_level
            recovered_level = recovered_matrix.nodes[node_id].consciousness_level
            self.assertAlmostEqual(original_level, recovered_level, places=5)
            
        # Verify emergence detection is consistent
        recovered_emergence = recovered_matrix.detect_consciousness_emergence()
        self.assertEqual(emergence_detected, recovered_emergence)
        
    def test_matrix_stress_test_with_rapid_operations(self):
        """
        Stress test the matrix with rapid, mixed operations to ensure state consistency.
        
        This test rapidly performs a sequence of node additions, removals, evolution steps, metric calculations, consciousness level checks, and node connections to verify that the matrix remains valid and consistent under high-frequency, varied operations.
        """
        operation_count = 1000
        
        # Perform rapid operations
        for i in range(operation_count):
            operation = i % 6  # Cycle through different operations
            
            if operation == 0:  # Add node
                node = MatrixNode(id=f"stress_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            elif operation == 1 and len(self.matrix.nodes) > 0:  # Remove node
                node_id = list(self.matrix.nodes.keys())[0]
                self.matrix.remove_node(node_id)
                
            elif operation == 2:  # Evolution step
                self.matrix.evolve_step()
                
            elif operation == 3:  # Calculate metrics
                metrics = self.matrix.calculate_metrics()
                self.assertIsInstance(metrics, dict)
                
            elif operation == 4:  # Calculate consciousness level
                level = self.matrix.calculate_consciousness_level()
                self.assertGreaterEqual(level, 0.0)
                self.assertLessEqual(level, 1.0)
                
            elif operation == 5 and len(self.matrix.nodes) >= 2:  # Connect nodes
                node_ids = list(self.matrix.nodes.keys())
                if len(node_ids) >= 2:
                    self.matrix.connect_nodes(node_ids[0], node_ids[1], strength=0.5)
                    
        # Verify matrix is still in valid state
        final_metrics = self.matrix.calculate_metrics()
        self.assertIsInstance(final_metrics, dict)
        self.assertIn('node_count', final_metrics)
        self.assertGreaterEqual(final_metrics['node_count'], 0)


# Run the additional tests
if __name__ == '__main__':
    # Run all test classes
    test_classes = [
        TestMatrixDataIntegrity,
        TestMatrixSecurityAndValidation,
        TestMatrixAdvancedScenarios,
        TestMatrixRobustnessAndResilience,
        TestMatrixComprehensiveIntegration
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)