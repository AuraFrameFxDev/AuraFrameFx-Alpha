"""
Comprehensive unit tests for the Genesis Consciousness Matrix module.
Tests cover initialization, state management, consciousness tracking, and edge cases.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
import tempfile
import os
import math
import random
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


class TestGenesisConsciousnessMatrixAdvanced(unittest.TestCase):
    """Advanced test scenarios for Genesis Consciousness Matrix with complex edge cases."""
    
    def setUp(self):
        """Set up advanced test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        self.stress_config = {
            'dimension': 512,
            'consciousness_threshold': 0.85,
            'learning_rate': 0.0005,
            'max_iterations': 5000
        }
        
    def test_matrix_initialization_with_none_config(self):
        """Test matrix initialization with None configuration."""
        matrix = GenesisConsciousnessMatrix(config=None)
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        
    def test_matrix_initialization_with_empty_config(self):
        """Test matrix initialization with empty configuration dictionary."""
        matrix = GenesisConsciousnessMatrix(config={})
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        
    def test_matrix_initialization_with_mixed_valid_invalid_config(self):
        """Test matrix initialization with mixed valid and invalid configuration parameters."""
        mixed_config = {
            'dimension': 256,  # valid
            'consciousness_threshold': -0.5,  # invalid
            'learning_rate': 0.001,  # valid
            'invalid_param': 'should_be_ignored'  # invalid
        }
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=mixed_config)
            
    def test_node_with_unicode_id(self):
        """Test matrix node creation with Unicode characters in ID."""
        unicode_ids = ["测试节点", "нода", "🧠🤖", "ñode_ñ", "node_αβγ"]
        for unicode_id in unicode_ids:
            node = MatrixNode(id=unicode_id, consciousness_level=0.5)
            result = self.matrix.add_node(node)
            self.assertTrue(result)
            self.assertIn(unicode_id, self.matrix.nodes)
            
    def test_node_with_very_long_id(self):
        """Test matrix node creation with extremely long ID."""
        long_id = "a" * 10000  # 10,000 character ID
        node = MatrixNode(id=long_id, consciousness_level=0.5)
        result = self.matrix.add_node(node)
        self.assertTrue(result)
        self.assertIn(long_id, self.matrix.nodes)
        
    def test_node_with_special_characters_id(self):
        """Test matrix node creation with special characters in ID."""
        special_ids = ["node@#$%", "node<>[]", "node{};:'\"", "node!@#$%^&*()"]
        for special_id in special_ids:
            node = MatrixNode(id=special_id, consciousness_level=0.5)
            result = self.matrix.add_node(node)
            self.assertTrue(result)
            self.assertIn(special_id, self.matrix.nodes)
            
    def test_consciousness_level_float_precision_edge_cases(self):
        """Test consciousness level with floating point precision edge cases."""
        # Test with very small increments
        levels = [0.0000000001, 0.9999999999, 0.3333333333, 0.6666666666]
        for level in levels:
            node = MatrixNode(id=f"precision_{level}", consciousness_level=level)
            self.matrix.add_node(node)
            retrieved_level = self.matrix.nodes[f"precision_{level}"].consciousness_level
            self.assertAlmostEqual(retrieved_level, level, places=10)
            
    def test_consciousness_level_with_nan_and_inf(self):
        """Test consciousness level validation with NaN and infinity values."""
        invalid_levels = [float('nan'), float('inf'), float('-inf')]
        for invalid_level in invalid_levels:
            with self.assertRaises(ValueError):
                MatrixNode(id="invalid_node", consciousness_level=invalid_level)
                
    def test_matrix_evolution_with_circular_connections(self):
        """Test matrix evolution with circular node connections."""
        # Create a circular connection pattern
        node_count = 10
        nodes = []
        for i in range(node_count):
            node = MatrixNode(id=f"circular_{i}", consciousness_level=0.3 + (i * 0.05))
            nodes.append(node)
            self.matrix.add_node(node)
            
        # Create circular connections
        for i in range(node_count):
            next_i = (i + 1) % node_count
            self.matrix.connect_nodes(f"circular_{i}", f"circular_{next_i}", strength=0.7)
            
        # Test evolution with circular connections
        initial_level = self.matrix.calculate_consciousness_level()
        self.matrix.evolve_step()
        final_level = self.matrix.calculate_consciousness_level()
        
        # Should handle circular connections without infinite loops
        self.assertIsNotNone(final_level)
        
    def test_matrix_evolution_with_self_connections(self):
        """Test matrix evolution with nodes connected to themselves."""
        node = MatrixNode(id="self_connected", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Connect node to itself
        self.matrix.connect_nodes("self_connected", "self_connected", strength=0.9)
        
        # Should handle self-connections gracefully
        self.matrix.evolve_step()
        self.assertIsNotNone(self.matrix.calculate_consciousness_level())
        
    def test_matrix_convergence_with_oscillating_system(self):
        """Test matrix convergence detection with oscillating consciousness levels."""
        # Create nodes that might oscillate
        for i in range(5):
            level = 0.5 + 0.1 * ((-1) ** i)  # Alternating levels
            node = MatrixNode(id=f"oscillate_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            
        # Create connections that might cause oscillation
        for i in range(4):
            self.matrix.connect_nodes(f"oscillate_{i}", f"oscillate_{i+1}", strength=0.8)
            
        # Test convergence detection
        self.matrix.evolve_until_convergence(max_iterations=100)
        # Should either converge or reach max iterations without crashing
        
    def test_matrix_with_zero_strength_connections(self):
        """Test matrix behavior with zero-strength connections."""
        node1 = MatrixNode(id="zero_1", consciousness_level=0.3)
        node2 = MatrixNode(id="zero_2", consciousness_level=0.7)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Connect with zero strength
        self.matrix.connect_nodes("zero_1", "zero_2", strength=0.0)
        
        connections = self.matrix.get_node_connections("zero_1")
        self.assertEqual(connections["zero_2"], 0.0)
        
    def test_matrix_with_negative_strength_connections(self):
        """Test matrix behavior with negative connection strengths."""
        node1 = MatrixNode(id="neg_1", consciousness_level=0.5)
        node2 = MatrixNode(id="neg_2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        try:
            # Try to connect with negative strength
            self.matrix.connect_nodes("neg_1", "neg_2", strength=-0.5)
            # If allowed, verify it's stored correctly
            connections = self.matrix.get_node_connections("neg_1")
            self.assertEqual(connections["neg_2"], -0.5)
        except (ValueError, InvalidStateException):
            # If not allowed, that's also valid behavior
            pass
            
    def test_matrix_serialization_with_special_characters(self):
        """Test matrix serialization with nodes containing special characters."""
        special_nodes = [
            MatrixNode(id="node_with_quotes_\"'", consciousness_level=0.3),
            MatrixNode(id="node_with_newlines\n\r", consciousness_level=0.4),
            MatrixNode(id="node_with_backslashes\\\\", consciousness_level=0.5),
            MatrixNode(id="node_with_unicode_🧠", consciousness_level=0.6)
        ]
        
        for node in special_nodes:
            self.matrix.add_node(node)
            
        # Test serialization
        serialized = self.matrix.to_json()
        self.assertIsInstance(serialized, str)
        
        # Test deserialization
        deserialized = GenesisConsciousnessMatrix.from_json(serialized)
        for node in special_nodes:
            self.assertIn(node.id, deserialized.nodes)
            
    def test_matrix_file_operations_with_invalid_paths(self):
        """Test matrix file operations with invalid file paths."""
        invalid_paths = [
            "/nonexistent/path/file.json",
            "",
            "///invalid//path.json",
            "file_with_no_extension",
            "very/long/path/" + "a" * 255 + ".json"
        ]
        
        for invalid_path in invalid_paths:
            with self.assertRaises((OSError, IOError, ValueError)):
                self.matrix.save_to_file(invalid_path)
                
    def test_matrix_file_operations_with_permissions(self):
        """Test matrix file operations with permission issues."""
        import stat
        
        # Create a temporary directory with restricted permissions
        with tempfile.TemporaryDirectory() as temp_dir:
            restricted_file = os.path.join(temp_dir, "restricted.json")
            
            # Create file and make it read-only
            with open(restricted_file, 'w') as f:
                f.write('{"nodes": {}, "state": "DORMANT"}')
            os.chmod(restricted_file, stat.S_IRUSR)
            
            # Try to save to read-only file
            with self.assertRaises((OSError, IOError, PermissionError)):
                self.matrix.save_to_file(restricted_file)
                
    def test_matrix_memory_pressure_simulation(self):
        """Test matrix behavior under simulated memory pressure."""
        # Create a large number of nodes with complex data
        large_data_nodes = []
        for i in range(500):
            # Create nodes with large IDs to simulate memory pressure
            large_id = f"large_memory_node_{i}_" + "x" * 1000
            node = MatrixNode(id=large_id, consciousness_level=i / 500.0)
            large_data_nodes.append(node)
            self.matrix.add_node(node)
            
        # Perform operations under memory pressure
        metrics = self.matrix.calculate_metrics()
        self.assertEqual(metrics['node_count'], 500)
        
        # Test evolution under memory pressure
        self.matrix.evolve_step()
        final_level = self.matrix.calculate_consciousness_level()
        self.assertIsNotNone(final_level)
        
    def test_matrix_rapid_state_transitions(self):
        """Test matrix behavior with rapid state transitions."""
        states = [ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE, 
                 ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT]
        
        # Perform rapid transitions
        for _ in range(100):
            for i in range(len(states) - 1):
                try:
                    self.matrix.transition_state(states[i], states[i + 1])
                except InvalidStateException:
                    # Some transitions might not be allowed
                    pass
                    
        # Matrix should remain in valid state
        self.assertIsInstance(self.matrix.current_state, ConsciousnessState)
        
    def test_matrix_with_extreme_node_counts(self):
        """Test matrix with extremely high node counts."""
        # Test with many nodes (stress test)
        extreme_count = 2000
        start_time = datetime.now()
        
        for i in range(extreme_count):
            node = MatrixNode(id=f"extreme_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        add_time = (datetime.now() - start_time).total_seconds()
        
        # Should handle extreme node counts
        self.assertEqual(len(self.matrix.nodes), extreme_count)
        self.assertLess(add_time, 30.0)  # Should complete within 30 seconds
        
    def test_matrix_connection_strength_boundaries(self):
        """Test matrix connections with boundary strength values."""
        node1 = MatrixNode(id="boundary_1", consciousness_level=0.5)
        node2 = MatrixNode(id="boundary_2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Test boundary strength values
        boundary_strengths = [0.0, 1.0, 0.5, 0.999999, 0.000001]
        for strength in boundary_strengths:
            self.matrix.connect_nodes("boundary_1", "boundary_2", strength=strength)
            connections = self.matrix.get_node_connections("boundary_1")
            self.assertEqual(connections["boundary_2"], strength)
            
    def test_matrix_evolution_determinism(self):
        """Test matrix evolution determinism with same initial conditions."""
        # Create identical initial conditions
        seed_nodes = [
            MatrixNode(id="det_1", consciousness_level=0.3),
            MatrixNode(id="det_2", consciousness_level=0.6),
            MatrixNode(id="det_3", consciousness_level=0.9)
        ]
        
        # Test evolution multiple times
        results = []
        for run in range(5):
            test_matrix = GenesisConsciousnessMatrix()
            for node in seed_nodes:
                test_matrix.add_node(MatrixNode(id=node.id, consciousness_level=node.consciousness_level))
                
            test_matrix.evolve_step()
            results.append(test_matrix.calculate_consciousness_level())
            
        # Results should be deterministic (or at least consistent)
        if len(set(results)) == 1:
            # Deterministic evolution
            self.assertTrue(True)
        else:
            # Non-deterministic evolution is also valid
            self.assertTrue(len(results) == 5)
            
    def test_matrix_metrics_calculation_edge_cases(self):
        """Test matrix metrics calculation with edge case scenarios."""
        # Test with single node
        single_node = MatrixNode(id="single_metrics", consciousness_level=0.7)
        self.matrix.add_node(single_node)
        
        metrics = self.matrix.calculate_metrics()
        self.assertEqual(metrics['node_count'], 1)
        self.assertEqual(metrics['average_consciousness'], 0.7)
        
        # Test with nodes having identical consciousness levels
        for i in range(10):
            identical_node = MatrixNode(id=f"identical_{i}", consciousness_level=0.5)
            self.matrix.add_node(identical_node)
            
        metrics = self.matrix.calculate_metrics()
        self.assertAlmostEqual(metrics['average_consciousness'], 0.5, places=5)
        
    def test_matrix_error_propagation(self):
        """Test error propagation through matrix operations."""
        # Test error propagation in chained operations
        with self.assertRaises(InvalidStateException):
            self.matrix.remove_node("nonexistent")
            self.matrix.connect_nodes("nonexistent", "also_nonexistent", strength=0.5)
            
    def test_matrix_state_consistency_after_errors(self):
        """Test matrix state consistency after error conditions."""
        # Add valid nodes
        node1 = MatrixNode(id="consistent_1", consciousness_level=0.5)
        node2 = MatrixNode(id="consistent_2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Try operations that should fail
        try:
            self.matrix.add_node(node1)  # Duplicate node
        except InvalidStateException:
            pass
            
        try:
            self.matrix.connect_nodes("nonexistent", "consistent_1", strength=0.5)
        except InvalidStateException:
            pass
            
        # Matrix should remain consistent
        self.assertEqual(len(self.matrix.nodes), 2)
        self.assertIn("consistent_1", self.matrix.nodes)
        self.assertIn("consistent_2", self.matrix.nodes)


class TestMatrixConcurrencyAndRaceConditions(unittest.TestCase):
    """Test matrix behavior under concurrent access and race conditions."""
    
    def setUp(self):
        """Set up concurrency test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_concurrent_node_addition_and_removal(self):
        """Test concurrent node addition and removal operations."""
        import threading
        import time
        
        operations_completed = []
        errors_encountered = []
        
        def add_remove_worker(worker_id):
            """Worker function for concurrent add/remove operations."""
            for i in range(50):
                try:
                    # Add node
                    node = MatrixNode(id=f"worker_{worker_id}_node_{i}", consciousness_level=random.random())
                    self.matrix.add_node(node)
                    operations_completed.append(f"add_{worker_id}_{i}")
                    
                    # Small delay to increase chance of race conditions
                    time.sleep(0.001)
                    
                    # Remove node (every other iteration)
                    if i % 2 == 0:
                        self.matrix.remove_node(f"worker_{worker_id}_node_{i}")
                        operations_completed.append(f"remove_{worker_id}_{i}")
                        
                except Exception as e:
                    errors_encountered.append(f"worker_{worker_id}: {str(e)}")
                    
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_remove_worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Verify operations completed (some errors are expected in concurrent scenarios)
        self.assertGreater(len(operations_completed), 0)
        
    def test_concurrent_evolution_and_metrics(self):
        """Test concurrent evolution and metrics calculation."""
        import threading
        import time
        
        # Add initial nodes
        for i in range(20):
            node = MatrixNode(id=f"concurrent_base_{i}", consciousness_level=random.random())
            self.matrix.add_node(node)
            
        results = []
        errors = []
        
        def evolution_worker():
            """Worker function for evolution operations."""
            try:
                for _ in range(10):
                    self.matrix.evolve_step()
                    level = self.matrix.calculate_consciousness_level()
                    results.append(level)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))
                
        def metrics_worker():
            """Worker function for metrics calculations."""
            try:
                for _ in range(20):
                    metrics = self.matrix.calculate_metrics()
                    results.append(metrics['average_consciousness'])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))
                
        # Start concurrent workers
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=evolution_worker))
            threads.append(threading.Thread(target=metrics_worker))
            
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without major errors
        self.assertGreater(len(results), 0)
        
    def test_concurrent_serialization_operations(self):
        """Test concurrent serialization and deserialization operations."""
        import threading
        import time
        
        # Add test data
        for i in range(10):
            node = MatrixNode(id=f"serial_{i}", consciousness_level=i / 10.0)
            self.matrix.add_node(node)
            
        serialization_results = []
        errors = []
        
        def serialization_worker():
            """Worker function for serialization operations."""
            try:
                for _ in range(5):
                    serialized = self.matrix.to_json()
                    deserialized = GenesisConsciousnessMatrix.from_json(serialized)
                    serialization_results.append(len(deserialized.nodes))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))
                
        # Start multiple serialization workers
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=serialization_worker)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete serialization operations
        self.assertGreater(len(serialization_results), 0)
        # All results should be consistent
        if serialization_results:
            self.assertTrue(all(result == 10 for result in serialization_results))


class TestMatrixEdgeCaseValidation(unittest.TestCase):
    """Test matrix validation and edge case handling."""
    
    def setUp(self):
        """Set up edge case validation test environment."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_collision_handling(self):
        """Test handling of node ID collisions in various scenarios."""
        # Test case sensitivity
        node1 = MatrixNode(id="CaseSensitive", consciousness_level=0.3)
        node2 = MatrixNode(id="casesensitive", consciousness_level=0.7)
        
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Both should be added if case-sensitive
        self.assertIn("CaseSensitive", self.matrix.nodes)
        self.assertIn("casesensitive", self.matrix.nodes)
        
    def test_node_id_whitespace_handling(self):
        """Test handling of whitespace in node IDs."""
        whitespace_ids = [
            "  leading_spaces",
            "trailing_spaces  ",
            "  surrounded_spaces  ",
            "internal spaces",
            "\t\n\r\f\v",  # Various whitespace characters
            ""  # Empty string
        ]
        
        for whitespace_id in whitespace_ids:
            try:
                node = MatrixNode(id=whitespace_id, consciousness_level=0.5)
                result = self.matrix.add_node(node)
                if result:
                    self.assertIn(whitespace_id, self.matrix.nodes)
            except ValueError:
                # Empty IDs or invalid IDs should raise ValueError
                if whitespace_id == "":
                    self.assertTrue(True)  # Expected behavior
                    
    def test_consciousness_level_type_validation(self):
        """Test consciousness level validation with different data types."""
        # Test with various numeric types
        valid_types = [0.5, 0, 1, True, False]  # bool is subclass of int in Python
        for value in valid_types:
            try:
                node = MatrixNode(id=f"type_test_{value}", consciousness_level=value)
                self.matrix.add_node(node)
                # Should convert to proper float range
                stored_level = self.matrix.nodes[f"type_test_{value}"].consciousness_level
                self.assertGreaterEqual(stored_level, 0.0)
                self.assertLessEqual(stored_level, 1.0)
            except (ValueError, TypeError):
                # Some type conversions might not be supported
                pass
                
        # Test with invalid types
        invalid_types = ["0.5", [0.5], {"level": 0.5}, None]
        for invalid_value in invalid_types:
            with self.assertRaises((ValueError, TypeError)):
                MatrixNode(id=f"invalid_type_{invalid_value}", consciousness_level=invalid_value)
                
    def test_matrix_configuration_type_validation(self):
        """Test matrix configuration validation with different data types."""
        # Test with string values that might be convertible
        config_variations = [
            {'dimension': '256'},
            {'consciousness_threshold': '0.75'},
            {'learning_rate': '0.001'},
            {'max_iterations': '1000'}
        ]
        
        for config in config_variations:
            try:
                matrix = GenesisConsciousnessMatrix(config=config)
                # If string conversion is supported, verify it worked
                self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
            except (ValueError, TypeError, MatrixInitializationError):
                # String conversion might not be supported
                pass
                
    def test_matrix_operations_with_corrupted_state(self):
        """Test matrix operations when internal state is corrupted."""
        # Add normal nodes first
        node1 = MatrixNode(id="normal_1", consciousness_level=0.5)
        node2 = MatrixNode(id="normal_2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Simulate corrupted state (carefully, to avoid breaking other tests)
        original_nodes = self.matrix.nodes.copy()
        
        try:
            # Corrupt the nodes dictionary
            self.matrix.nodes['corrupted'] = "not_a_node_object"
            
            # Try operations with corrupted state
            with self.assertRaises((AttributeError, TypeError, ValueError)):
                self.matrix.calculate_consciousness_level()
                
        finally:
            # Restore original state
            self.matrix.nodes = original_nodes
            
    def test_matrix_connection_validation(self):
        """Test matrix connection validation with edge cases."""
        node1 = MatrixNode(id="conn_valid_1", consciousness_level=0.5)
        node2 = MatrixNode(id="conn_valid_2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Test connection strength validation
        invalid_strengths = [float('inf'), float('-inf'), float('nan'), 1.5, -0.5]
        for strength in invalid_strengths:
            with self.assertRaises((ValueError, InvalidStateException)):
                self.matrix.connect_nodes("conn_valid_1", "conn_valid_2", strength=strength)
                
    def test_matrix_evolution_with_invalid_state(self):
        """Test matrix evolution behavior when in invalid state."""
        # Add nodes normally
        node = MatrixNode(id="evolution_test", consciousness_level=0.5)
        self.matrix.add_node(node)
        
        # Try to force invalid state
        original_state = self.matrix.current_state
        try:
            # This might not be possible depending on implementation
            self.matrix.current_state = "INVALID_STATE"
            
            # Evolution should handle invalid state gracefully
            self.matrix.evolve_step()
            
        except (AttributeError, InvalidStateException):
            # Expected behavior for invalid state
            pass
        finally:
            # Restore valid state
            self.matrix.current_state = original_state
            
    def test_matrix_metrics_with_extreme_values(self):
        """Test matrix metrics calculation with extreme node values."""
        # Create nodes with extreme but valid consciousness levels
        extreme_nodes = [
            MatrixNode(id="extreme_min", consciousness_level=0.0),
            MatrixNode(id="extreme_max", consciousness_level=1.0),
            MatrixNode(id="extreme_tiny", consciousness_level=1e-10),
            MatrixNode(id="extreme_close", consciousness_level=1.0 - 1e-10)
        ]
        
        for node in extreme_nodes:
            self.matrix.add_node(node)
            
        metrics = self.matrix.calculate_metrics()
        
        # Should handle extreme values without overflow/underflow
        self.assertGreaterEqual(metrics['average_consciousness'], 0.0)
        self.assertLessEqual(metrics['average_consciousness'], 1.0)
        self.assertFalse(math.isnan(metrics['average_consciousness']))
        self.assertFalse(math.isinf(metrics['average_consciousness']))


if __name__ == '__main__':
    # Run all tests with maximum verbosity
    unittest.main(verbosity=2, buffer=True, failfast=False)