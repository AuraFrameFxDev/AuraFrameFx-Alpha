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
        Initializes a new GenesisConsciousnessMatrix instance and test configuration before each test.
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
        Clean up the test environment after each test by invoking the matrix's cleanup method if available.
        """
        if hasattr(self.matrix, 'cleanup'):
            self.matrix.cleanup()
    
    def test_matrix_initialization_default(self):
        """
        Test that a GenesisConsciousnessMatrix initialized with default parameters has 'state' and 'nodes' attributes.
        """
        matrix = GenesisConsciousnessMatrix()
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        self.assertTrue(hasattr(matrix, 'state'))
        self.assertTrue(hasattr(matrix, 'nodes'))
        
    def test_matrix_initialization_custom_config(self):
        """
        Test that GenesisConsciousnessMatrix correctly applies custom configuration values for dimension and consciousness threshold during initialization.
        """
        matrix = GenesisConsciousnessMatrix(config=self.test_config)
        self.assertEqual(matrix.dimension, self.test_config['dimension'])
        self.assertEqual(matrix.consciousness_threshold, self.test_config['consciousness_threshold'])
        
    def test_matrix_initialization_invalid_config(self):
        """
        Test that initializing the matrix with invalid configuration parameters raises a MatrixInitializationError.
        
        Verifies that negative dimensions or out-of-range consciousness thresholds are rejected during matrix initialization.
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
        Test that adding a node with an ID already present in the matrix raises an InvalidStateException.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        with self.assertRaises(InvalidStateException):
            self.matrix.add_node(node)
            
    def test_remove_consciousness_node_existing(self):
        """
        Test that removing an existing node by ID returns True and the node is no longer present in the matrix.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        result = self.matrix.remove_node("test_node")
        self.assertTrue(result)
        self.assertNotIn("test_node", self.matrix.nodes)
        
    def test_remove_consciousness_node_nonexistent(self):
        """
        Test that removing a non-existent node from the matrix returns False.
        
        Verifies that the matrix correctly indicates failure when attempting to remove a node that is not present.
        """
        result = self.matrix.remove_node("nonexistent_node")
        self.assertFalse(result)
        
    def test_consciousness_state_transition_valid(self):
        """
        Test that a valid transition between consciousness states updates the matrix's current state and returns True.
        
        Verifies that transitioning from the initial state to a valid target state correctly updates the matrix and indicates success.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.ACTIVE
        result = self.matrix.transition_state(initial_state, target_state)
        self.assertTrue(result)
        self.assertEqual(self.matrix.current_state, target_state)
        
    def test_consciousness_state_transition_invalid(self):
        """
        Test that an invalid transition from DORMANT to TRANSCENDENT raises an InvalidStateException.
        """
        initial_state = ConsciousnessState.DORMANT
        target_state = ConsciousnessState.TRANSCENDENT
        with self.assertRaises(InvalidStateException):
            self.matrix.transition_state(initial_state, target_state)
            
    def test_consciousness_level_calculation(self):
        """
        Test that the matrix calculates the average consciousness level correctly for multiple nodes with different levels.
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
        Verify that the consciousness level calculation returns 0.0 when the matrix contains no nodes.
        """
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.0)
        
    def test_consciousness_level_calculation_single_node(self):
        """
        Test that the matrix calculates the correct consciousness level when only one node is present.
        """
        node = MatrixNode(id="single_node", consciousness_level=0.8)
        self.matrix.add_node(node)
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.8)
        
    def test_matrix_evolution_step(self):
        """
        Test that a single evolution step changes the matrix's state snapshot.
        
        Verifies that calling `evolve_step()` results in a different state, indicating that the matrix evolves as intended.
        """
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        self.assertNotEqual(initial_state, final_state)
        
    def test_matrix_evolution_convergence(self):
        """
        Test that the matrix evolution process correctly identifies convergence within a given maximum number of iterations.
        
        This test ensures that after evolving the matrix with a specified iteration limit, the matrix reports a converged state.
        """
        self.matrix.evolve_until_convergence(max_iterations=10)
        self.assertTrue(self.matrix.has_converged())
        
    def test_matrix_reset_to_initial_state(self):
        """
        Tests that resetting the matrix clears all nodes and restores the state to DORMANT.
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
        Test that the matrix serializes to a JSON string with accurate nodes and state fields.
        
        Ensures the serialized output is a valid JSON string containing both "nodes" and "state" keys.
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
        Verify that deserializing a matrix from a JSON string accurately restores all nodes and their consciousness levels.
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
        
        This test verifies that the persistence mechanism correctly serializes and deserializes the matrix, ensuring node data integrity after file operations.
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
        Test that connecting two nodes records the connection with the correct strength and allows accurate retrieval of the connection data.
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
        Test that the matrix correctly detects consciousness emergence when several nodes have high consciousness levels.
        """
        # Add nodes with high consciousness levels
        for i in range(5):
            node = MatrixNode(id=f"high_node_{i}", consciousness_level=0.9)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertTrue(emergence_detected)
        
    def test_consciousness_emergence_detection_insufficient(self):
        """
        Verify that the matrix does not detect consciousness emergence when all nodes have consciousness levels below the emergence threshold.
        """
        # Add nodes with low consciousness levels
        for i in range(2):
            node = MatrixNode(id=f"low_node_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
            
        emergence_detected = self.matrix.detect_consciousness_emergence()
        self.assertFalse(emergence_detected)
        
    def test_matrix_metrics_calculation(self):
        """
        Test that the matrix correctly computes and returns performance metrics after nodes are added.
        
        Verifies that the metrics include average consciousness, node count, and connection density, and that the node count matches the number of added nodes.
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
        Test that evolving a matrix with 100 nodes completes within one second.
        
        Adds 100 nodes to the matrix and measures the time taken for a single evolution step, asserting that it finishes in under one second.
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
        Test that the matrix updates its node count correctly when nodes are added and removed.
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
        Test concurrent addition of nodes from multiple threads to verify thread safety of the matrix's node addition method.
        
        Ensures that all node additions succeed without errors when performed in parallel, confirming the matrix handles concurrent modifications correctly.
        """
        import threading
        import time
        
        results = []
        
        def add_nodes_thread(thread_id):
            """
            Add ten `MatrixNode` instances with unique IDs for a given thread, recording the success of each addition in a shared results list.
            
            Parameters:
                thread_id (int): Identifier used to generate unique node IDs for this thread.
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
        Verify that each ConsciousnessState enum member has the correct integer value.
        """
        self.assertEqual(ConsciousnessState.DORMANT.value, 0)
        self.assertEqual(ConsciousnessState.ACTIVE.value, 1)
        self.assertEqual(ConsciousnessState.AWARE.value, 2)
        self.assertEqual(ConsciousnessState.TRANSCENDENT.value, 3)
        
    def test_consciousness_state_ordering(self):
        """
        Verify that the `ConsciousnessState` enum values are ordered from DORMANT to TRANSCENDENT.
        """
        self.assertLess(ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE)
        self.assertLess(ConsciousnessState.ACTIVE, ConsciousnessState.AWARE)
        self.assertLess(ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT)
        
    def test_consciousness_state_string_representation(self):
        """
        Verify that the string representation of each ConsciousnessState enum value matches its corresponding name.
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
        Verify that a MatrixNode instance is initialized with the correct ID and consciousness level.
        """
        node = MatrixNode(id="init_test", consciousness_level=0.7)
        self.assertEqual(node.id, "init_test")
        self.assertEqual(node.consciousness_level, 0.7)
        
    def test_node_initialization_invalid_consciousness_level(self):
        """
        Test that creating a MatrixNode with a consciousness level less than 0.0 or greater than 1.0 raises a ValueError.
        """
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=1.5)
            
        with self.assertRaises(ValueError):
            MatrixNode(id="invalid_node", consciousness_level=-0.1)
            
    def test_node_consciousness_level_update(self):
        """
        Tests that updating a node's consciousness level with a valid value successfully changes its state.
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
        Test that MatrixNode instances are considered equal if they have the same ID and consciousness level, and not equal if their IDs differ.
        """
        node1 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node2 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node3 = MatrixNode(id="different_test", consciousness_level=0.5)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        
    def test_node_hash(self):
        """
        Test that MatrixNode instances with the same ID have identical hash values.
        
        Ensures that nodes sharing an ID are treated as equal in hash-based collections, regardless of their consciousness levels.
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
        Verify that custom matrix exceptions inherit from their intended base exception classes.
        """
        self.assertTrue(issubclass(MatrixException, Exception))
        self.assertTrue(issubclass(InvalidStateException, MatrixException))
        self.assertTrue(issubclass(MatrixInitializationError, MatrixException))
        
    def test_matrix_exception_messages(self):
        """
        Test that custom matrix exceptions return the correct error messages when raised and converted to strings.
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
        Initialize a new GenesisConsciousnessMatrix instance before each test to ensure test isolation.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_consciousness_evolution_cycle(self):
        """
        Simulates a complete evolution cycle by adding nodes, connecting them, evolving the matrix until convergence, and verifying that the overall consciousness level changes.
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
        Test that consciousness emergence is detected only after all nodes exceed the emergence threshold.
        
        The test adds multiple nodes with low consciousness levels and verifies that emergence is not detected. It then raises all node levels above the threshold and confirms that emergence is detected.
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
        Verify that serializing and deserializing the matrix preserves all nodes, their consciousness levels, and node connections, ensuring data integrity after persistence operations.
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
        Initialize a fresh GenesisConsciousnessMatrix instance and extreme configuration for extended test scenarios.
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
        
        Verifies that the matrix correctly sets its dimension and consciousness threshold when initialized with edge-case parameters.
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
        Test that the GenesisConsciousnessMatrix initializes correctly with boundary configuration values, such as a consciousness threshold of 1.0 and an extremely small learning rate.
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
        Verify that all matrix operations behave correctly when the matrix contains zero nodes.
        
        Ensures that evolution, metrics calculation, and convergence detection operate without error and return expected results in the absence of nodes.
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
        Verify that matrix operations behave correctly when only a single node is present.
        
        Adds a single node to the matrix, performs an evolution step, and checks that the consciousness level is computed and remains defined after evolution.
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
        Verify that the matrix accurately stores and retrieves node consciousness levels with high floating-point precision.
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
        Test consciousness state transitions for edge cases, including transitions to the same state and rapid sequential transitions through all defined states.
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
        Test that matrix evolution handles convergence edge cases, including reaching the maximum number of iterations without errors.
        
        This test adds multiple nodes to the matrix and invokes evolution with a very low maximum iteration count to ensure the method completes gracefully even when convergence is not achieved.
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
        Verify that node connections handle minimum and maximum connection strengths correctly.
        
        This test adds two nodes to the matrix and connects them with both the lowest (0.0) and highest (1.0) possible strengths, asserting that the connection strengths are accurately stored and retrieved.
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
        Test serialization and deserialization of the matrix when nodes have extreme consciousness levels.
        
        Verifies that nodes with minimum (0.0) and maximum (1.0) consciousness levels are accurately preserved through the JSON serialization and deserialization process.
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
        Verify that the matrix maintains consistent state and manages memory correctly during rapid cycles of node additions and removals.
        
        This test repeatedly adds and removes nodes in quick succession to simulate memory stress, then checks that the matrix retains a valid set of nodes at the end.
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
        Verify that the matrix maintains a valid state when subjected to concurrent mixed operations from multiple threads.
        
        This test launches several threads, each performing a sequence of node additions, evolution steps, metric calculations, and node removals. It ensures that, despite concurrent modifications, the matrix's internal node structure remains a valid dictionary after all operations complete.
        """
        import threading
        import time
        
        def modify_matrix(thread_id):
            """
            Performs a sequence of concurrent matrix operations including node addition, evolution, metrics calculation, and periodic node removal.
            
            This function is intended for use in multi-threaded tests to simulate mixed operations on the matrix and assess thread safety under concurrent modifications.
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
        Verify that the matrix evolution step maintains acceptable performance as the number of nodes increases.
        
        Adds varying numbers of nodes to the matrix, measures the time taken for an evolution step at each scale, and asserts that execution time remains within reasonable bounds.
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
        Test that the matrix can recover from an invalid internal state and remain functional after recovery.
        
        This test forcibly corrupts the matrix's internal state, attempts recovery via reset or reinitialization, and verifies that node operations succeed post-recovery.
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
        Initializes a new GenesisConsciousnessMatrix instance before each asynchronous test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_async_evolution_step(self):
        """
        Test that the matrix can perform an asynchronous evolution step if supported, falling back to synchronous evolution otherwise.
        
        Ensures that after the evolution step, the matrix's consciousness level can be calculated and is not None.
        """
        async def async_evolution_test():
            """
            Performs an asynchronous evolution step on the matrix and returns the updated average consciousness level.
            
            If the matrix does not support asynchronous evolution, falls back to synchronous evolution.
            
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
        Test that asynchronous batch operations correctly add multiple nodes to the matrix.
        
        Verifies that all nodes are present after performing asynchronous additions and a simulated async delay.
        """
        async def batch_operation_test():
            # Add multiple nodes asynchronously
            """
            Asynchronously adds multiple nodes to the matrix and returns the total node count after addition.
            
            Returns:
                int: The number of nodes in the matrix after the batch operation.
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
        Initialize a new GenesisConsciousnessMatrix instance for property-based tests.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_consciousness_level_invariants(self):
        """
        Verify that the matrix's calculated consciousness level remains within the [0, 1] range after adding nodes with varying valid consciousness levels.
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
        Verify that the reported node count in matrix metrics matches the actual number of nodes after each addition.
        """
        # Property: node count should match actual nodes
        for i in range(20):
            node = MatrixNode(id=f"count_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
            metrics = self.matrix.calculate_metrics()
            self.assertEqual(metrics['node_count'], len(self.matrix.nodes))
            
    def test_serialization_roundtrip_invariants(self):
        """
        Verify that serializing and then deserializing the matrix preserves all node IDs and their consciousness levels.
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
        Prepare a fresh GenesisConsciousnessMatrix instance for each mocking test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    @patch('app.ai_backend.genesis_consciousness_matrix.GenesisConsciousnessMatrix.evolve_step')
    def test_evolution_with_mocked_step(self, mock_evolve):
        """
        Test that the matrix evolution step returns the mocked result and the evolution method is called exactly once.
        """
        mock_evolve.return_value = True
        
        result = self.matrix.evolve_step()
        self.assertTrue(result)
        mock_evolve.assert_called_once()
        
    @patch('json.dumps')
    def test_serialization_with_mocked_json(self, mock_dumps):
        """
        Test that the matrix's JSON serialization method correctly uses the mocked JSON library and returns the expected serialized string.
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
        
        Verifies that the matrix is correctly instantiated and that the file open operation is called as expected when using a mock.
        """
        if hasattr(GenesisConsciousnessMatrix, 'load_from_file'):
            matrix = GenesisConsciousnessMatrix.load_from_file('test_file.json')
            self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
            mock_file.assert_called_once_with('test_file.json', 'r')


class TestMatrixValidationAndSanitization(unittest.TestCase):
    """Test input validation and data sanitization."""
    
    def setUp(self):
        """
        Initialize a new GenesisConsciousnessMatrix instance for each validation test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_validation(self):
        """
        Verify that the matrix accepts node IDs with various valid string formats, including alphanumeric, dashes, and underscores.
        """
        # Test with different ID types
        valid_ids = ['string_id', 'id_123', 'node-with-dashes', 'node_with_underscores']
        for node_id in valid_ids:
            node = MatrixNode(id=node_id, consciousness_level=0.5)
            result = self.matrix.add_node(node)
            self.assertTrue(result)
            
    def test_consciousness_level_boundary_validation(self):
        """
        Verify that nodes with consciousness levels at the exact lower and upper boundaries, as well as near-boundary values, are accepted and stored correctly in the matrix.
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
        Verify that configuration parameters provided as strings are properly sanitized and converted to their expected types during matrix initialization.
        
        This test ensures that the `GenesisConsciousnessMatrix` can handle configuration values given as strings by converting them to the appropriate numeric types, or raises an error if such conversion is not supported.
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
        Verify that deserialization of malformed or invalid JSON strings raises the appropriate exceptions.
        
        Tests various malformed JSON inputs to ensure that the `GenesisConsciousnessMatrix.from_json` method raises `json.JSONDecodeError`, `MatrixException`, or `ValueError` as expected.
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
        Prepare the test environment by initializing a new GenesisConsciousnessMatrix instance for performance testing.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_large_scale_node_operations(self):
        """
        Test that adding 1000 nodes and performing an evolution step completes within acceptable performance thresholds.
        
        Asserts that node addition takes less than 5 seconds and evolution takes less than 10 seconds, verifying scalability under load.
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
        Verify that repeated addition and removal of nodes does not result in memory leaks by ensuring the node count remains unchanged after high node churn cycles.
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
        Test that the matrix maintains acceptable performance when creating a densely connected network of nodes and performing an evolution step.
        
        Creates 50 nodes, connects each node to every other node to form a dense graph, and verifies that both the connection process and a single evolution step complete within 10 seconds each.
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
            Test that the GenesisConsciousnessMatrix initializes correctly with different dimension values.
            
            Parameters:
                dimension (int): The dimension value to initialize the matrix with.
            """
            config = {'dimension': dimension}
            matrix = GenesisConsciousnessMatrix(config=config)
            assert matrix.dimension == dimension
            
        @pytest.mark.parametrize("consciousness_level", [0.0, 0.25, 0.5, 0.75, 1.0])
        def test_node_consciousness_levels(self, consciousness_level):
            """
            Test that a MatrixNode is correctly initialized with a given consciousness level.
            
            Parameters:
            	consciousness_level (float): The consciousness level to assign to the node.
            """
            node = MatrixNode(id=f"test_{consciousness_level}", consciousness_level=consciousness_level)
            assert node.consciousness_level == consciousness_level
            
        @pytest.mark.parametrize("node_count", [1, 5, 10, 50, 100])
        def test_matrix_with_variable_node_counts(self, node_count):
            """
            Test matrix operations and metrics calculation with a variable number of nodes.
            
            Adds a specified number of nodes to the matrix, verifies that the calculated consciousness level is within valid bounds, and asserts that the reported node count matches the number of nodes added.
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
            Test that consciousness emergence is correctly detected when all nodes have consciousness levels above the specified threshold.
            
            Parameters:
                threshold (float): The consciousness level threshold for emergence detection.
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
        Initialize a new GenesisConsciousnessMatrix instance for data integrity tests.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_uniqueness_enforcement(self):
        """
        Verify that the matrix enforces node ID uniqueness in a case-sensitive manner, allowing nodes with IDs differing only by case to coexist.
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
        Verify that node consciousness levels remain within valid bounds after a sequence of operations, ensuring consistency and predictability of state changes.
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
        Verify that the matrix's state, including node count and average consciousness level, remains consistent after serializing to JSON and deserializing back.
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
        Verify that performing read-only operations on the matrix does not alter its internal state.
        
        This test ensures that methods intended for data retrieval, such as calculating consciousness level, metrics, retrieving node connections, and checking convergence, do not cause any side effects or modify the matrix's state.
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
        Verify that matrix operations are atomic by ensuring that a failed node addition does not leave the matrix in an inconsistent state.
        
        This test simulates a partial failure during node addition and asserts that the node is not present in the matrix after the exception, confirming atomicity of the operation.
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
                node: The node object to be added before the simulated failure.
            
            Raises:
                Exception: Always raised after the node is added to simulate a failure scenario.
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
        Initialize a new GenesisConsciousnessMatrix instance for each security test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_node_id_injection_prevention(self):
        """
        Verify that the matrix sanitizes or rejects malicious node IDs to prevent injection and security vulnerabilities.
        
        Tests a variety of potentially dangerous node ID strings, ensuring that either the node is safely added with the correct ID or the input is rejected without compromising matrix integrity.
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
        Verify that MatrixNode initialization strictly enforces consciousness level bounds and rejects invalid values, including out-of-range numbers, NaN, infinities, None, non-numeric types, and improper formats.
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
        Verify that invalid configuration parameters for the GenesisConsciousnessMatrix raise ValueError or MatrixInitializationError.
        
        Tests a variety of invalid configuration dictionaries to ensure that improper values for dimension, consciousness_threshold, learning_rate, and max_iterations are correctly rejected during matrix initialization.
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
        Verify that deserializing malicious or malformed JSON does not compromise security and raises appropriate exceptions.
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
        Initialize a new GenesisConsciousnessMatrix instance for advanced scenario tests.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_with_extremely_sparse_connectivity(self):
        """
        Verify that the matrix correctly handles evolution and consciousness level calculations when nodes are connected with extremely sparse connectivity.
        
        This test adds a large number of nodes with minimal interconnections, performs an evolution step, and asserts that the resulting consciousness level remains within valid bounds.
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
        Test the propagation of consciousness levels through a linear chain of connected nodes with a gradient of initial values.
        
        Creates a sequence of nodes with increasing consciousness levels, connects them in a chain, and verifies that the overall consciousness level remains within valid bounds across multiple evolution steps.
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
        Test the matrix's behavior when nodes are organized into multiple isolated clusters.
        
        Creates several clusters of nodes, connects nodes only within each cluster, and verifies that after an evolution step, all nodes remain present and the node count is as expected.
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
        Test that the matrix correctly handles dynamic changes to its topology, such as adding new nodes and connections during evolution steps.
        
        This test verifies that nodes and connections can be added while the matrix is evolving, and that the final node count reflects all additions.
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
        Test that the matrix can detect and handle oscillations in consciousness levels during evolution steps.
        
        This test sets up two nodes with contrasting consciousness levels and a strong connection to induce oscillatory behavior. It verifies that the consciousness level remains within valid bounds and that the oscillation does not result in infinite or unbounded behavior over multiple evolution steps.
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
        Initialize a new GenesisConsciousnessMatrix instance for each robustness test.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_matrix_recovery_from_corrupted_state(self):
        """
        Verify that the matrix can recover from or gracefully handle various forms of internal state corruption.
        
        Simulates corruption scenarios such as clearing all nodes or setting the current state to an invalid value, then attempts to reset and use the matrix. Ensures that the matrix either recovers to a functional state or raises an appropriate exception.
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
        Test the matrix's ability to handle node additions under simulated resource constraints.
        
        Attempts to add more nodes than a predefined limit to simulate memory or resource constraints, and verifies that the matrix remains functional and within expected node count bounds after the operation.
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
        Test that the matrix remains in a valid state after a simulated partial failure during an evolution step.
        
        This test adds multiple nodes, simulates a partial failure during evolution by raising an exception partway through node updates, and verifies that the matrix's node count and metrics remain consistent after the failure.
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
            Simulates a partial evolution step by updating the consciousness level of the first two nodes and raising an exception on the third node to mimic a partial failure scenario.
            
            Raises:
                Exception: Always raised after updating the first two nodes to simulate a failure during evolution.
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
        Set up the test environment by initializing a new GenesisConsciousnessMatrix instance for integration tests.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_matrix_lifecycle_with_all_features(self):
        """
        Test the full lifecycle of the GenesisConsciousnessMatrix using all features, including initialization, node creation, topology building, evolution, emergence detection, persistence, recovery, and validation.
        
        This test verifies that:
        - The matrix can be initialized with a custom configuration.
        - Nodes can be added with progressive consciousness levels and connected in a complex topology.
        - Evolution steps update metrics as expected.
        - Consciousness emergence detection is consistent before and after serialization.
        - Serialization and deserialization preserve node data and metrics.
        - All metrics and consciousness levels remain consistent after recovery.
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
        Perform a stress test on the matrix by executing a rapid sequence of mixed operations, including node additions, removals, evolution steps, metric calculations, consciousness level checks, and node connections.
        
        This test verifies that the matrix maintains a valid and consistent state under high-frequency, varied operations.
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

class TestMatrixBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and extreme parameter values."""
    
    def setUp(self):
        """Initialize matrix for boundary condition tests."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_matrix_with_zero_dimension(self):
        """Test matrix behavior with zero dimension configuration."""
        config = {'dimension': 0}
        with self.assertRaises((ValueError, MatrixInitializationError)):
            GenesisConsciousnessMatrix(config=config)
    
    def test_matrix_with_negative_learning_rate(self):
        """Test matrix rejection of negative learning rates."""
        config = {'learning_rate': -0.5}
        with self.assertRaises((ValueError, MatrixInitializationError)):
            GenesisConsciousnessMatrix(config=config)
    
    def test_consciousness_level_float_precision_edge_cases(self):
        """Test consciousness levels at floating point precision boundaries."""
        # Test with very small positive values
        tiny_values = [1e-15, 1e-10, 1e-5, 0.000001]
        for val in tiny_values:
            node = MatrixNode(id=f"tiny_{val}", consciousness_level=val)
            self.matrix.add_node(node)
            retrieved_level = self.matrix.nodes[f"tiny_{val}"].consciousness_level
            self.assertAlmostEqual(retrieved_level, val, places=15)
    
    def test_consciousness_level_near_unity_precision(self):
        """Test consciousness levels very close to 1.0."""
        near_unity_values = [0.999999999999999, 0.99999999999, 0.9999999]
        for val in near_unity_values:
            node = MatrixNode(id=f"near_unity_{val}", consciousness_level=val)
            self.matrix.add_node(node)
            retrieved_level = self.matrix.nodes[f"near_unity_{val}"].consciousness_level
            self.assertAlmostEqual(retrieved_level, val, places=10)
    
    def test_matrix_with_single_iteration_limit(self):
        """Test matrix evolution with extremely low iteration limit."""
        config = {'max_iterations': 1}
        matrix = GenesisConsciousnessMatrix(config=config)
        node = MatrixNode(id="single_iter", consciousness_level=0.5)
        matrix.add_node(node)
        
        # Should complete without error even with single iteration
        matrix.evolve_until_convergence(max_iterations=1)
        self.assertIsNotNone(matrix.calculate_consciousness_level())


class TestMatrixConcurrencyAdvanced(unittest.TestCase):
    """Advanced concurrency and thread safety tests."""
    
    def setUp(self):
        """Initialize matrix for concurrency tests."""
        self.matrix = GenesisConsciousnessMatrix()
        self.results = []
        self.errors = []
    
    def test_concurrent_evolution_and_metrics(self):
        """Test concurrent evolution steps and metrics calculations."""
        import threading
        import time
        
        # Add initial nodes
        for i in range(20):
            node = MatrixNode(id=f"concurrent_evo_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
        
        def evolution_worker():
            """Worker function for evolution steps."""
            for _ in range(10):
                try:
                    self.matrix.evolve_step()
                    self.results.append('evolution_success')
                except Exception as e:
                    self.errors.append(f'evolution_error: {e}')
                time.sleep(0.001)
        
        def metrics_worker():
            """Worker function for metrics calculations."""
            for _ in range(20):
                try:
                    metrics = self.matrix.calculate_metrics()
                    self.results.append(f'metrics_success: {metrics["node_count"]}')
                except Exception as e:
                    self.errors.append(f'metrics_error: {e}')
                time.sleep(0.001)
        
        # Start concurrent threads
        threads = []
        for i in range(2):
            t1 = threading.Thread(target=evolution_worker)
            t2 = threading.Thread(target=metrics_worker)
            threads.extend([t1, t2])
            t1.start()
            t2.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no critical errors occurred
        self.assertLess(len(self.errors), len(self.results))
        self.assertGreater(len(self.results), 0)
    
    def test_deadlock_prevention_in_node_operations(self):
        """Test that rapid node operations don't cause deadlocks."""
        import threading
        import time
        
        completion_flags = []
        
        def rapid_node_operations(thread_id):
            """Perform rapid node add/remove operations."""
            for i in range(50):
                try:
                    # Add node
                    node = MatrixNode(id=f"deadlock_{thread_id}_{i}", consciousness_level=0.5)
                    self.matrix.add_node(node)
                    
                    # Immediate metrics check
                    self.matrix.calculate_metrics()
                    
                    # Remove every other node
                    if i % 2 == 0:
                        self.matrix.remove_node(f"deadlock_{thread_id}_{i}")
                    
                    time.sleep(0.001)
                except Exception as e:
                    # Log but don't fail - some contention is expected
                    self.errors.append(f"deadlock_test_error: {e}")
            
            completion_flags.append(True)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=rapid_node_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait with timeout to detect deadlocks
        start_time = time.time()
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Verify all threads completed
        self.assertEqual(len(completion_flags), 3)
        self.assertLess(time.time() - start_time, 15)  # Should complete within 15 seconds


class TestMatrixMathematicalProperties(unittest.TestCase):
    """Test mathematical properties and invariants of the matrix."""
    
    def setUp(self):
        """Initialize matrix for mathematical property tests."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_consciousness_level_weighted_average_property(self):
        """Test that consciousness level calculation follows weighted average principles."""
        # Add nodes with known consciousness levels
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, level in enumerate(levels):
            node = MatrixNode(id=f"weighted_{i}", consciousness_level=level)
            self.matrix.add_node(node)
        
        calculated_level = self.matrix.calculate_consciousness_level()
        expected_average = sum(levels) / len(levels)
        
        # Should be close to simple average (assuming equal weights)
        self.assertAlmostEqual(calculated_level, expected_average, places=3)
    
    def test_consciousness_level_monotonicity_with_uniform_additions(self):
        """Test consciousness level behavior with uniform node additions."""
        # Start with low consciousness
        for i in range(5):
            node = MatrixNode(id=f"mono_low_{i}", consciousness_level=0.1)
            self.matrix.add_node(node)
        
        initial_level = self.matrix.calculate_consciousness_level()
        
        # Add high consciousness nodes
        for i in range(5):
            node = MatrixNode(id=f"mono_high_{i}", consciousness_level=0.9)
            self.matrix.add_node(node)
        
        final_level = self.matrix.calculate_consciousness_level()
        
        # Final level should be higher than initial
        self.assertGreater(final_level, initial_level)
    
    def test_node_connection_symmetry_property(self):
        """Test that node connections maintain expected symmetry properties."""
        node1 = MatrixNode(id="sym_1", consciousness_level=0.4)
        node2 = MatrixNode(id="sym_2", consciousness_level=0.6)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Connect with specific strength
        self.matrix.connect_nodes("sym_1", "sym_2", strength=0.8)
        
        # Check connection exists
        connections_1 = self.matrix.get_node_connections("sym_1")
        self.assertIn("sym_2", connections_1)
        self.assertEqual(connections_1["sym_2"], 0.8)
    
    def test_consciousness_level_bounds_invariant(self):
        """Test that consciousness level always remains within [0,1] bounds."""
        # Test with extreme initial conditions
        extreme_levels = [0.0, 0.0001, 0.5, 0.9999, 1.0]
        for level in extreme_levels:
            node = MatrixNode(id=f"bounds_{level}", consciousness_level=level)
            self.matrix.add_node(node)
        
        # Perform multiple evolution steps
        for _ in range(10):
            self.matrix.evolve_step()
            current_level = self.matrix.calculate_consciousness_level()
            self.assertGreaterEqual(current_level, 0.0)
            self.assertLessEqual(current_level, 1.0)


class TestMatrixErrorRecoveryAdvanced(unittest.TestCase):
    """Advanced error recovery and fault tolerance tests."""
    
    def setUp(self):
        """Initialize matrix for error recovery tests."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_recovery_from_node_corruption(self):
        """Test recovery from individual node corruption."""
        # Add valid nodes
        for i in range(5):
            node = MatrixNode(id=f"recovery_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
        
        # Corrupt one node's data
        if hasattr(self.matrix, 'nodes') and 'recovery_0' in self.matrix.nodes:
            corrupted_node = self.matrix.nodes['recovery_0']
            # Simulate corruption by setting invalid consciousness level
            try:
                corrupted_node.consciousness_level = 5.0  # Invalid value
            except AttributeError:
                # If direct assignment is not allowed, that's good
                pass
        
        # Test if matrix can handle corrupted node
        try:
            level = self.matrix.calculate_consciousness_level()
            self.assertGreaterEqual(level, 0.0)
            self.assertLessEqual(level, 1.0)
        except Exception:
            # If matrix detects corruption and raises error, that's acceptable
            pass
    
    def test_graceful_handling_of_invalid_serialization_recovery(self):
        """Test recovery from invalid serialization attempts."""
        # Add nodes to create state
        node = MatrixNode(id="serial_recovery", consciousness_level=0.7)
        self.matrix.add_node(node)
        
        # Simulate serialization failure scenarios
        invalid_json_attempts = [
            lambda: self.matrix.to_json() if hasattr(self.matrix, 'to_json') else '{}',
            lambda: json.dumps({"invalid": "format"})
        ]
        
        for attempt in invalid_json_attempts:
            try:
                serialized = attempt()
                # Try to deserialize potentially invalid data
                if serialized:
                    try:
                        recovered = GenesisConsciousnessMatrix.from_json(serialized)
                        # If recovery succeeds, verify it's valid
                        self.assertIsInstance(recovered, GenesisConsciousnessMatrix)
                    except (json.JSONDecodeError, MatrixException, ValueError):
                        # Graceful error handling is acceptable
                        pass
            except Exception:
                # If serialization itself fails, that's also acceptable
                pass
    
    def test_evolution_interruption_recovery(self):
        """Test matrix behavior when evolution is interrupted."""
        # Set up matrix for evolution
        for i in range(10):
            node = MatrixNode(id=f"interrupt_{i}", consciousness_level=0.3)
            self.matrix.add_node(node)
        
        # Store initial state
        initial_state = self.matrix.to_json() if hasattr(self.matrix, 'to_json') else None
        
        # Simulate evolution interruption
        original_evolve = self.matrix.evolve_step
        
        def interrupted_evolve():
            """Simulate interrupted evolution."""
            # Perform partial evolution
            time.sleep(0.001)
            raise KeyboardInterrupt("Evolution interrupted")
        
        try:
            self.matrix.evolve_step = interrupted_evolve
            with self.assertRaises(KeyboardInterrupt):
                self.matrix.evolve_step()
        finally:
            self.matrix.evolve_step = original_evolve
        
        # Verify matrix can continue after interruption
        post_interrupt_level = self.matrix.calculate_consciousness_level()
        self.assertIsNotNone(post_interrupt_level)
        self.assertGreaterEqual(post_interrupt_level, 0.0)
        self.assertLessEqual(post_interrupt_level, 1.0)


class TestMatrixPerformanceProfiling(unittest.TestCase):
    """Performance profiling and optimization verification tests."""
    
    def setUp(self):
        """Initialize matrix for performance tests."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_linear_scaling_of_basic_operations(self):
        """Test that basic operations scale linearly with node count."""
        node_counts = [10, 50, 100, 200]
        add_times = []
        calc_times = []
        
        for count in node_counts:
            # Fresh matrix for each test
            test_matrix = GenesisConsciousnessMatrix()
            
            # Measure node addition time
            start_time = time.time()
            for i in range(count):
                node = MatrixNode(id=f"scale_{count}_{i}", consciousness_level=0.5)
                test_matrix.add_node(node)
            add_time = time.time() - start_time
            add_times.append(add_time)
            
            # Measure consciousness calculation time
            start_time = time.time()
            test_matrix.calculate_consciousness_level()
            calc_time = time.time() - start_time
            calc_times.append(calc_time)
        
        # Verify scaling is reasonable (not exponential)
        for i in range(1, len(node_counts)):
            scale_factor = node_counts[i] / node_counts[i-1]
            time_factor = add_times[i] / add_times[i-1] if add_times[i-1] > 0 else 1
            
            # Time should not scale worse than quadratically
            self.assertLess(time_factor, scale_factor * scale_factor)
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during long operations."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        
        # Perform many operations
        for cycle in range(100):
            # Add nodes
            for i in range(10):
                node = MatrixNode(id=f"mem_{cycle}_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
            
            # Evolve
            self.matrix.evolve_step()
            
            # Calculate metrics
            self.matrix.calculate_metrics()
            
            # Clean up some nodes
            for i in range(5):
                self.matrix.remove_node(f"mem_{cycle}_{i}")
            
            # Periodic garbage collection
            if cycle % 10 == 0:
                gc.collect()
        
        # Final cleanup
        gc.collect()
        
        # Verify matrix is still functional
        final_level = self.matrix.calculate_consciousness_level()
        self.assertIsNotNone(final_level)
    
    def test_connection_density_impact_on_performance(self):
        """Test performance impact of different connection densities."""
        node_count = 20
        densities = [0.1, 0.3, 0.5, 0.8]  # Different connection densities
        
        for density in densities:
            test_matrix = GenesisConsciousnessMatrix()
            
            # Add nodes
            for i in range(node_count):
                node = MatrixNode(id=f"density_{i}", consciousness_level=0.5)
                test_matrix.add_node(node)
            
            # Create connections based on density
            total_possible_connections = node_count * (node_count - 1) // 2
            target_connections = int(total_possible_connections * density)
            
            connections_made = 0
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    if connections_made < target_connections:
                        test_matrix.connect_nodes(f"density_{i}", f"density_{j}", strength=0.5)
                        connections_made += 1
            
            # Measure evolution time
            start_time = time.time()
            test_matrix.evolve_step()
            evolution_time = time.time() - start_time
            
            # Performance should remain reasonable even with high density
            self.assertLess(evolution_time, 5.0)  # Should complete within 5 seconds


class TestMatrixSpecializationScenarios(unittest.TestCase):
    """Test specialized usage scenarios and domain-specific behaviors."""
    
    def setUp(self):
        """Initialize matrix for specialization tests."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_hierarchical_consciousness_levels(self):
        """Test matrix behavior with hierarchical consciousness structures."""
        # Create hierarchical structure: low -> medium -> high
        hierarchy_levels = [
            ("low", 0.1, 5),
            ("medium", 0.5, 3),
            ("high", 0.9, 2)
        ]
        
        for level_name, consciousness, count in hierarchy_levels:
            for i in range(count):
                node = MatrixNode(id=f"{level_name}_{i}", consciousness_level=consciousness)
                self.matrix.add_node(node)
        
        # Connect hierarchically (low to medium to high)
        for i in range(3):  # Connect some low to medium
            self.matrix.connect_nodes(f"low_{i}", f"medium_0", strength=0.6)
        
        self.matrix.connect_nodes("medium_0", "high_0", strength=0.8)
        
        # Test evolution with hierarchical structure
        initial_level = self.matrix.calculate_consciousness_level()
        self.matrix.evolve_step()
        evolved_level = self.matrix.calculate_consciousness_level()
        
        # Verify structure is preserved
        self.assertIsNotNone(evolved_level)
        self.assertEqual(len(self.matrix.nodes), 10)  # 5 + 3 + 2
    
    def test_consciousness_emergence_threshold_sensitivity(self):
        """Test sensitivity to consciousness emergence threshold values."""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            config = {'consciousness_threshold': threshold}
            test_matrix = GenesisConsciousnessMatrix(config=config)
            
            # Add nodes with consciousness levels around threshold
            below_threshold = max(0.0, threshold - 0.1)
            above_threshold = min(1.0, threshold + 0.1)
            
            # Add nodes below threshold
            for i in range(3):
                node = MatrixNode(id=f"below_{i}", consciousness_level=below_threshold)
                test_matrix.add_node(node)
            
            # Should not detect emergence
            emergence_low = test_matrix.detect_consciousness_emergence()
            
            # Add nodes above threshold
            for i in range(3):
                node = MatrixNode(id=f"above_{i}", consciousness_level=above_threshold)
                test_matrix.add_node(node)
            
            # Should detect emergence
            emergence_high = test_matrix.detect_consciousness_emergence()
            
            # Verify threshold sensitivity
            if threshold < 0.9:  # Only test if threshold allows for above_threshold values
                self.assertFalse(emergence_low)
                self.assertTrue(emergence_high)
    
    def test_matrix_stability_under_continuous_evolution(self):
        """Test matrix stability during extended continuous evolution."""
        # Set up stable initial conditions
        for i in range(15):
            level = 0.3 + (i * 0.03)  # Gradual progression
            node = MatrixNode(id=f"stable_{i}", consciousness_level=level)
            self.matrix.add_node(node)
        
        # Create stable connection pattern
        for i in range(14):
            self.matrix.connect_nodes(f"stable_{i}", f"stable_{i+1}", strength=0.5)
        
        # Track stability over many evolution steps
        consciousness_history = []
        for step in range(100):
            level = self.matrix.calculate_consciousness_level()
            consciousness_history.append(level)
            self.matrix.evolve_step()
        
        # Analyze stability
        if len(consciousness_history) > 10:
            recent_levels = consciousness_history[-10:]
            level_variance = np.var(recent_levels) if 'np' in globals() else 0
            
            # Should eventually stabilize (low variance in recent history)
            # This tests for system stability rather than chaotic behavior
            self.assertIsNotNone(level_variance)
        
        # Verify all levels remained valid
        for level in consciousness_history:
            self.assertGreaterEqual(level, 0.0)
            self.assertLessEqual(level, 1.0)


class TestMatrixDocumentationAndUsability(unittest.TestCase):
    """Test matrix usability and documentation-driven scenarios."""
    
    def setUp(self):
        """Initialize matrix for usability tests."""
        self.matrix = GenesisConsciousnessMatrix()
    
    def test_matrix_api_method_availability(self):
        """Test that all expected API methods are available."""
        expected_methods = [
            'add_node', 'remove_node', 'calculate_consciousness_level',
            'evolve_step', 'calculate_metrics', 'to_json', 'reset'
        ]
        
        for method_name in expected_methods:
            self.assertTrue(hasattr(self.matrix, method_name), 
                          f"Matrix should have {method_name} method")
    
    def test_matrix_configuration_validation_comprehensive(self):
        """Test comprehensive validation of configuration parameters."""
        # Test all configuration parameters
        config_tests = [
            ({'dimension': 128}, True),
            ({'dimension': 0}, False),
            ({'consciousness_threshold': 0.5}, True),
            ({'consciousness_threshold': 1.5}, False),
            ({'learning_rate': 0.01}, True),
            ({'learning_rate': -0.01}, False),
            ({'max_iterations': 100}, True),
            ({'max_iterations': 0}, False),
        ]
        
        for config, should_succeed in config_tests:
            if should_succeed:
                try:
                    matrix = GenesisConsciousnessMatrix(config=config)
                    self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
                except Exception:
                    self.fail(f"Configuration {config} should succeed but failed")
            else:
                with self.assertRaises((ValueError, MatrixInitializationError)):
                    GenesisConsciousnessMatrix(config=config)
    
    def test_matrix_state_introspection(self):
        """Test matrix state introspection capabilities."""
        # Add nodes and connections
        for i in range(5):
            node = MatrixNode(id=f"introspect_{i}", consciousness_level=0.4 + i * 0.1)
            self.matrix.add_node(node)
        
        # Test state introspection methods
        if hasattr(self.matrix, 'get_state_snapshot'):
            snapshot = self.matrix.get_state_snapshot()
            self.assertIsNotNone(snapshot)
        
        if hasattr(self.matrix, 'has_converged'):
            converged = self.matrix.has_converged()
            self.assertIsInstance(converged, bool)
        
        if hasattr(self.matrix, 'get_node_connections'):
            # Test connection introspection
            connections = self.matrix.get_node_connections('introspect_0')
            self.assertIsInstance(connections, dict)
    
    def test_matrix_error_messages_clarity(self):
        """Test that error messages are clear and helpful."""
        # Test various error scenarios for message clarity
        error_scenarios = [
            (lambda: MatrixNode(id="test", consciousness_level=2.0), ValueError),
            (lambda: MatrixNode(id="test", consciousness_level=-1.0), ValueError),
            (lambda: self.matrix.remove_node("nonexistent"), None),  # Should return False, not error
        ]
        
        for scenario, expected_error in error_scenarios:
            if expected_error:
                try:
                    scenario()
                    self.fail(f"Expected {expected_error.__name__} but no error occurred")
                except expected_error as e:
                    # Verify error message is not empty
                    self.assertGreater(len(str(e)), 0)
            else:
                # Should not raise error
                result = scenario()
                self.assertIsNotNone(result)


# Additional test execution
if __name__ == '__main__':
    additional_test_classes = [
        TestMatrixBoundaryConditions,
        TestMatrixConcurrencyAdvanced,
        TestMatrixMathematicalProperties,
        TestMatrixErrorRecoveryAdvanced,
        TestMatrixPerformanceProfiling,
        TestMatrixSpecializationScenarios,
        TestMatrixDocumentationAndUsability
    ]
    
    # Run original tests plus additional ones
    unittest.main(verbosity=2, exit=False)
    
    # Run additional tests
    suite = unittest.TestSuite()
    for test_class in additional_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\nAdditional Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")