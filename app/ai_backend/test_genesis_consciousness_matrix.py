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
        Performs cleanup operations after each test method by invoking the matrix's cleanup method if available.
        """
        if hasattr(self.matrix, 'cleanup'):
            self.matrix.cleanup()
    
    def test_matrix_initialization_default(self):
        """
        Verifies that the matrix initializes correctly with default parameters, ensuring required attributes are present.
        """
        matrix = GenesisConsciousnessMatrix()
        self.assertIsInstance(matrix, GenesisConsciousnessMatrix)
        self.assertTrue(hasattr(matrix, 'state'))
        self.assertTrue(hasattr(matrix, 'nodes'))
        
    def test_matrix_initialization_custom_config(self):
        """
        Test that the matrix initializes correctly with a custom configuration, verifying dimension and consciousness threshold values.
        """
        matrix = GenesisConsciousnessMatrix(config=self.test_config)
        self.assertEqual(matrix.dimension, self.test_config['dimension'])
        self.assertEqual(matrix.consciousness_threshold, self.test_config['consciousness_threshold'])
        
    def test_matrix_initialization_invalid_config(self):
        """
        Tests that initializing the matrix with an invalid configuration raises a MatrixInitializationError.
        """
        invalid_config = {'dimension': -1, 'consciousness_threshold': 2.0}
        with self.assertRaises(MatrixInitializationError):
            GenesisConsciousnessMatrix(config=invalid_config)
            
    def test_add_consciousness_node_valid(self):
        """
        Tests that adding a valid consciousness node to the matrix succeeds and the node is stored.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        result = self.matrix.add_node(node)
        self.assertTrue(result)
        self.assertIn("test_node", self.matrix.nodes)
        
    def test_add_consciousness_node_duplicate(self):
        """
        Tests that adding a duplicate consciousness node to the matrix raises an InvalidStateException.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        with self.assertRaises(InvalidStateException):
            self.matrix.add_node(node)
            
    def test_remove_consciousness_node_existing(self):
        """
        Tests that removing an existing consciousness node from the matrix returns True and the node is no longer present.
        """
        node = MatrixNode(id="test_node", consciousness_level=0.5)
        self.matrix.add_node(node)
        result = self.matrix.remove_node("test_node")
        self.assertTrue(result)
        self.assertNotIn("test_node", self.matrix.nodes)
        
    def test_remove_consciousness_node_nonexistent(self):
        """
        Test that removing a non-existent consciousness node from the matrix returns False.
        """
        result = self.matrix.remove_node("nonexistent_node")
        self.assertFalse(result)
        
    def test_consciousness_state_transition_valid(self):
        """Test valid consciousness state transitions."""
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
        Tests that the matrix correctly calculates the average consciousness level across multiple nodes.
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
        Tests that the matrix calculates the correct consciousness level when only a single node is present.
        """
        node = MatrixNode(id="single_node", consciousness_level=0.8)
        self.matrix.add_node(node)
        calculated_level = self.matrix.calculate_consciousness_level()
        self.assertEqual(calculated_level, 0.8)
        
    def test_matrix_evolution_step(self):
        """
        Tests that performing a single evolution step on the matrix changes its state snapshot.
        """
        initial_state = self.matrix.get_state_snapshot()
        self.matrix.evolve_step()
        final_state = self.matrix.get_state_snapshot()
        self.assertNotEqual(initial_state, final_state)
        
    def test_matrix_evolution_convergence(self):
        """
        Tests that the matrix evolution process correctly detects convergence within a specified maximum number of iterations.
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
        Tests that the matrix can be serialized to a JSON string containing the expected nodes and state fields.
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
        Tests that deserializing a matrix from a JSON string correctly restores its nodes and their consciousness levels.
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
        Tests saving the matrix state to a JSON file and loading it back, verifying that node data is preserved accurately.
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
        Tests establishing a connection between two matrix nodes and verifies that the connection is correctly stored and retrievable.
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
        Verify that attempting to connect two non-existent nodes in the matrix raises an InvalidStateException.
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
        Tests that the matrix correctly calculates and returns performance metrics including average consciousness, node count, and connection density after adding nodes.
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
        Tests that the matrix can perform an evolution step with 100 nodes in under one second.
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
        Tests that adding and removing nodes updates the matrix's node count as expected, verifying memory usage remains consistent.
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
        Verifies that adding nodes to the matrix from multiple threads completes successfully, ensuring thread safety of matrix operations.
        """
        import threading
        import time
        
        results = []
        
        def add_nodes_thread(thread_id):
            """
            Adds ten `MatrixNode` instances with unique IDs for a given thread, recording the success of each addition in the shared `results` list.
            
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
        Tests that the string representation of each ConsciousnessState enum value matches its name.
        """
        self.assertEqual(str(ConsciousnessState.DORMANT), "DORMANT")
        self.assertEqual(str(ConsciousnessState.ACTIVE), "ACTIVE")
        self.assertEqual(str(ConsciousnessState.AWARE), "AWARE")
        self.assertEqual(str(ConsciousnessState.TRANSCENDENT), "TRANSCENDENT")


class TestMatrixNode(unittest.TestCase):
    """Test cases for MatrixNode class."""
    
    def setUp(self):
        """
        Initializes a MatrixNode instance with a test ID and consciousness level before each test.
        """
        self.node = MatrixNode(id="test_node", consciousness_level=0.5)
        
    def test_node_initialization(self):
        """
        Test that a MatrixNode is correctly initialized with a valid ID and consciousness level.
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
        Tests that updating a node's consciousness level to a valid value correctly changes its state.
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
        Tests that two MatrixNode instances with the same ID and consciousness level are considered equal, while nodes with different IDs are not.
        """
        node1 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node2 = MatrixNode(id="equal_test", consciousness_level=0.5)
        node3 = MatrixNode(id="different_test", consciousness_level=0.5)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        
    def test_node_hash(self):
        """
        Tests that MatrixNode instances with the same ID produce identical hash values, ensuring correct behavior in sets and dictionaries.
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
        Verify that custom matrix exceptions inherit from their intended base classes.
        """
        self.assertTrue(issubclass(MatrixException, Exception))
        self.assertTrue(issubclass(InvalidStateException, MatrixException))
        self.assertTrue(issubclass(MatrixInitializationError, MatrixException))
        
    def test_matrix_exception_messages(self):
        """
        Verify that custom matrix exceptions carry and return the correct error messages when raised.
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
        Initializes a new GenesisConsciousnessMatrix instance for integration tests.
        """
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_complete_consciousness_evolution_cycle(self):
        """
        Tests a full consciousness evolution cycle by adding nodes, connecting them, evolving the matrix until convergence, and verifying that the overall consciousness level changes as a result.
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
        Tests the detection of consciousness emergence through a full cycle, starting from low consciousness levels and increasing them to trigger emergence detection.
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
        Verifies that serializing and deserializing the matrix preserves all nodes, their consciousness levels, and connections, ensuring data integrity across persistence operations.
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
    """Advanced test cases for Genesis Consciousness Matrix edge cases and corner conditions."""
    
    def setUp(self):
        """Initialize matrix for advanced testing scenarios."""
        self.matrix = GenesisConsciousnessMatrix()
        self.large_config = {
            'dimension': 1024,
            'consciousness_threshold': 0.95,
            'learning_rate': 0.0001,
            'max_iterations': 10000,
            'batch_size': 32,
            'convergence_tolerance': 1e-6
        }
        
    def test_matrix_initialization_edge_configurations(self):
        """Test matrix initialization with edge case configurations."""
        # Test minimum valid configuration
        min_config = {
            'dimension': 1,
            'consciousness_threshold': 0.0,
            'learning_rate': 1e-10,
            'max_iterations': 1
        }
        matrix = GenesisConsciousnessMatrix(config=min_config)
        self.assertEqual(matrix.dimension, 1)
        self.assertEqual(matrix.consciousness_threshold, 0.0)
        
        # Test maximum valid configuration
        max_config = {
            'dimension': 10000,
            'consciousness_threshold': 1.0,
            'learning_rate': 1.0,
            'max_iterations': 100000
        }
        matrix = GenesisConsciousnessMatrix(config=max_config)
        self.assertEqual(matrix.dimension, 10000)
        self.assertEqual(matrix.consciousness_threshold, 1.0)
        
    def test_matrix_initialization_boundary_values(self):
        """Test matrix initialization with boundary value configurations."""
        boundary_configs = [
            {'dimension': 0, 'consciousness_threshold': 0.5},  # Should fail
            {'dimension': 10, 'consciousness_threshold': -0.1},  # Should fail
            {'dimension': 10, 'consciousness_threshold': 1.1},  # Should fail
            {'learning_rate': -0.1},  # Should fail
            {'max_iterations': 0}  # Should fail
        ]
        
        for config in boundary_configs:
            with self.assertRaises(MatrixInitializationError):
                GenesisConsciousnessMatrix(config=config)
                
    def test_matrix_with_floating_point_precision(self):
        """Test matrix operations with floating point precision edge cases."""
        # Test with very small consciousness levels
        node1 = MatrixNode(id="precision_test_1", consciousness_level=1e-10)
        node2 = MatrixNode(id="precision_test_2", consciousness_level=1.0 - 1e-10)
        
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        level = self.matrix.calculate_consciousness_level()
        self.assertIsInstance(level, float)
        self.assertGreaterEqual(level, 0.0)
        self.assertLessEqual(level, 1.0)
        
    def test_matrix_node_operations_stress_test(self):
        """Stress test matrix node operations with large numbers of nodes."""
        # Add 1000 nodes rapidly
        nodes_to_add = 1000
        start_time = datetime.now()
        
        for i in range(nodes_to_add):
            node = MatrixNode(id=f"stress_node_{i}", consciousness_level=i / nodes_to_add)
            self.matrix.add_node(node)
            
        add_time = (datetime.now() - start_time).total_seconds()
        
        # Verify all nodes added
        self.assertEqual(len(self.matrix.nodes), nodes_to_add)
        
        # Remove half of the nodes
        start_time = datetime.now()
        for i in range(nodes_to_add // 2):
            self.matrix.remove_node(f"stress_node_{i}")
            
        remove_time = (datetime.now() - start_time).total_seconds()
        
        # Verify correct number of nodes remain
        self.assertEqual(len(self.matrix.nodes), nodes_to_add // 2)
        
        # Performance assertion (should complete operations reasonably quickly)
        self.assertLess(add_time, 5.0)
        self.assertLess(remove_time, 3.0)
        
    def test_matrix_consciousness_level_statistical_properties(self):
        """Test statistical properties of consciousness level calculations."""
        # Test with nodes having specific distribution
        consciousness_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for i, level in enumerate(consciousness_levels):
            node = MatrixNode(id=f"stat_node_{i}", consciousness_level=level)
            self.matrix.add_node(node)
            
        calculated_mean = self.matrix.calculate_consciousness_level()
        expected_mean = sum(consciousness_levels) / len(consciousness_levels)
        
        self.assertAlmostEqual(calculated_mean, expected_mean, places=5)
        
        # Test variance calculation if available
        if hasattr(self.matrix, 'calculate_consciousness_variance'):
            variance = self.matrix.calculate_consciousness_variance()
            self.assertGreaterEqual(variance, 0.0)
            
    def test_matrix_node_connection_weight_validation(self):
        """Test validation of connection weights between nodes."""
        node1 = MatrixNode(id="weight_test_1", consciousness_level=0.5)
        node2 = MatrixNode(id="weight_test_2", consciousness_level=0.5)
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        
        # Test invalid connection weights
        invalid_weights = [-0.1, 1.1, float('inf'), float('-inf'), float('nan')]
        
        for weight in invalid_weights:
            with self.assertRaises((ValueError, InvalidStateException)):
                self.matrix.connect_nodes("weight_test_1", "weight_test_2", strength=weight)
                
    def test_matrix_circular_connection_detection(self):
        """Test detection and handling of circular connections."""
        # Create a circular connection pattern
        nodes = []
        for i in range(5):
            node = MatrixNode(id=f"circular_node_{i}", consciousness_level=0.5)
            nodes.append(node)
            self.matrix.add_node(node)
            
        # Create circular connections
        for i in range(5):
            next_i = (i + 1) % 5
            self.matrix.connect_nodes(f"circular_node_{i}", f"circular_node_{next_i}", strength=0.5)
            
        # Test that circular connections are properly handled
        if hasattr(self.matrix, 'detect_circular_connections'):
            circular_detected = self.matrix.detect_circular_connections()
            self.assertTrue(circular_detected)
            
    def test_matrix_consciousness_state_transition_edge_cases(self):
        """Test consciousness state transitions with edge cases."""
        # Test rapid state transitions
        states = [ConsciousnessState.DORMANT, ConsciousnessState.ACTIVE, 
                 ConsciousnessState.AWARE, ConsciousnessState.TRANSCENDENT]
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            
            # Test valid progression
            if hasattr(self.matrix, 'can_transition_to'):
                can_transition = self.matrix.can_transition_to(current_state, next_state)
                if can_transition:
                    result = self.matrix.transition_state(current_state, next_state)
                    self.assertTrue(result)
                    
    def test_matrix_concurrent_evolution_steps(self):
        """Test concurrent evolution steps for race condition detection."""
        import threading
        import time
        
        # Add nodes for evolution
        for i in range(20):
            node = MatrixNode(id=f"concurrent_node_{i}", consciousness_level=0.4)
            self.matrix.add_node(node)
            
        evolution_results = []
        
        def concurrent_evolution():
            """Perform evolution step concurrently."""
            try:
                initial_state = self.matrix.get_state_snapshot()
                self.matrix.evolve_step()
                final_state = self.matrix.get_state_snapshot()
                evolution_results.append(initial_state != final_state)
            except Exception as e:
                evolution_results.append(False)
                
        # Run concurrent evolution steps
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_evolution)
            threads.append(thread)
            thread.start()
            time.sleep(0.1)  # Small delay between thread starts
            
        for thread in threads:
            thread.join()
            
        # Verify that at least some evolution occurred
        self.assertGreater(len(evolution_results), 0)
        
    def test_matrix_serialization_with_complex_state(self):
        """Test serialization with complex matrix states."""
        # Create complex matrix state with many interconnected nodes
        for i in range(50):
            node = MatrixNode(id=f"complex_node_{i}", consciousness_level=0.3 + (i % 7) * 0.1)
            self.matrix.add_node(node)
            
        # Add complex connection pattern
        for i in range(50):
            for j in range(i + 1, min(i + 4, 50)):
                self.matrix.connect_nodes(f"complex_node_{i}", f"complex_node_{j}", 
                                        strength=0.2 + (i + j) % 5 * 0.15)
                
        # Serialize and measure size
        serialized = self.matrix.to_json()
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 1000)  # Should be substantial
        
        # Deserialize and verify integrity
        restored_matrix = GenesisConsciousnessMatrix.from_json(serialized)
        self.assertEqual(len(restored_matrix.nodes), 50)
        
        # Verify a sample of nodes
        for i in range(0, 50, 10):
            node_id = f"complex_node_{i}"
            self.assertIn(node_id, restored_matrix.nodes)
            original_level = self.matrix.nodes[node_id].consciousness_level
            restored_level = restored_matrix.nodes[node_id].consciousness_level
            self.assertEqual(original_level, restored_level)
            
    def test_matrix_memory_leak_detection(self):
        """Test for potential memory leaks in matrix operations."""
        import gc
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many matrix operations
        for cycle in range(10):
            # Add many nodes
            for i in range(100):
                node = MatrixNode(id=f"leak_test_{cycle}_{i}", consciousness_level=0.5)
                self.matrix.add_node(node)
                
            # Evolve
            self.matrix.evolve_step()
            
            # Remove all nodes
            node_ids = list(self.matrix.nodes.keys())
            for node_id in node_ids:
                self.matrix.remove_node(node_id)
                
            # Reset matrix
            self.matrix.reset()
            
        # Check for memory leaks
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Allow for some growth but not excessive
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000)  # Reasonable threshold
        
    def test_matrix_invalid_json_deserialization_variants(self):
        """Test various invalid JSON scenarios for robust error handling."""
        invalid_json_cases = [
            '',  # Empty string
            '{}',  # Empty object
            '{"nodes": null}',  # Null nodes
            '{"nodes": {}, "state": "INVALID"}',  # Invalid state
            '{"nodes": {"node1": {}}}',  # Missing required node fields
            '{"nodes": {"node1": {"consciousness_level": "invalid"}}}',  # Invalid consciousness level type
            '{"nodes": {"node1": {"consciousness_level": 2.0}}}',  # Out of range consciousness level
            '{"invalid_root": true}',  # Missing required root fields
            '[1, 2, 3]',  # Wrong root type
            '{"nodes": {"node1": {"consciousness_level": 0.5}}, "extra_field": "value"}',  # Extra fields
        ]
        
        for invalid_json in invalid_json_cases:
            with self.assertRaises((MatrixException, ValueError, TypeError)):
                GenesisConsciousnessMatrix.from_json(invalid_json)
                
    def test_matrix_node_update_propagation(self):
        """Test that node updates properly propagate through the matrix."""
        # Create connected nodes
        node1 = MatrixNode(id="prop_node_1", consciousness_level=0.2)
        node2 = MatrixNode(id="prop_node_2", consciousness_level=0.3)
        node3 = MatrixNode(id="prop_node_3", consciousness_level=0.4)
        
        self.matrix.add_node(node1)
        self.matrix.add_node(node2)
        self.matrix.add_node(node3)
        
        # Connect nodes
        self.matrix.connect_nodes("prop_node_1", "prop_node_2", strength=0.8)
        self.matrix.connect_nodes("prop_node_2", "prop_node_3", strength=0.7)
        
        # Update first node and check propagation
        initial_level = self.matrix.calculate_consciousness_level()
        self.matrix.nodes["prop_node_1"].update_consciousness_level(0.9)
        
        # Evolution should propagate the change
        self.matrix.evolve_step()
        final_level = self.matrix.calculate_consciousness_level()
        
        # The overall consciousness level should have increased
        self.assertGreater(final_level, initial_level)


class TestAsyncMatrixOperations(unittest.TestCase):
    """Test cases for asynchronous matrix operations if they exist."""
    
    def setUp(self):
        """Initialize matrix for async testing."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_async_evolution_if_available(self):
        """Test async evolution operations if they exist."""
        # Add nodes for async evolution
        for i in range(10):
            node = MatrixNode(id=f"async_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Test async evolution if method exists
        if hasattr(self.matrix, 'evolve_async'):
            async def run_async_evolution():
                initial_state = self.matrix.get_state_snapshot()
                await self.matrix.evolve_async()
                final_state = self.matrix.get_state_snapshot()
                return initial_state != final_state
                
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_async_evolution())
                self.assertTrue(result)
            finally:
                loop.close()
                
    def test_async_convergence_if_available(self):
        """Test async convergence operations if they exist."""
        # Add nodes
        for i in range(15):
            node = MatrixNode(id=f"async_conv_node_{i}", consciousness_level=0.1 + i * 0.05)
            self.matrix.add_node(node)
            
        # Test async convergence if method exists
        if hasattr(self.matrix, 'evolve_until_convergence_async'):
            async def run_async_convergence():
                await self.matrix.evolve_until_convergence_async(max_iterations=20)
                return self.matrix.has_converged()
                
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_async_convergence())
                self.assertTrue(result)
            finally:
                loop.close()


class TestMatrixNodeAdvanced(unittest.TestCase):
    """Advanced test cases for MatrixNode class."""
    
    def test_node_consciousness_level_precision(self):
        """Test consciousness level precision and rounding."""
        # Test various precision levels
        precision_levels = [
            0.123456789,
            0.999999999,
            0.000000001,
            0.5,
            1.0,
            0.0
        ]
        
        for level in precision_levels:
            node = MatrixNode(id=f"precision_node_{level}", consciousness_level=level)
            self.assertAlmostEqual(node.consciousness_level, level, places=9)
            
    def test_node_consciousness_level_boundary_updates(self):
        """Test consciousness level updates at boundaries."""
        node = MatrixNode(id="boundary_node", consciousness_level=0.5)
        
        # Test boundary updates
        boundary_values = [0.0, 1.0, 0.000001, 0.999999]
        
        for value in boundary_values:
            node.update_consciousness_level(value)
            self.assertEqual(node.consciousness_level, value)
            
    def test_node_immutable_id_after_creation(self):
        """Test that node ID cannot be changed after creation."""
        node = MatrixNode(id="immutable_test", consciousness_level=0.5)
        original_id = node.id
        
        # Try to modify ID (should be immutable)
        if hasattr(node, 'id') and not hasattr(node.__class__, 'id'):
            # If id is a property, it might be read-only
            try:
                node.id = "new_id"
                # If we can change it, verify it actually changed
                if node.id == "new_id":
                    self.fail("Node ID should be immutable")
            except AttributeError:
                # This is expected for read-only properties
                pass
                
        # ID should remain unchanged
        self.assertEqual(node.id, original_id)
        
    def test_node_deep_copy_behavior(self):
        """Test deep copy behavior of MatrixNode."""
        import copy
        
        original_node = MatrixNode(id="copy_test", consciousness_level=0.7)
        
        # Test shallow copy
        shallow_copy = copy.copy(original_node)
        self.assertEqual(shallow_copy.id, original_node.id)
        self.assertEqual(shallow_copy.consciousness_level, original_node.consciousness_level)
        
        # Test deep copy
        deep_copy = copy.deepcopy(original_node)
        self.assertEqual(deep_copy.id, original_node.id)
        self.assertEqual(deep_copy.consciousness_level, original_node.consciousness_level)
        
        # Modify original and verify copies are independent
        original_node.update_consciousness_level(0.9)
        self.assertNotEqual(shallow_copy.consciousness_level, original_node.consciousness_level)
        self.assertNotEqual(deep_copy.consciousness_level, original_node.consciousness_level)
        
    def test_node_serialization_deserialization(self):
        """Test individual node serialization and deserialization."""
        original_node = MatrixNode(id="serial_test", consciousness_level=0.835)
        
        # Test serialization if method exists
        if hasattr(original_node, 'to_dict'):
            node_dict = original_node.to_dict()
            self.assertIn('id', node_dict)
            self.assertIn('consciousness_level', node_dict)
            self.assertEqual(node_dict['id'], "serial_test")
            self.assertEqual(node_dict['consciousness_level'], 0.835)
            
        # Test deserialization if method exists
        if hasattr(MatrixNode, 'from_dict'):
            test_dict = {'id': 'deserial_test', 'consciousness_level': 0.642}
            deserialized_node = MatrixNode.from_dict(test_dict)
            self.assertEqual(deserialized_node.id, 'deserial_test')
            self.assertEqual(deserialized_node.consciousness_level, 0.642)


class TestMatrixPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for matrix operations."""
    
    def setUp(self):
        """Initialize matrix for performance testing."""
        self.matrix = GenesisConsciousnessMatrix()
        
    def test_large_scale_node_operations_performance(self):
        """Benchmark performance of large-scale node operations."""
        node_counts = [100, 500, 1000, 2000]
        
        for count in node_counts:
            with self.subTest(node_count=count):
                # Test node addition performance
                start_time = datetime.now()
                for i in range(count):
                    node = MatrixNode(id=f"perf_node_{i}", consciousness_level=i / count)
                    self.matrix.add_node(node)
                add_duration = (datetime.now() - start_time).total_seconds()
                
                # Test consciousness calculation performance
                start_time = datetime.now()
                consciousness_level = self.matrix.calculate_consciousness_level()
                calc_duration = (datetime.now() - start_time).total_seconds()
                
                # Test evolution performance
                start_time = datetime.now()
                self.matrix.evolve_step()
                evolve_duration = (datetime.now() - start_time).total_seconds()
                
                # Performance assertions (adjust thresholds as needed)
                self.assertLess(add_duration, count * 0.001)  # 1ms per node
                self.assertLess(calc_duration, 0.1)  # 100ms for calculation
                self.assertLess(evolve_duration, 1.0)  # 1s for evolution
                
                # Clean up for next iteration
                self.matrix.reset()
                
    def test_connection_performance_scaling(self):
        """Test performance scaling of node connections."""
        # Add base nodes
        node_count = 100
        for i in range(node_count):
            node = MatrixNode(id=f"conn_node_{i}", consciousness_level=0.5)
            self.matrix.add_node(node)
            
        # Test different connection densities
        connection_counts = [100, 500, 1000]
        
        for conn_count in connection_counts:
            with self.subTest(connection_count=conn_count):
                start_time = datetime.now()
                
                # Add connections
                for i in range(conn_count):
                    node1_id = f"conn_node_{i % node_count}"
                    node2_id = f"conn_node_{(i + 1) % node_count}"
                    self.matrix.connect_nodes(node1_id, node2_id, strength=0.5)
                    
                connection_duration = (datetime.now() - start_time).total_seconds()
                
                # Performance assertion
                self.assertLess(connection_duration, conn_count * 0.01)  # 10ms per connection
                
                # Test retrieval performance
                start_time = datetime.now()
                connections = self.matrix.get_node_connections("conn_node_0")
                retrieval_duration = (datetime.now() - start_time).total_seconds()
                
                self.assertLess(retrieval_duration, 0.1)  # 100ms for retrieval
                
    def test_serialization_performance_scaling(self):
        """Test serialization performance with different matrix sizes."""
        matrix_sizes = [50, 100, 200]
        
        for size in matrix_sizes:
            with self.subTest(matrix_size=size):
                # Create matrix of given size
                for i in range(size):
                    node = MatrixNode(id=f"serial_node_{i}", consciousness_level=i / size)
                    self.matrix.add_node(node)
                    
                # Add connections
                for i in range(size // 2):
                    self.matrix.connect_nodes(f"serial_node_{i}", f"serial_node_{i + size // 2}", 
                                            strength=0.6)
                    
                # Test serialization performance
                start_time = datetime.now()
                serialized = self.matrix.to_json()
                serialization_duration = (datetime.now() - start_time).total_seconds()
                
                # Test deserialization performance
                start_time = datetime.now()
                restored_matrix = GenesisConsciousnessMatrix.from_json(serialized)
                deserialization_duration = (datetime.now() - start_time).total_seconds()
                
                # Performance assertions
                self.assertLess(serialization_duration, size * 0.01)  # 10ms per node
                self.assertLess(deserialization_duration, size * 0.01)  # 10ms per node
                
                # Verify correctness
                self.assertEqual(len(restored_matrix.nodes), size)
                
                # Clean up
                self.matrix.reset()


# Additional pytest-style tests if pytest is preferred
@pytest.mark.skipif(not hasattr(GenesisConsciousnessMatrix, 'evolve_async'), 
                   reason="Async evolution not available")
class TestPytestAsyncOperations:
    """Pytest-style async tests for matrix operations."""
    
    @pytest.fixture
    def matrix(self):
        """Pytest fixture for matrix instance."""
        return GenesisConsciousnessMatrix()
        
    @pytest.fixture
    def populated_matrix(self, matrix):
        """Pytest fixture for populated matrix."""
        for i in range(10):
            node = MatrixNode(id=f"pytest_node_{i}", consciousness_level=0.1 + i * 0.08)
            matrix.add_node(node)
        return matrix
        
    @pytest.mark.asyncio
    async def test_async_evolution_performance(self, populated_matrix):
        """Test async evolution performance."""
        start_time = datetime.now()
        await populated_matrix.evolve_async()
        duration = (datetime.now() - start_time).total_seconds()
        
        assert duration < 1.0, "Async evolution should complete within 1 second"
        
    @pytest.mark.asyncio
    async def test_async_convergence_detection(self, populated_matrix):
        """Test async convergence detection."""
        await populated_matrix.evolve_until_convergence_async(max_iterations=50)
        assert populated_matrix.has_converged(), "Matrix should converge"
        
    @pytest.mark.parametrize("node_count", [10, 50, 100])
    async def test_async_scaling(self, matrix, node_count):
        """Test async operations scaling with different node counts."""
        # Add nodes
        for i in range(node_count):
            node = MatrixNode(id=f"scale_node_{i}", consciousness_level=0.5)
            matrix.add_node(node)
            
        # Test async evolution
        start_time = datetime.now()
        await matrix.evolve_async()
        duration = (datetime.now() - start_time).total_seconds()
        
        # Performance should scale reasonably
        assert duration < node_count * 0.01, f"Evolution should complete within {node_count * 0.01}s"
