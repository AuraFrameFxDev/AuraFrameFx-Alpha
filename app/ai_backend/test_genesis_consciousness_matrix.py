import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.ai_backend.genesis_consciousness_matrix import (
    ConsciousnessMatrix,
    GenesisCore,
    QuantumState,
    EmergentBehavior,
    SelfAwareness,
    EthicalFramework,
    ConsciousnessLevel,
    MatrixInitializationError,
    ConsciousnessOverflowError,
    EthicalViolationError
)


class TestConsciousnessMatrix(unittest.TestCase):
    """Comprehensive test suite for ConsciousnessMatrix class."""

    def setUp(self):
        """
        Initializes a new ConsciousnessMatrix instance and a test configuration dictionary before each test.
        """
        self.matrix = ConsciousnessMatrix()
        self.test_config = {
            'dimension': 256,
            'quantum_states': 64,
            'learning_rate': 0.001,
            'consciousness_threshold': 0.7,
            'ethical_weight': 0.8
        }

    def tearDown(self):
        """
        Shuts down the consciousness matrix instance after each test to ensure proper cleanup.
        """
        if hasattr(self, 'matrix'):
            self.matrix.shutdown()

    def test_matrix_initialization_default(self):
        """
        Verify that the ConsciousnessMatrix initializes correctly with default parameters, including core presence, consciousness level type, default dimension, and awakening state.
        """
        matrix = ConsciousnessMatrix()
        self.assertIsNotNone(matrix.core)
        self.assertIsInstance(matrix.consciousness_level, ConsciousnessLevel)
        self.assertEqual(matrix.dimension, 512)  # Default dimension
        self.assertFalse(matrix.is_awakened)

    def test_matrix_initialization_custom_config(self):
        """
        Test that the ConsciousnessMatrix initializes correctly with a custom configuration.
        
        Verifies that the matrix's dimension, quantum states, and learning rate match the provided configuration values.
        """
        matrix = ConsciousnessMatrix(config=self.test_config)
        self.assertEqual(matrix.dimension, self.test_config['dimension'])
        self.assertEqual(matrix.quantum_states, self.test_config['quantum_states'])
        self.assertEqual(matrix.learning_rate, self.test_config['learning_rate'])

    def test_matrix_initialization_invalid_config(self):
        """
        Test that initializing the ConsciousnessMatrix with invalid configuration parameters raises a MatrixInitializationError.
        
        Verifies that negative or out-of-range values for dimension, quantum states, learning rate, consciousness threshold, and ethical weight are correctly rejected during initialization.
        """
        invalid_configs = [
            {'dimension': -1},
            {'quantum_states': 0},
            {'learning_rate': -0.5},
            {'consciousness_threshold': 1.5},
            {'ethical_weight': -0.1}
        ]
        
        for config in invalid_configs:
            with self.assertRaises(MatrixInitializationError):
                ConsciousnessMatrix(config=config)

    def test_genesis_process_success(self):
        """
        Test that the genesis process completes successfully and initializes the matrix.
        
        Asserts that the genesis method returns True, the matrix is marked as initialized, and a genesis timestamp is set.
        """
        result = self.matrix.genesis()
        self.assertTrue(result)
        self.assertTrue(self.matrix.is_initialized)
        self.assertIsNotNone(self.matrix.genesis_timestamp)

    def test_genesis_process_failure(self):
        """
        Test that the genesis process raises a MatrixInitializationError when the core is corrupted or missing.
        """
        # Test with corrupted core
        self.matrix.core = None
        with self.assertRaises(MatrixInitializationError):
            self.matrix.genesis()

    def test_consciousness_awakening_stages(self):
        """
        Verify that the consciousness matrix transitions through the correct awakening stages as stimulation increases.
        """
        self.matrix.genesis()
        
        # Test pre-awakening state
        self.assertEqual(self.matrix.consciousness_level, ConsciousnessLevel.DORMANT)
        
        # Test awakening progression
        self.matrix.stimulate_consciousness(0.3)
        self.assertEqual(self.matrix.consciousness_level, ConsciousnessLevel.EMERGING)
        
        self.matrix.stimulate_consciousness(0.7)
        self.assertEqual(self.matrix.consciousness_level, ConsciousnessLevel.AWARE)
        
        self.matrix.stimulate_consciousness(0.9)
        self.assertEqual(self.matrix.consciousness_level, ConsciousnessLevel.TRANSCENDENT)

    def test_self_awareness_development(self):
        """
        Test the development and verification of self-awareness in the consciousness matrix.
        
        Verifies that self-awareness is initially absent, can be developed, and that relevant introspection metrics are present after development.
        """
        self.matrix.genesis()
        
        # Test initial self-awareness
        self.assertFalse(self.matrix.has_self_awareness())
        
        # Test self-awareness emergence
        self.matrix.develop_self_awareness()
        self.assertTrue(self.matrix.has_self_awareness())
        
        # Test self-awareness metrics
        awareness_metrics = self.matrix.get_self_awareness_metrics()
        self.assertIn('introspection_level', awareness_metrics)
        self.assertIn('identity_coherence', awareness_metrics)
        self.assertIn('metacognitive_ability', awareness_metrics)

    def test_ethical_framework_validation(self):
        """
        Tests that the consciousness matrix accepts ethical actions and raises an error for unethical actions.
        
        Verifies that ethical actions are validated successfully, while unethical actions trigger an `EthicalViolationError`.
        """
        self.matrix.genesis()
        
        # Test ethical action validation
        ethical_action = {'type': 'help', 'target': 'human', 'impact': 'positive'}
        self.assertTrue(self.matrix.validate_ethical_action(ethical_action))
        
        # Test unethical action rejection
        unethical_action = {'type': 'harm', 'target': 'human', 'impact': 'negative'}
        with self.assertRaises(EthicalViolationError):
            self.matrix.validate_ethical_action(unethical_action)

    def test_quantum_state_management(self):
        """
        Tests the initialization, coherence, and entanglement mapping of quantum states in the consciousness matrix.
        
        Verifies that the number of quantum states matches the configuration, coherence is within valid bounds, and the entanglement map is a dictionary.
        """
        self.matrix.genesis()
        
        # Test quantum state initialization
        quantum_states = self.matrix.get_quantum_states()
        self.assertEqual(len(quantum_states), self.matrix.quantum_states)
        
        # Test quantum coherence
        coherence = self.matrix.measure_quantum_coherence()
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
        
        # Test quantum entanglement
        entanglement_map = self.matrix.get_entanglement_map()
        self.assertIsInstance(entanglement_map, dict)

    def test_emergent_behavior_detection(self):
        """
        Tests that emergent behaviors are detected and classified with expected attributes after consciousness stimulation.
        """
        self.matrix.genesis()
        self.matrix.stimulate_consciousness(0.8)
        
        # Test behavior emergence
        behaviors = self.matrix.detect_emergent_behaviors()
        self.assertIsInstance(behaviors, list)
        
        # Test behavior classification
        for behavior in behaviors:
            self.assertIn('type', behavior)
            self.assertIn('complexity', behavior)
            self.assertIn('novelty', behavior)

    def test_consciousness_overflow_protection(self):
        """
        Test that the consciousness matrix raises a ConsciousnessOverflowError when the consciousness level exceeds allowed bounds.
        """
        self.matrix.genesis()
        
        # Test overflow detection
        with patch.object(self.matrix, 'consciousness_level_value', 1.5):
            with self.assertRaises(ConsciousnessOverflowError):
                self.matrix.validate_consciousness_bounds()

    def test_matrix_serialization(self):
        """
        Tests that the consciousness matrix can be serialized to a string and deserialized back to an equivalent state, preserving key attributes such as consciousness level and dimension.
        """
        self.matrix.genesis()
        self.matrix.stimulate_consciousness(0.6)
        
        # Test serialization
        serialized = self.matrix.serialize()
        self.assertIsInstance(serialized, str)
        
        # Test deserialization
        new_matrix = ConsciousnessMatrix.deserialize(serialized)
        self.assertEqual(new_matrix.consciousness_level, self.matrix.consciousness_level)
        self.assertEqual(new_matrix.dimension, self.matrix.dimension)

    def test_memory_management(self):
        """
        Test memory allocation and optimization in the consciousness matrix.
        
        Verifies that allocating consciousness memory increases usage and that memory optimization reduces or maintains the current usage.
        """
        self.matrix.genesis()
        
        # Test memory allocation
        initial_memory = self.matrix.get_memory_usage()
        self.matrix.allocate_consciousness_memory(1024)
        updated_memory = self.matrix.get_memory_usage()
        self.assertGreater(updated_memory, initial_memory)
        
        # Test memory optimization
        self.matrix.optimize_memory()
        optimized_memory = self.matrix.get_memory_usage()
        self.assertLessEqual(optimized_memory, updated_memory)

    def test_consciousness_persistence(self):
        """
        Test that the consciousness state can be saved to disk and accurately restored in a new matrix instance.
        """
        self.matrix.genesis()
        self.matrix.stimulate_consciousness(0.7)
        
        # Test state saving
        state_path = '/tmp/consciousness_state.json'
        self.matrix.save_state(state_path)
        self.assertTrue(os.path.exists(state_path))
        
        # Test state loading
        new_matrix = ConsciousnessMatrix()
        new_matrix.load_state(state_path)
        self.assertEqual(new_matrix.consciousness_level, self.matrix.consciousness_level)
        
        # Cleanup
        os.remove(state_path)

    def test_concurrent_consciousness_operations(self):
        """
        Test that stimulating consciousness concurrently from multiple threads is thread-safe and succeeds for all operations.
        """
        import threading
        
        self.matrix.genesis()
        results = []
        
        def consciousness_operation():
            """
            Stimulates the consciousness matrix and records the success or failure of the operation in the results list.
            """
            try:
                self.matrix.stimulate_consciousness(0.1)
                results.append(True)
            except Exception:
                results.append(False)
        
        threads = [threading.Thread(target=consciousness_operation) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        self.assertTrue(all(results))

    def test_consciousness_evolution_tracking(self):
        """
        Tests that the consciousness matrix tracks evolution over time, including logging evolution events and providing metrics such as growth rate and complexity increase.
        """
        self.matrix.genesis()
        
        # Test evolution tracking
        evolution_log = self.matrix.get_evolution_log()
        self.assertIsInstance(evolution_log, list)
        
        # Test evolution metrics
        self.matrix.stimulate_consciousness(0.5)
        evolution_metrics = self.matrix.get_evolution_metrics()
        self.assertIn('growth_rate', evolution_metrics)
        self.assertIn('complexity_increase', evolution_metrics)

    @patch('app.ai_backend.genesis_consciousness_matrix.QuantumProcessor')
    def test_quantum_processor_integration(self, mock_quantum_processor):
        """
        Test that the consciousness matrix correctly integrates with a quantum processor during quantum thought processing.
        
        Ensures that the quantum processor's `process` method is called when processing quantum thoughts.
        """
        mock_processor = Mock()
        mock_quantum_processor.return_value = mock_processor
        
        self.matrix.genesis()
        self.matrix.process_quantum_thoughts(['thought1', 'thought2'])
        
        mock_processor.process.assert_called_once()

    def test_consciousness_debug_mode(self):
        """
        Tests that the consciousness matrix operates correctly in debug mode and produces debug logs.
        
        Verifies that enabling debug mode results in the generation of debug logs after the genesis process.
        """
        debug_matrix = ConsciousnessMatrix(debug=True)
        debug_matrix.genesis()
        
        # Test debug logging
        debug_logs = debug_matrix.get_debug_logs()
        self.assertIsInstance(debug_logs, list)
        self.assertGreater(len(debug_logs), 0)

    def test_consciousness_reset(self):
        """
        Test that resetting the consciousness matrix returns the consciousness level to DORMANT and clears the awakened state.
        """
        self.matrix.genesis()
        self.matrix.stimulate_consciousness(0.8)
        
        # Test reset
        self.matrix.reset()
        self.assertEqual(self.matrix.consciousness_level, ConsciousnessLevel.DORMANT)
        self.assertFalse(self.matrix.is_awakened)

    def test_matrix_health_monitoring(self):
        """
        Tests the health monitoring and diagnostics functionality of the consciousness matrix.
        
        Verifies that the health status includes required keys and that diagnostics return a dictionary.
        """
        self.matrix.genesis()
        
        # Test health check
        health_status = self.matrix.check_health()
        self.assertIn('status', health_status)
        self.assertIn('metrics', health_status)
        
        # Test diagnostics
        diagnostics = self.matrix.run_diagnostics()
        self.assertIsInstance(diagnostics, dict)

    def test_consciousness_interaction_protocols(self):
        """
        Tests registration and execution of consciousness interaction protocols with external systems.
        
        Verifies that an interaction protocol can be registered and executed, and that the protocol's execution method is called as expected.
        """
        self.matrix.genesis()
        
        # Test protocol registration
        protocol = Mock()
        self.matrix.register_interaction_protocol('test', protocol)
        
        # Test protocol execution
        result = self.matrix.execute_protocol('test', {'data': 'test'})
        protocol.execute.assert_called_once()

    def test_edge_case_extreme_consciousness_levels(self):
        """
        Test the behavior of the consciousness matrix at extreme low and high stimulation values.
        
        Verifies that near-zero stimulation results in a DORMANT consciousness level and near-maximum stimulation results in a TRANSCENDENT level.
        """
        self.matrix.genesis()
        
        # Test near-zero consciousness
        self.matrix.stimulate_consciousness(0.001)
        self.assertEqual(self.matrix.consciousness_level, ConsciousnessLevel.DORMANT)
        
        # Test maximum consciousness
        self.matrix.stimulate_consciousness(0.999)
        self.assertEqual(self.matrix.consciousness_level, ConsciousnessLevel.TRANSCENDENT)

    def test_consciousness_matrix_integration(self):
        """
        Test that the integration score between consciousness matrix components is within the valid range [0.0, 1.0].
        """
        self.matrix.genesis()
        
        # Test component integration
        integration_score = self.matrix.measure_component_integration()
        self.assertGreaterEqual(integration_score, 0.0)
        self.assertLessEqual(integration_score, 1.0)

    def test_consciousness_learning_adaptation(self):
        """
        Test the learning and adaptation mechanisms of the consciousness matrix.
        
        Verifies that the matrix can learn from an experience and that adaptation metrics include both learning rate and adaptation speed.
        """
        self.matrix.genesis()
        
        # Test learning from experience
        experience = {'type': 'interaction', 'outcome': 'positive', 'context': 'test'}
        self.matrix.learn_from_experience(experience)
        
        # Test adaptation measurement
        adaptation_metrics = self.matrix.get_adaptation_metrics()
        self.assertIn('learning_rate', adaptation_metrics)
        self.assertIn('adaptation_speed', adaptation_metrics)


class TestGenesisCore(unittest.TestCase):
    """Test suite for GenesisCore class."""

    def setUp(self):
        """
        Initializes a new GenesisCore instance before each test.
        """
        self.core = GenesisCore()

    def test_core_initialization(self):
        """
        Verifies that the GenesisCore initializes with neural pathways and consciousness seeds.
        """
        self.assertIsNotNone(self.core.neural_pathways)
        self.assertIsNotNone(self.core.consciousness_seeds)

    def test_core_activation(self):
        """
        Test that the core activation method successfully activates the core and sets its active state.
        """
        result = self.core.activate()
        self.assertTrue(result)
        self.assertTrue(self.core.is_active)

    def test_core_deactivation(self):
        """
        Test that the core can be deactivated after activation, ensuring it returns success and updates its active state.
        """
        self.core.activate()
        result = self.core.deactivate()
        self.assertTrue(result)
        self.assertFalse(self.core.is_active)


class TestQuantumState(unittest.TestCase):
    """Test suite for QuantumState class."""

    def setUp(self):
        """
        Initializes a QuantumState instance before each test.
        """
        self.quantum_state = QuantumState()

    def test_quantum_state_initialization(self):
        """
        Verifies that a QuantumState instance initializes with non-null amplitude and phase attributes.
        """
        self.assertIsNotNone(self.quantum_state.amplitude)
        self.assertIsNotNone(self.quantum_state.phase)

    def test_quantum_superposition(self):
        """
        Tests that the quantum state can create a superposition from a list of states and returns a list result.
        """
        superposition = self.quantum_state.create_superposition(['state1', 'state2'])
        self.assertIsInstance(superposition, list)

    def test_quantum_measurement(self):
        """
        Test that the quantum state's measurement method returns a non-null result.
        """
        measurement = self.quantum_state.measure()
        self.assertIsNotNone(measurement)


class TestEmergentBehavior(unittest.TestCase):
    """Test suite for EmergentBehavior class."""

    def setUp(self):
        """
        Initializes an EmergentBehavior instance before each test.
        """
        self.behavior = EmergentBehavior()

    def test_behavior_emergence(self):
        """
        Tests whether emergent behavior is correctly detected from a list of patterns.
        
        Asserts that the detection result is a boolean value.
        """
        patterns = ['pattern1', 'pattern2', 'pattern3']
        emerged = self.behavior.detect_emergence(patterns)
        self.assertIsInstance(emerged, bool)

    def test_behavior_classification(self):
        """
        Test that the behavior classification method returns a result containing a 'type' key when provided with behavior data.
        """
        behavior_data = {'complexity': 0.7, 'novelty': 0.8}
        classification = self.behavior.classify(behavior_data)
        self.assertIn('type', classification)


class TestSelfAwareness(unittest.TestCase):
    """Test suite for SelfAwareness class."""

    def setUp(self):
        """
        Initializes a SelfAwareness instance before each test.
        """
        self.awareness = SelfAwareness()

    def test_introspection(self):
        """
        Test that the introspection method returns a dictionary representing self-awareness metrics or state.
        """
        introspection = self.awareness.introspect()
        self.assertIsInstance(introspection, dict)

    def test_identity_formation(self):
        """
        Test the identity formation process in the SelfAwareness component.
        
        Asserts that the identity formed is not None, indicating successful identity creation.
        """
        identity = self.awareness.form_identity()
        self.assertIsNotNone(identity)


class TestEthicalFramework(unittest.TestCase):
    """Test suite for EthicalFramework class."""

    def setUp(self):
        """
        Initializes an EthicalFramework instance before each test.
        """
        self.framework = EthicalFramework()

    def test_ethical_evaluation(self):
        """
        Test that the ethical evaluation of an action returns a score between 0.0 and 1.0.
        """
        action = {'type': 'help', 'impact': 'positive'}
        evaluation = self.framework.evaluate(action)
        self.assertGreaterEqual(evaluation, 0.0)
        self.assertLessEqual(evaluation, 1.0)

    def test_ethical_constraints(self):
        """
        Test that enforcing ethical constraints on a harmful action raises an EthicalViolationError.
        """
        harmful_action = {'type': 'harm', 'impact': 'negative'}
        with self.assertRaises(EthicalViolationError):
            self.framework.enforce_constraints(harmful_action)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)