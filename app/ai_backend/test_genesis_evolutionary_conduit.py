import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime
import json
import tempfile
import shutil

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genesis_evolutionary_conduit import (
    GenesisEvolutionaryConduit,
    EvolutionaryAgent,
    GeneticAlgorithm,
    FitnessEvaluator,
    MutationOperator,
    CrossoverOperator,
    SelectionOperator,
    PopulationManager,
    EvolutionaryException,
    ConduitInitializationError,
    AgentEvolutionError,
    PopulationEvolutionError
)


class TestGenesisEvolutionaryConduit(unittest.TestCase):
    """Test suite for GenesisEvolutionaryConduit class."""
    
    def setUp(self):
        """
        Initializes a GenesisEvolutionaryConduit instance and test parameters before each test.
        """
        self.conduit = GenesisEvolutionaryConduit()
        self.mock_population_size = 100
        self.mock_generation_limit = 50
        self.mock_mutation_rate = 0.1
        self.mock_crossover_rate = 0.8
        
    def tearDown(self):
        """
        Removes the temporary directory created during the test, if it exists.
        """
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_default_parameters(self):
        """
        Test that GenesisEvolutionaryConduit initializes correctly with default parameter values.
        """
        conduit = GenesisEvolutionaryConduit()
        self.assertIsNotNone(conduit)
        self.assertEqual(conduit.population_size, 100)
        self.assertEqual(conduit.generation_limit, 1000)
        self.assertEqual(conduit.mutation_rate, 0.01)
        self.assertEqual(conduit.crossover_rate, 0.7)
        
    def test_initialization_custom_parameters(self):
        """
        Verify that the GenesisEvolutionaryConduit initializes correctly with custom population size, generation limit, mutation rate, and crossover rate parameters.
        """
        conduit = GenesisEvolutionaryConduit(
            population_size=self.mock_population_size,
            generation_limit=self.mock_generation_limit,
            mutation_rate=self.mock_mutation_rate,
            crossover_rate=self.mock_crossover_rate
        )
        self.assertEqual(conduit.population_size, self.mock_population_size)
        self.assertEqual(conduit.generation_limit, self.mock_generation_limit)
        self.assertEqual(conduit.mutation_rate, self.mock_mutation_rate)
        self.assertEqual(conduit.crossover_rate, self.mock_crossover_rate)
        
    def test_initialization_invalid_population_size(self):
        """
        Test that initializing GenesisEvolutionaryConduit with a non-positive population size raises ConduitInitializationError.
        """
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(population_size=0)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(population_size=-10)
            
    def test_initialization_invalid_generation_limit(self):
        """
        Test that initializing GenesisEvolutionaryConduit with a non-positive generation limit raises ConduitInitializationError.
        """
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(generation_limit=0)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(generation_limit=-5)
            
    def test_initialization_invalid_mutation_rate(self):
        """
        Test that initializing GenesisEvolutionaryConduit with an invalid mutation rate raises ConduitInitializationError.
        """
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(mutation_rate=-0.1)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(mutation_rate=1.1)
            
    def test_initialization_invalid_crossover_rate(self):
        """
        Test that initializing GenesisEvolutionaryConduit with an invalid crossover rate raises ConduitInitializationError.
        """
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(crossover_rate=-0.1)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(crossover_rate=1.1)
            
    @patch('genesis_evolutionary_conduit.PopulationManager')
    def test_initialize_population_success(self, mock_population_manager):
        """
        Test that `initialize_population` successfully creates and returns a population using the mocked `PopulationManager`.
        """
        mock_population = Mock()
        mock_population_manager.return_value = mock_population
        
        result = self.conduit.initialize_population()
        
        self.assertEqual(result, mock_population)
        mock_population_manager.assert_called_once_with(
            size=self.conduit.population_size,
            fitness_evaluator=self.conduit.fitness_evaluator
        )
        
    @patch('genesis_evolutionary_conduit.PopulationManager')
    def test_initialize_population_failure(self, mock_population_manager):
        """
        Test that population initialization raises PopulationEvolutionError when the population manager fails.
        """
        mock_population_manager.side_effect = Exception("Population initialization failed")
        
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.initialize_population()
            
    @patch('genesis_evolutionary_conduit.random.random')
    def test_mutate_agent_success(self, mock_random):
        """
        Test that an agent is successfully mutated when the random value is below the mutation rate.
        """
        mock_random.return_value = 0.005  # Below mutation rate
        mock_agent = Mock()
        mock_agent.mutate = Mock()
        
        result = self.conduit.mutate_agent(mock_agent)
        
        self.assertEqual(result, mock_agent)
        mock_agent.mutate.assert_called_once()
        
    @patch('genesis_evolutionary_conduit.random.random')
    def test_mutate_agent_no_mutation(self, mock_random):
        """
        Test that an agent is not mutated when the random probability exceeds the mutation rate.
        
        Ensures the mutation method is not called and the original agent is returned unchanged.
        """
        mock_random.return_value = 0.5  # Above mutation rate
        mock_agent = Mock()
        mock_agent.mutate = Mock()
        
        result = self.conduit.mutate_agent(mock_agent)
        
        self.assertEqual(result, mock_agent)
        mock_agent.mutate.assert_not_called()
        
    def test_mutate_agent_mutation_failure(self):
        """
        Test that mutate_agent raises AgentEvolutionError when the agent's mutate method fails.
        """
        mock_agent = Mock()
        mock_agent.mutate.side_effect = Exception("Mutation failed")
        
        with patch('genesis_evolutionary_conduit.random.random', return_value=0.005):
            with self.assertRaises(AgentEvolutionError):
                self.conduit.mutate_agent(mock_agent)
                
    def test_crossover_agents_success(self):
        """
        Test that the crossover_agents method returns the expected offspring when crossover is successful.
        """
        mock_parent1 = Mock()
        mock_parent2 = Mock()
        mock_offspring = Mock()
        
        with patch('genesis_evolutionary_conduit.CrossoverOperator') as mock_crossover:
            mock_crossover.return_value.crossover.return_value = mock_offspring
            
            result = self.conduit.crossover_agents(mock_parent1, mock_parent2)
            
            self.assertEqual(result, mock_offspring)
            mock_crossover.assert_called_once()
            
    def test_crossover_agents_failure(self):
        """
        Test that agent crossover raises AgentEvolutionError when the crossover operation fails.
        """
        mock_parent1 = Mock()
        mock_parent2 = Mock()
        
        with patch('genesis_evolutionary_conduit.CrossoverOperator') as mock_crossover:
            mock_crossover.return_value.crossover.side_effect = Exception("Crossover failed")
            
            with self.assertRaises(AgentEvolutionError):
                self.conduit.crossover_agents(mock_parent1, mock_parent2)
                
    def test_select_parents_success(self):
        """
        Test that parent selection returns two agents when selection is successful.
        """
        mock_population = [Mock() for _ in range(10)]
        mock_selector = Mock()
        mock_selector.select.return_value = [mock_population[0], mock_population[1]]
        
        with patch('genesis_evolutionary_conduit.SelectionOperator', return_value=mock_selector):
            result = self.conduit.select_parents(mock_population)
            
            self.assertEqual(len(result), 2)
            mock_selector.select.assert_called_once_with(mock_population, 2)
            
    def test_select_parents_empty_population(self):
        """
        Test that selecting parents from an empty population raises a PopulationEvolutionError.
        """
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.select_parents([])
            
    def test_select_parents_insufficient_population(self):
        """
        Test that selecting parents from a population with insufficient size raises a PopulationEvolutionError.
        """
        mock_population = [Mock()]
        
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.select_parents(mock_population)
            
    def test_evolve_generation_success(self):
        """
        Test that evolving a generation produces a new population of the same size as the input population.
        """
        mock_population = [Mock() for _ in range(10)]
        mock_new_population = [Mock() for _ in range(10)]
        
        with patch.object(self.conduit, 'select_parents') as mock_select:
            with patch.object(self.conduit, 'crossover_agents') as mock_crossover:
                with patch.object(self.conduit, 'mutate_agent') as mock_mutate:
                    mock_select.return_value = [mock_population[0], mock_population[1]]
                    mock_crossover.return_value = mock_new_population[0]
                    mock_mutate.side_effect = lambda x: x
                    
                    result = self.conduit.evolve_generation(mock_population)
                    
                    self.assertEqual(len(result), len(mock_population))
                    
    def test_evolve_generation_failure(self):
        """
        Test that evolve_generation raises PopulationEvolutionError when parent selection fails.
        """
        mock_population = [Mock() for _ in range(10)]
        
        with patch.object(self.conduit, 'select_parents') as mock_select:
            mock_select.side_effect = Exception("Selection failed")
            
            with self.assertRaises(PopulationEvolutionError):
                self.conduit.evolve_generation(mock_population)
                
    def test_evaluate_fitness_success(self):
        """
        Test that the fitness evaluation method returns the expected fitness value for a valid agent.
        """
        mock_agent = Mock()
        mock_agent.fitness = 0.8
        
        with patch.object(self.conduit, 'fitness_evaluator') as mock_evaluator:
            mock_evaluator.evaluate.return_value = 0.8
            
            result = self.conduit.evaluate_fitness(mock_agent)
            
            self.assertEqual(result, 0.8)
            mock_evaluator.evaluate.assert_called_once_with(mock_agent)
            
    def test_evaluate_fitness_failure(self):
        """
        Test that `evaluate_fitness` raises `AgentEvolutionError` when the fitness evaluator fails.
        """
        mock_agent = Mock()
        
        with patch.object(self.conduit, 'fitness_evaluator') as mock_evaluator:
            mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")
            
            with self.assertRaises(AgentEvolutionError):
                self.conduit.evaluate_fitness(mock_agent)
                
    def test_run_evolution_success(self):
        """
        Test that the evolution run completes successfully and returns the evolved population when convergence is achieved.
        """
        mock_population = [Mock() for _ in range(5)]
        mock_evolved_population = [Mock() for _ in range(5)]
        
        with patch.object(self.conduit, 'initialize_population') as mock_init:
            with patch.object(self.conduit, 'evolve_generation') as mock_evolve:
                with patch.object(self.conduit, 'check_convergence') as mock_converge:
                    mock_init.return_value = mock_population
                    mock_evolve.return_value = mock_evolved_population
                    mock_converge.return_value = True
                    
                    result = self.conduit.run_evolution()
                    
                    self.assertEqual(result, mock_evolved_population)
                    mock_init.assert_called_once()
                    mock_evolve.assert_called()
                    
    def test_run_evolution_max_generations(self):
        """
        Test that the evolution run stops after reaching the maximum number of generations when convergence does not occur.
        """
        mock_population = [Mock() for _ in range(5)]
        
        with patch.object(self.conduit, 'initialize_population') as mock_init:
            with patch.object(self.conduit, 'evolve_generation') as mock_evolve:
                with patch.object(self.conduit, 'check_convergence') as mock_converge:
                    mock_init.return_value = mock_population
                    mock_evolve.return_value = mock_population
                    mock_converge.return_value = False
                    
                    result = self.conduit.run_evolution()
                    
                    self.assertEqual(result, mock_population)
                    self.assertEqual(mock_evolve.call_count, self.conduit.generation_limit)
                    
    def test_check_convergence_converged(self):
        """
        Test that the convergence check returns True when the population is considered converged.
        """
        mock_agent = Mock()
        mock_agent.fitness = 0.95
        mock_population = [mock_agent] * 10
        
        result = self.conduit.check_convergence(mock_population)
        
        self.assertTrue(result)
        
    def test_check_convergence_not_converged(self):
        """
        Test that the convergence check returns False when the population has not converged.
        
        Creates a mock population with varying fitness values and verifies that the conduit does not report convergence.
        """
        mock_agents = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_agents):
            agent.fitness = i * 0.1
        
        result = self.conduit.check_convergence(mock_agents)
        
        self.assertFalse(result)
        
    def test_get_best_agent_success(self):
        """
        Test that the best agent is correctly retrieved from a population based on highest fitness.
        """
        mock_agents = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_agents):
            agent.fitness = i * 0.1
        
        result = self.conduit.get_best_agent(mock_agents)
        
        self.assertEqual(result, mock_agents[-1])  # Highest fitness
        
    def test_get_best_agent_empty_population(self):
        """
        Test that retrieving the best agent from an empty population raises a PopulationEvolutionError.
        """
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.get_best_agent([])
            
    def test_save_evolution_state_success(self):
        """
        Test that the evolution state is saved successfully to a file using mocked file and JSON operations.
        """
        self.temp_dir = tempfile.mkdtemp()
        state_file = os.path.join(self.temp_dir, 'evolution_state.json')
        
        mock_population = [Mock() for _ in range(5)]
        generation = 10
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json:
                self.conduit.save_evolution_state(mock_population, generation, state_file)
                
                mock_file.assert_called_once_with(state_file, 'w')
                mock_json.assert_called_once()
                
    def test_load_evolution_state_success(self):
        """
        Test that the evolution state is loaded successfully from a file and matches the expected state.
        """
        self.temp_dir = tempfile.mkdtemp()
        state_file = os.path.join(self.temp_dir, 'evolution_state.json')
        
        mock_state = {
            'population': [],
            'generation': 10,
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.load', return_value=mock_state) as mock_json:
                result = self.conduit.load_evolution_state(state_file)
                
                self.assertEqual(result, mock_state)
                mock_file.assert_called_once_with(state_file, 'r')
                mock_json.assert_called_once()
                
    def test_load_evolution_state_file_not_found(self):
        """
        Test that loading the evolution state from a non-existent file raises a FileNotFoundError.
        """
        non_existent_file = '/non/existent/file.json'
        
        with self.assertRaises(FileNotFoundError):
            self.conduit.load_evolution_state(non_existent_file)
            
    def test_reset_evolution_state(self):
        """
        Test that resetting the evolution state sets the current generation and best fitness to their initial values.
        """
        self.conduit.current_generation = 50
        self.conduit.best_fitness = 0.8
        
        self.conduit.reset_evolution_state()
        
        self.assertEqual(self.conduit.current_generation, 0)
        self.assertEqual(self.conduit.best_fitness, 0.0)
        
    def test_get_evolution_statistics(self):
        """
        Test that evolution statistics are correctly retrieved from a population.
        
        Verifies that the returned statistics dictionary includes generation, population size, best, average, and worst fitness values.
        """
        mock_population = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_population):
            agent.fitness = i * 0.1
            
        stats = self.conduit.get_evolution_statistics(mock_population)
        
        self.assertIn('generation', stats)
        self.assertIn('population_size', stats)
        self.assertIn('best_fitness', stats)
        self.assertIn('average_fitness', stats)
        self.assertIn('worst_fitness', stats)
        
    def test_validate_parameters_valid(self):
        """
        Test that parameter validation succeeds with valid conduit parameters.
        """
        # Should not raise any exceptions
        self.conduit.validate_parameters()
        
    def test_validate_parameters_invalid_population_size(self):
        """
        Test that parameter validation raises ConduitInitializationError when population size is invalid.
        """
        self.conduit.population_size = -10
        
        with self.assertRaises(ConduitInitializationError):
            self.conduit.validate_parameters()
            
    def test_validate_parameters_invalid_rates(self):
        """
        Test that parameter validation raises ConduitInitializationError when mutation_rate is set to an invalid value.
        """
        self.conduit.mutation_rate = 1.5
        
        with self.assertRaises(ConduitInitializationError):
            self.conduit.validate_parameters()


class TestEvolutionaryAgent(unittest.TestCase):
    """Test suite for EvolutionaryAgent class."""
    
    def setUp(self):
        """
        Initialize a new EvolutionaryAgent instance before each test.
        """
        self.agent = EvolutionaryAgent()
        
    def test_initialization_default(self):
        """
        Test that an EvolutionaryAgent is initialized with a genome, default fitness of 0.0, and a unique ID.
        """
        agent = EvolutionaryAgent()
        self.assertIsNotNone(agent.genome)
        self.assertEqual(agent.fitness, 0.0)
        self.assertIsNotNone(agent.id)
        
    def test_initialization_custom_genome(self):
        """
        Test that an EvolutionaryAgent is initialized with a provided custom genome.
        """
        custom_genome = [1, 0, 1, 1, 0]
        agent = EvolutionaryAgent(genome=custom_genome)
        self.assertEqual(agent.genome, custom_genome)
        
    def test_mutate_success(self):
        """
        Test that mutating an agent successfully changes its genome.
        """
        original_genome = self.agent.genome.copy()
        self.agent.mutate()
        
        # Genome should be modified
        self.assertNotEqual(self.agent.genome, original_genome)
        
    def test_mutate_with_rate(self):
        """
        Test that mutating an agent with a 100% mutation rate always changes its genome.
        """
        original_genome = self.agent.genome.copy()
        self.agent.mutate(mutation_rate=1.0)  # 100% mutation rate
        
        # With 100% rate, genome should definitely change
        self.assertNotEqual(self.agent.genome, original_genome)
        
    def test_crossover_success(self):
        """
        Test that the crossover operation between two agents produces a valid offspring with a genome of the expected length.
        """
        other_agent = EvolutionaryAgent()
        offspring = self.agent.crossover(other_agent)
        
        self.assertIsInstance(offspring, EvolutionaryAgent)
        self.assertEqual(len(offspring.genome), len(self.agent.genome))
        
    def test_crossover_invalid_agent(self):
        """
        Test that attempting to perform crossover with an invalid agent raises an AgentEvolutionError.
        """
        with self.assertRaises(AgentEvolutionError):
            self.agent.crossover(None)
            
    def test_evaluate_fitness_success(self):
        """
        Test that the agent's fitness is correctly evaluated and assigned using a mock evaluator.
        """
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = 0.8
        
        self.agent.evaluate_fitness(mock_evaluator)
        
        self.assertEqual(self.agent.fitness, 0.8)
        mock_evaluator.evaluate.assert_called_once_with(self.agent)
        
    def test_clone_success(self):
        """
        Test that cloning an agent produces a distinct agent with the same genome and fitness but a different ID.
        """
        self.agent.fitness = 0.7
        clone = self.agent.clone()
        
        self.assertIsNot(clone, self.agent)
        self.assertEqual(clone.genome, self.agent.genome)
        self.assertEqual(clone.fitness, self.agent.fitness)
        self.assertNotEqual(clone.id, self.agent.id)
        
    def test_to_dict_success(self):
        """
        Test that converting an agent to a dictionary includes 'id', 'genome', and 'fitness' fields with correct values.
        """
        self.agent.fitness = 0.6
        agent_dict = self.agent.to_dict()
        
        self.assertIn('id', agent_dict)
        self.assertIn('genome', agent_dict)
        self.assertIn('fitness', agent_dict)
        self.assertEqual(agent_dict['fitness'], 0.6)
        
    def test_from_dict_success(self):
        """
        Test that an EvolutionaryAgent instance is correctly created from a valid dictionary representation.
        """
        agent_dict = {
            'id': 'test_id',
            'genome': [1, 0, 1, 0],
            'fitness': 0.5
        }
        
        agent = EvolutionaryAgent.from_dict(agent_dict)
        
        self.assertEqual(agent.id, 'test_id')
        self.assertEqual(agent.genome, [1, 0, 1, 0])
        self.assertEqual(agent.fitness, 0.5)
        
    def test_from_dict_missing_fields(self):
        """
        Test that creating an EvolutionaryAgent from a dictionary missing required fields raises an AgentEvolutionError.
        """
        agent_dict = {'genome': [1, 0, 1, 0]}
        
        with self.assertRaises(AgentEvolutionError):
            EvolutionaryAgent.from_dict(agent_dict)


class TestGeneticAlgorithm(unittest.TestCase):
    """Test suite for GeneticAlgorithm class."""
    
    def setUp(self):
        """
        Set up the GeneticAlgorithm instance before each test.
        """
        self.algorithm = GeneticAlgorithm()
        
    def test_initialization_default(self):
        """
        Test that the GeneticAlgorithm initializes with default selection, crossover, and mutation operators.
        """
        algorithm = GeneticAlgorithm()
        self.assertIsNotNone(algorithm.selection_operator)
        self.assertIsNotNone(algorithm.crossover_operator)
        self.assertIsNotNone(algorithm.mutation_operator)
        
    def test_run_algorithm_success(self):
        """
        Test that the genetic algorithm successfully produces a new population of the same size as the input when all operators function correctly.
        """
        mock_population = [Mock() for _ in range(10)]
        
        with patch.object(self.algorithm, 'selection_operator') as mock_selection:
            with patch.object(self.algorithm, 'crossover_operator') as mock_crossover:
                with patch.object(self.algorithm, 'mutation_operator') as mock_mutation:
                    mock_selection.select.return_value = mock_population[:2]
                    mock_crossover.crossover.return_value = Mock()
                    mock_mutation.mutate.side_effect = lambda x: x
                    
                    result = self.algorithm.run_algorithm(mock_population)
                    
                    self.assertIsInstance(result, list)
                    self.assertEqual(len(result), len(mock_population))
                    
    def test_run_algorithm_empty_population(self):
        """
        Test that running the genetic algorithm with an empty population raises a PopulationEvolutionError.
        """
        with self.assertRaises(PopulationEvolutionError):
            self.algorithm.run_algorithm([])


class TestFitnessEvaluator(unittest.TestCase):
    """Test suite for FitnessEvaluator class."""
    
    def setUp(self):
        """
        Set up the FitnessEvaluator instance before each test.
        """
        self.evaluator = FitnessEvaluator()
        
    def test_evaluate_agent_success(self):
        """
        Test that the fitness evaluator successfully computes a valid fitness score for an agent.
        
        Verifies that the returned fitness is a float within the range [0.0, 1.0].
        """
        mock_agent = Mock()
        mock_agent.genome = [1, 0, 1, 1, 0]
        
        fitness = self.evaluator.evaluate(mock_agent)
        
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        
    def test_evaluate_agent_none(self):
        """
        Test that evaluating a None agent raises an AgentEvolutionError.
        """
        with self.assertRaises(AgentEvolutionError):
            self.evaluator.evaluate(None)
            
    def test_evaluate_population_success(self):
        """
        Test that evaluating a population returns a list of float fitness scores, one for each agent.
        """
        mock_population = [Mock() for _ in range(5)]
        for agent in mock_population:
            agent.genome = [1, 0, 1, 0, 1]
            
        fitness_scores = self.evaluator.evaluate_population(mock_population)
        
        self.assertEqual(len(fitness_scores), 5)
        self.assertTrue(all(isinstance(score, float) for score in fitness_scores))
        
    def test_evaluate_population_empty(self):
        """
        Test that evaluating an empty population returns an empty list of fitness scores.
        """
        fitness_scores = self.evaluator.evaluate_population([])
        self.assertEqual(len(fitness_scores), 0)


class TestMutationOperator(unittest.TestCase):
    """Test suite for MutationOperator class."""
    
    def setUp(self):
        """
        Set up the MutationOperator instance before each test.
        """
        self.operator = MutationOperator()
        
    def test_mutate_agent_success(self):
        """
        Test that the mutation operator successfully mutates an agent and returns the same agent instance.
        """
        mock_agent = Mock()
        mock_agent.genome = [1, 0, 1, 0, 1]
        
        result = self.operator.mutate(mock_agent)
        
        self.assertEqual(result, mock_agent)
        
    def test_mutate_agent_with_rate(self):
        """
        Test that the mutation operator mutates an agent using a specified mutation rate.
        """
        mock_agent = Mock()
        mock_agent.genome = [1, 0, 1, 0, 1]
        
        result = self.operator.mutate(mock_agent, mutation_rate=0.5)
        
        self.assertEqual(result, mock_agent)
        
    def test_mutate_agent_none(self):
        """
        Test that mutating a None agent raises an AgentEvolutionError.
        """
        with self.assertRaises(AgentEvolutionError):
            self.operator.mutate(None)


class TestCrossoverOperator(unittest.TestCase):
    """Test suite for CrossoverOperator class."""
    
    def setUp(self):
        """
        Set up the test fixture by initializing a CrossoverOperator instance before each test.
        """
        self.operator = CrossoverOperator()
        
    def test_crossover_agents_success(self):
        """
        Test that the crossover operator successfully produces an offspring agent from two parent agents.
        """
        mock_parent1 = Mock()
        mock_parent2 = Mock()
        mock_parent1.genome = [1, 0, 1, 0, 1]
        mock_parent2.genome = [0, 1, 0, 1, 0]
        
        offspring = self.operator.crossover(mock_parent1, mock_parent2)
        
        self.assertIsNotNone(offspring)
        
    def test_crossover_agents_none_parents(self):
        """
        Test that the crossover operator raises an AgentEvolutionError when both parent agents are None.
        """
        with self.assertRaises(AgentEvolutionError):
            self.operator.crossover(None, None)
            
    def test_crossover_agents_single_none_parent(self):
        """
        Test that the crossover operation raises an AgentEvolutionError when one parent is None.
        """
        mock_parent = Mock()
        mock_parent.genome = [1, 0, 1, 0, 1]
        
        with self.assertRaises(AgentEvolutionError):
            self.operator.crossover(mock_parent, None)


class TestSelectionOperator(unittest.TestCase):
    """Test suite for SelectionOperator class."""
    
    def setUp(self):
        """
        Set up the SelectionOperator instance before each test.
        """
        self.operator = SelectionOperator()
        
    def test_select_agents_success(self):
        """
        Test that the selection operator successfully selects the specified number of agents from a population.
        
        Verifies that the selected agents are members of the original population and that the correct number of agents is returned.
        """
        mock_population = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_population):
            agent.fitness = i * 0.1
            
        selected = self.operator.select(mock_population, 2)
        
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(agent in mock_population for agent in selected))
        
    def test_select_agents_empty_population(self):
        """
        Test that selecting agents from an empty population raises a PopulationEvolutionError.
        """
        with self.assertRaises(PopulationEvolutionError):
            self.operator.select([], 2)
            
    def test_select_agents_invalid_count(self):
        """
        Test that selecting agents with an invalid count raises a PopulationEvolutionError.
        """
        mock_population = [Mock() for _ in range(5)]
        
        with self.assertRaises(PopulationEvolutionError):
            self.operator.select(mock_population, 0)
            
    def test_select_agents_count_exceeds_population(self):
        """
        Test that selecting more agents than the population size raises a PopulationEvolutionError.
        """
        mock_population = [Mock() for _ in range(5)]
        
        with self.assertRaises(PopulationEvolutionError):
            self.operator.select(mock_population, 10)


class TestPopulationManager(unittest.TestCase):
    """Test suite for PopulationManager class."""
    
    def setUp(self):
        """
        Set up a PopulationManager instance with a population size of 10 before each test.
        """
        self.manager = PopulationManager(size=10)
        
    def test_initialization_success(self):
        """
        Test that the PopulationManager initializes successfully with a specified size and a fitness evaluator.
        """
        manager = PopulationManager(size=20)
        self.assertEqual(manager.size, 20)
        self.assertIsNotNone(manager.fitness_evaluator)
        
    def test_initialization_invalid_size(self):
        """
        Test that initializing a PopulationManager with an invalid size raises a PopulationEvolutionError.
        """
        with self.assertRaises(PopulationEvolutionError):
            PopulationManager(size=0)
            
    def test_generate_population_success(self):
        """
        Test that `generate_population` creates a population of the correct size with all elements as `EvolutionaryAgent` instances.
        """
        population = self.manager.generate_population()
        
        self.assertEqual(len(population), 10)
        self.assertTrue(all(isinstance(agent, EvolutionaryAgent) for agent in population))
        
    def test_evaluate_population_success(self):
        """
        Test that evaluating a population assigns the expected fitness value to each agent.
        """
        mock_population = [Mock() for _ in range(5)]
        
        with patch.object(self.manager.fitness_evaluator, 'evaluate', return_value=0.5):
            self.manager.evaluate_population(mock_population)
            
            for agent in mock_population:
                self.assertEqual(agent.fitness, 0.5)
                
    def test_sort_population_by_fitness(self):
        """
        Test that the population is correctly sorted in descending order by agent fitness.
        """
        mock_population = [Mock() for _ in range(5)]
        for i, agent in enumerate(mock_population):
            agent.fitness = (4 - i) * 0.2  # Reverse order
            
        sorted_population = self.manager.sort_population_by_fitness(mock_population)
        
        # Should be sorted in descending order
        for i in range(len(sorted_population) - 1):
            self.assertGreaterEqual(sorted_population[i].fitness, sorted_population[i + 1].fitness)
            
    def test_get_population_statistics(self):
        """
        Test that population statistics are correctly calculated and returned for a given population.
        
        Verifies that the statistics dictionary includes size, best, average, and worst fitness values, and that the population size is accurate.
        """
        mock_population = [Mock() for _ in range(5)]
        for i, agent in enumerate(mock_population):
            agent.fitness = i * 0.2
            
        stats = self.manager.get_population_statistics(mock_population)
        
        self.assertIn('size', stats)
        self.assertIn('best_fitness', stats)
        self.assertIn('average_fitness', stats)
        self.assertIn('worst_fitness', stats)
        self.assertEqual(stats['size'], 5)


class TestEvolutionaryExceptions(unittest.TestCase):
    """Test suite for evolutionary exception classes."""
    
    def test_evolutionary_exception_creation(self):
        """Test creating EvolutionaryException."""
        exception = EvolutionaryException("Test message")
        self.assertEqual(str(exception), "Test message")
        
    def test_conduit_initialization_error_creation(self):
        """
        Test that a ConduitInitializationError is created with the correct message and inherits from EvolutionaryException.
        """
        exception = ConduitInitializationError("Initialization failed")
        self.assertEqual(str(exception), "Initialization failed")
        self.assertIsInstance(exception, EvolutionaryException)
        
    def test_agent_evolution_error_creation(self):
        """
        Test that an AgentEvolutionError is created with the correct message and inherits from EvolutionaryException.
        """
        exception = AgentEvolutionError("Agent evolution failed")
        self.assertEqual(str(exception), "Agent evolution failed")
        self.assertIsInstance(exception, EvolutionaryException)
        
    def test_population_evolution_error_creation(self):
        """
        Test that a PopulationEvolutionError is created with the correct message and inherits from EvolutionaryException.
        """
        exception = PopulationEvolutionError("Population evolution failed")
        self.assertEqual(str(exception), "Population evolution failed")
        self.assertIsInstance(exception, EvolutionaryException)


def mock_open(read_data=''):
    """
    Return a MagicMock object that simulates the built-in open function for file operations.
    
    Parameters:
        read_data (str): Optional string to simulate file contents.
    
    Returns:
        MagicMock: A mock object with the same interface as the built-in open function.
    """
    return MagicMock(spec=open)


if __name__ == '__main__':
    unittest.main()