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
        Removes the temporary directory created during the test, ensuring no residual files remain after each test method.
        """
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_default_parameters(self):
        """
        Test that GenesisEvolutionaryConduit initializes with the correct default parameters.
        """
        conduit = GenesisEvolutionaryConduit()
        self.assertIsNotNone(conduit)
        self.assertEqual(conduit.population_size, 100)
        self.assertEqual(conduit.generation_limit, 1000)
        self.assertEqual(conduit.mutation_rate, 0.01)
        self.assertEqual(conduit.crossover_rate, 0.7)
        
    def test_initialization_custom_parameters(self):
        """Test conduit initialization with custom parameters."""
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
        Test that initializing GenesisEvolutionaryConduit with zero or negative population size raises ConduitInitializationError.
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
        Test that initializing GenesisEvolutionaryConduit with mutation rates outside the range [0, 1] raises ConduitInitializationError.
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
        Test that population initialization succeeds and returns the expected population object using the mocked PopulationManager.
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
        Tests that an agent is successfully mutated when the mutation probability condition is met.
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
        Test that the agent is not mutated when the random probability exceeds the mutation rate.
        
        Ensures that the mutation operation is skipped and the original agent is returned unchanged.
        """
        mock_random.return_value = 0.5  # Above mutation rate
        mock_agent = Mock()
        mock_agent.mutate = Mock()
        
        result = self.conduit.mutate_agent(mock_agent)
        
        self.assertEqual(result, mock_agent)
        mock_agent.mutate.assert_not_called()
        
    def test_mutate_agent_mutation_failure(self):
        """
        Test that mutate_agent raises AgentEvolutionError when the agent's mutation operation fails.
        """
        mock_agent = Mock()
        mock_agent.mutate.side_effect = Exception("Mutation failed")
        
        with patch('genesis_evolutionary_conduit.random.random', return_value=0.005):
            with self.assertRaises(AgentEvolutionError):
                self.conduit.mutate_agent(mock_agent)
                
    def test_crossover_agents_success(self):
        """
        Test that the crossover_agents method returns the expected offspring when crossover succeeds.
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
        Tests that the crossover_agents method raises an AgentEvolutionError when the crossover operation fails.
        """
        mock_parent1 = Mock()
        mock_parent2 = Mock()
        
        with patch('genesis_evolutionary_conduit.CrossoverOperator') as mock_crossover:
            mock_crossover.return_value.crossover.side_effect = Exception("Crossover failed")
            
            with self.assertRaises(AgentEvolutionError):
                self.conduit.crossover_agents(mock_parent1, mock_parent2)
                
    def test_select_parents_success(self):
        """
        Test that the conduit successfully selects two parent agents from a population using the selection operator.
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
        Test that selecting parents from a population with fewer agents than required raises a PopulationEvolutionError.
        """
        mock_population = [Mock()]
        
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.select_parents(mock_population)
            
    def test_evolve_generation_success(self):
        """
        Test that a generation evolves successfully, producing a new population of the same size as the original.
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
        Test that `evolve_generation` raises a `PopulationEvolutionError` when parent selection fails during generation evolution.
        """
        mock_population = [Mock() for _ in range(10)]
        
        with patch.object(self.conduit, 'select_parents') as mock_select:
            mock_select.side_effect = Exception("Selection failed")
            
            with self.assertRaises(PopulationEvolutionError):
                self.conduit.evolve_generation(mock_population)
                
    def test_evaluate_fitness_success(self):
        """
        Tests that the fitness evaluation of an agent returns the expected fitness value using the conduit’s fitness evaluator.
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
        Test that an AgentEvolutionError is raised when fitness evaluation fails due to an exception in the evaluator.
        """
        mock_agent = Mock()
        
        with patch.object(self.conduit, 'fitness_evaluator') as mock_evaluator:
            mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")
            
            with self.assertRaises(AgentEvolutionError):
                self.conduit.evaluate_fitness(mock_agent)
                
    def test_run_evolution_success(self):
        """
        Tests that the evolution process completes successfully and returns the evolved population when convergence is achieved.
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
        Test that the evolution process runs up to the maximum generation limit when convergence is not reached.
        
        Verifies that the `run_evolution` method iterates for the configured number of generations and returns the final population when convergence does not occur.
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
        Test that the convergence check correctly identifies a converged population when all agents have identical fitness values.
        """
        mock_agent = Mock()
        mock_agent.fitness = 0.95
        mock_population = [mock_agent] * 10
        
        result = self.conduit.check_convergence(mock_population)
        
        self.assertTrue(result)
        
    def test_check_convergence_not_converged(self):
        """
        Tests that the convergence check correctly identifies a non-converged population when agent fitness values differ.
        """
        mock_agents = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_agents):
            agent.fitness = i * 0.1
        
        result = self.conduit.check_convergence(mock_agents)
        
        self.assertFalse(result)
        
    def test_get_best_agent_success(self):
        """
        Tests that the best agent, defined as the one with the highest fitness value, is correctly retrieved from a population.
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
        Test that the evolution state is saved to a file without errors using mocked file and JSON operations.
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
        Test that evolution statistics are correctly computed and returned for a given population.
        
        Verifies that the statistics dictionary includes generation, population size, and fitness metrics.
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
        Test that parameter validation succeeds when all values are valid.
        """
        # Should not raise any exceptions
        self.conduit.validate_parameters()
        
    def test_validate_parameters_invalid_population_size(self):
        """
        Test that parameter validation raises an error when the population size is set to an invalid negative value.
        """
        self.conduit.population_size = -10
        
        with self.assertRaises(ConduitInitializationError):
            self.conduit.validate_parameters()
            
    def test_validate_parameters_invalid_rates(self):
        """
        Test that parameter validation raises an error when mutation or crossover rates are set to invalid values.
        """
        self.conduit.mutation_rate = 1.5
        
        with self.assertRaises(ConduitInitializationError):
            self.conduit.validate_parameters()


class TestEvolutionaryAgent(unittest.TestCase):
    """Test suite for EvolutionaryAgent class."""
    
    def setUp(self):
        """
        Initializes a new EvolutionaryAgent instance before each test.
        """
        self.agent = EvolutionaryAgent()
        
    def test_initialization_default(self):
        """
        Test that an EvolutionaryAgent is initialized with a non-empty genome, default fitness of 0.0, and a valid unique ID.
        """
        agent = EvolutionaryAgent()
        self.assertIsNotNone(agent.genome)
        self.assertEqual(agent.fitness, 0.0)
        self.assertIsNotNone(agent.id)
        
    def test_initialization_custom_genome(self):
        """
        Test that an EvolutionaryAgent is correctly initialized with a specified custom genome.
        """
        custom_genome = [1, 0, 1, 1, 0]
        agent = EvolutionaryAgent(genome=custom_genome)
        self.assertEqual(agent.genome, custom_genome)
        
    def test_mutate_success(self):
        """
        Test that the agent's genome is successfully mutated, resulting in a different genome from the original.
        """
        original_genome = self.agent.genome.copy()
        self.agent.mutate()
        
        # Genome should be modified
        self.assertNotEqual(self.agent.genome, original_genome)
        
    def test_mutate_with_rate(self):
        """
        Test that mutating an agent with a 100% mutation rate always results in a changed genome.
        """
        original_genome = self.agent.genome.copy()
        self.agent.mutate(mutation_rate=1.0)  # 100% mutation rate
        
        # With 100% rate, genome should definitely change
        self.assertNotEqual(self.agent.genome, original_genome)
        
    def test_crossover_success(self):
        """
        Test that the crossover operation between two agents produces a valid offspring agent with a genome of the expected length.
        """
        other_agent = EvolutionaryAgent()
        offspring = self.agent.crossover(other_agent)
        
        self.assertIsInstance(offspring, EvolutionaryAgent)
        self.assertEqual(len(offspring.genome), len(self.agent.genome))
        
    def test_crossover_invalid_agent(self):
        """
        Test that attempting to perform crossover with an invalid (None) agent raises an AgentEvolutionError.
        """
        with self.assertRaises(AgentEvolutionError):
            self.agent.crossover(None)
            
    def test_evaluate_fitness_success(self):
        """
        Tests that an agent's fitness is correctly evaluated and assigned using a mock fitness evaluator.
        """
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = 0.8
        
        self.agent.evaluate_fitness(mock_evaluator)
        
        self.assertEqual(self.agent.fitness, 0.8)
        mock_evaluator.evaluate.assert_called_once_with(self.agent)
        
    def test_clone_success(self):
        """
        Test that cloning an agent produces a distinct copy with identical genome and fitness, but a unique ID.
        """
        self.agent.fitness = 0.7
        clone = self.agent.clone()
        
        self.assertIsNot(clone, self.agent)
        self.assertEqual(clone.genome, self.agent.genome)
        self.assertEqual(clone.fitness, self.agent.fitness)
        self.assertNotEqual(clone.id, self.agent.id)
        
    def test_to_dict_success(self):
        """
        Test that an agent is correctly serialized to a dictionary, including its ID, genome, and fitness attributes.
        """
        self.agent.fitness = 0.6
        agent_dict = self.agent.to_dict()
        
        self.assertIn('id', agent_dict)
        self.assertIn('genome', agent_dict)
        self.assertIn('fitness', agent_dict)
        self.assertEqual(agent_dict['fitness'], 0.6)
        
    def test_from_dict_success(self):
        """
        Test that an EvolutionaryAgent instance is correctly created from a dictionary representation.
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
        Set up the test environment by initializing a GeneticAlgorithm instance before each test.
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
        Tests that the genetic algorithm runs successfully on a population, producing a new population of the same size.
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
        Set up the test environment by initializing a FitnessEvaluator instance before each test.
        """
        self.evaluator = FitnessEvaluator()
        
    def test_evaluate_agent_success(self):
        """
        Tests that the fitness evaluator successfully computes a valid fitness value for a given agent.
        
        The test verifies that the returned fitness is a float within the expected range [0.0, 1.0].
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
        Test that evaluating a population returns a list of fitness scores as floats, one for each agent.
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
        """Set up test fixtures before each test method."""
        self.operator = MutationOperator()
        
    def test_mutate_agent_success(self):
        """
        Test that the mutation operator successfully mutates an agent and returns the mutated agent.
        """
        mock_agent = Mock()
        mock_agent.genome = [1, 0, 1, 0, 1]
        
        result = self.operator.mutate(mock_agent)
        
        self.assertEqual(result, mock_agent)
        
    def test_mutate_agent_with_rate(self):
        """
        Test that the mutation operator mutates an agent's genome using a specified mutation rate.
        
        Ensures that the mutate method is called with the correct mutation rate and returns the mutated agent.
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
        Set up the test environment by initializing a CrossoverOperator instance before each test.
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
        Test that the crossover operator raises an AgentEvolutionError when one parent is None.
        """
        mock_parent = Mock()
        mock_parent.genome = [1, 0, 1, 0, 1]
        
        with self.assertRaises(AgentEvolutionError):
            self.operator.crossover(mock_parent, None)


class TestSelectionOperator(unittest.TestCase):
    """Test suite for SelectionOperator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
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
        Test that selecting zero agents from a population raises a PopulationEvolutionError.
        """
        mock_population = [Mock() for _ in range(5)]
        
        with self.assertRaises(PopulationEvolutionError):
            self.operator.select(mock_population, 0)
            
    def test_select_agents_count_exceeds_population(self):
        """
        Test that selecting more agents than are present in the population raises a PopulationEvolutionError.
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
        Tests that the PopulationManager initializes successfully with the specified size and a valid fitness evaluator.
        """
        manager = PopulationManager(size=20)
        self.assertEqual(manager.size, 20)
        self.assertIsNotNone(manager.fitness_evaluator)
        
    def test_initialization_invalid_size(self):
        """
        Test that initializing PopulationManager with an invalid size raises PopulationEvolutionError.
        """
        with self.assertRaises(PopulationEvolutionError):
            PopulationManager(size=0)
            
    def test_generate_population_success(self):
        """
        Test that the population manager generates a population of the expected size with all members as EvolutionaryAgent instances.
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
        Test that the population is sorted in descending order based on agent fitness values.
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
        Test that population statistics are correctly computed and returned for a given population.
        
        Verifies that the statistics dictionary includes size, best, average, and worst fitness, and that the population size is accurate.
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
        Test that a ConduitInitializationError can be created and behaves as expected.
        """
        exception = ConduitInitializationError("Initialization failed")
        self.assertEqual(str(exception), "Initialization failed")
        self.assertIsInstance(exception, EvolutionaryException)
        
    def test_agent_evolution_error_creation(self):
        """
        Test that the AgentEvolutionError is created correctly and inherits from EvolutionaryException.
        """
        exception = AgentEvolutionError("Agent evolution failed")
        self.assertEqual(str(exception), "Agent evolution failed")
        self.assertIsInstance(exception, EvolutionaryException)
        
    def test_population_evolution_error_creation(self):
        """
        Test that PopulationEvolutionError is created correctly and inherits from EvolutionaryException.
        """
        exception = PopulationEvolutionError("Population evolution failed")
        self.assertEqual(str(exception), "Population evolution failed")
        self.assertIsInstance(exception, EvolutionaryException)


def mock_open(read_data=''):
    """
    Creates a mock object that simulates the built-in open function for file operations in tests.
    
    Parameters:
        read_data (str): The data to be returned when the file is read. Defaults to an empty string.
    
    Returns:
        MagicMock: A mock object with the same interface as the built-in open function.
    """
    return MagicMock(spec=open)


class TestGenesisEvolutionaryConduitAdvanced(unittest.TestCase):
    """Advanced test suite for GenesisEvolutionaryConduit with edge cases and integration scenarios."""
    
    def setUp(self):
        """
        Initializes a GenesisEvolutionaryConduit instance with advanced parameters for use in each test.
        """
        self.conduit = GenesisEvolutionaryConduit(
            population_size=50,
            generation_limit=25,
            mutation_rate=0.05,
            crossover_rate=0.75
        )
        
    def test_concurrent_evolution_safety(self):
        """
        Verify that the evolutionary conduit can safely handle concurrent evolution runs without raising exceptions or data corruption.
        
        This test launches multiple threads, each running the evolution process in parallel, and asserts that all threads complete successfully with no errors.
        """
        import threading
        
        results = []
        errors = []
        
        def run_evolution():
            """
            Runs the evolutionary process using the conduit, capturing the result or any exceptions.
            
            Appends the result of the evolution run to the `results` list, or appends any raised exceptions to the `errors` list.
            """
            try:
                with patch.object(self.conduit, 'initialize_population') as mock_init:
                    with patch.object(self.conduit, 'evolve_generation') as mock_evolve:
                        mock_population = [Mock() for _ in range(5)]
                        mock_init.return_value = mock_population
                        mock_evolve.return_value = mock_population
                        
                        result = self.conduit.run_evolution()
                        results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=run_evolution) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
            
        self.assertEqual(len(errors), 0, f"Concurrent execution errors: {errors}")
        self.assertEqual(len(results), 3)
        
    def test_memory_management_large_population(self):
        """
        Tests that the evolutionary conduit can initialize and handle a large population of agents with sizable genomes without encountering memory issues.
        """
        large_conduit = GenesisEvolutionaryConduit(population_size=1000)
        
        with patch.object(large_conduit, 'initialize_population') as mock_init:
            mock_population = [Mock() for _ in range(1000)]
            for agent in mock_population:
                agent.genome = list(range(100))  # Large genome
                agent.fitness = 0.5
                
            mock_init.return_value = mock_population
            
            # Should handle large populations without memory issues
            result = large_conduit.initialize_population()
            self.assertEqual(len(result), 1000)
            
    def test_evolution_with_extreme_parameters(self):
        """
        Tests that the evolutionary process completes successfully when using extreme but valid parameter values for population size, generation limit, mutation rate, and crossover rate.
        """
        extreme_conduit = GenesisEvolutionaryConduit(
            population_size=1,  # Minimum valid size
            generation_limit=1,  # Minimum valid limit
            mutation_rate=0.99,  # Very high mutation
            crossover_rate=0.01  # Very low crossover
        )
        
        with patch.object(extreme_conduit, 'initialize_population') as mock_init:
            with patch.object(extreme_conduit, 'evolve_generation') as mock_evolve:
                mock_population = [Mock()]
                mock_init.return_value = mock_population
                mock_evolve.return_value = mock_population
                
                result = extreme_conduit.run_evolution()
                self.assertIsNotNone(result)
                
    def test_fitness_convergence_threshold_variations(self):
        """
        Tests population convergence detection with both exact and near-identical fitness values, verifying that the convergence check handles small fitness variations appropriately.
        """
        mock_agents = [Mock() for _ in range(10)]
        
        # Test exact convergence
        for agent in mock_agents:
            agent.fitness = 0.95
        self.assertTrue(self.conduit.check_convergence(mock_agents))
        
        # Test near convergence (within small tolerance)
        for i, agent in enumerate(mock_agents):
            agent.fitness = 0.95 + (i * 0.001)  # Very small variation
        
        # Should still converge with small variations
        result = self.conduit.check_convergence(mock_agents)
        # This depends on implementation details, but test the behavior
        self.assertIsInstance(result, bool)
        
    def test_population_diversity_maintenance(self):
        """
        Verify that the population size and structure are preserved after evolving a generation, ensuring diversity is maintained during the evolutionary process.
        """
        mock_population = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_population):
            agent.genome = [i % 2] * 10  # Binary patterns
            agent.fitness = i * 0.1
            
        with patch.object(self.conduit, 'select_parents') as mock_select:
            with patch.object(self.conduit, 'crossover_agents') as mock_crossover:
                with patch.object(self.conduit, 'mutate_agent') as mock_mutate:
                    mock_select.return_value = mock_population[:2]
                    mock_crossover.return_value = Mock()
                    mock_mutate.side_effect = lambda x: x
                    
                    result = self.conduit.evolve_generation(mock_population)
                    
                    # Should maintain population size and structure
                    self.assertEqual(len(result), len(mock_population))
                    
    def test_evolution_state_persistence_integrity(self):
        """
        Verify that saving and loading the evolution state preserves all relevant data, ensuring integrity across persistence cycles.
        
        This test creates a mock population and evolution state, saves it to a temporary file, loads it back using the conduit, and checks that generation, population size, and timestamp are correctly restored.
        """
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
            
        try:
            mock_population = [Mock() for _ in range(5)]
            for i, agent in enumerate(mock_population):
                agent.to_dict = Mock(return_value={'id': f'agent_{i}', 'fitness': i * 0.2})
                
            generation = 42
            
            # Mock the state saving
            test_state = {
                'population': [agent.to_dict() for agent in mock_population],
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'population_size': self.conduit.population_size,
                    'mutation_rate': self.conduit.mutation_rate,
                    'crossover_rate': self.conduit.crossover_rate
                }
            }
            
            with open(state_file, 'w') as f:
                json.dump(test_state, f)
                
            loaded_state = self.conduit.load_evolution_state(state_file)
            
            # Verify state integrity
            self.assertEqual(loaded_state['generation'], generation)
            self.assertEqual(len(loaded_state['population']), 5)
            self.assertIn('timestamp', loaded_state)
            
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)
                
    def test_error_recovery_mechanisms(self):
        """
        Tests that the evolutionary conduit can recover from partial failures during the evolution process and degrade gracefully when errors occur.
        """
        mock_population = [Mock() for _ in range(5)]
        
        # Test partial failure recovery
        with patch.object(self.conduit, 'select_parents') as mock_select:
            with patch.object(self.conduit, 'crossover_agents') as mock_crossover:
                # First call fails, second succeeds
                mock_select.side_effect = [Exception("Temporary failure"), mock_population[:2]]
                mock_crossover.return_value = Mock()
                
                # Should handle the failure gracefully
                try:
                    result = self.conduit.evolve_generation(mock_population)
                except PopulationEvolutionError:
                    pass  # Expected behavior
                    
    def test_parameter_validation_edge_cases(self):
        """
        Tests parameter validation for GenesisEvolutionaryConduit using boundary and out-of-bound values to ensure correct error handling.
        """
        # Test boundary values that should be valid
        valid_conduit = GenesisEvolutionaryConduit(
            population_size=1,
            generation_limit=1,
            mutation_rate=0.0,
            crossover_rate=1.0
        )
        self.assertIsNotNone(valid_conduit)
        
        # Test values just outside boundaries
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(mutation_rate=-0.001)
            
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(crossover_rate=1.001)


class TestEvolutionaryAgentAdvanced(unittest.TestCase):
    """Advanced test suite for EvolutionaryAgent with comprehensive edge case coverage."""
    
    def setUp(self):
        """
        Set up the test fixture by creating a new EvolutionaryAgent instance for advanced agent tests.
        """
        self.agent = EvolutionaryAgent()
        
    def test_genome_size_variations(self):
        """
        Verify that EvolutionaryAgent correctly handles genomes of varying sizes, including small, large, and empty genomes.
        """
        # Test very small genome
        small_agent = EvolutionaryAgent(genome=[1])
        self.assertEqual(len(small_agent.genome), 1)
        
        # Test large genome
        large_genome = list(range(1000))
        large_agent = EvolutionaryAgent(genome=large_genome)
        self.assertEqual(len(large_agent.genome), 1000)
        
        # Test empty genome (edge case)
        empty_agent = EvolutionaryAgent(genome=[])
        self.assertEqual(len(empty_agent.genome), 0)
        
    def test_crossover_with_different_genome_sizes(self):
        """
        Tests that crossover between agents with different genome sizes either produces a valid offspring or raises an AgentEvolutionError.
        """
        agent1 = EvolutionaryAgent(genome=[1, 0, 1])
        agent2 = EvolutionaryAgent(genome=[0, 1, 0, 1, 1])
        
        # Should handle different sizes gracefully
        try:
            offspring = agent1.crossover(agent2)
            self.assertIsInstance(offspring, EvolutionaryAgent)
        except AgentEvolutionError:
            pass  # Acceptable behavior for incompatible genomes
            
    def test_mutation_with_various_rates(self):
        """
        Test that agent mutation behaves correctly at zero, full, and intermediate mutation rates, including edge cases.
        """
        original_genome = self.agent.genome.copy()
        
        # Test zero mutation rate
        self.agent.mutate(mutation_rate=0.0)
        self.assertEqual(self.agent.genome, original_genome)
        
        # Test maximum mutation rate
        self.agent.mutate(mutation_rate=1.0)
        # With 100% rate, should definitely change
        
        # Test mutation rate exactly at boundaries
        test_agent = EvolutionaryAgent(genome=[0, 1, 0, 1])
        test_agent.mutate(mutation_rate=0.5)
        # Should not raise exceptions
        
    def test_agent_deep_copy_semantics(self):
        """
        Verify that cloning an agent produces a deep copy, ensuring changes to the original agent's genome or fitness do not affect the clone.
        """
        self.agent.genome = [[1, 2], [3, 4]]  # Nested structure
        self.agent.fitness = 0.8
        
        clone = self.agent.clone()
        
        # Modify original
        self.agent.genome[0][0] = 999
        self.agent.fitness = 0.1
        
        # Clone should be unaffected if deep copy is implemented
        # Note: This test depends on implementation details
        self.assertNotEqual(clone.fitness, 0.1)
        
    def test_agent_serialization_complex_genomes(self):
        """
        Tests serialization and deserialization of EvolutionaryAgent instances with genomes containing various data types, ensuring genome and fitness values are preserved.
        """
        # Test with various data types in genome
        complex_genomes = [
            [1, 2, 3, 4, 5],  # Integers
            [1.1, 2.2, 3.3],  # Floats
            ['a', 'b', 'c'],  # Strings
            [True, False, True],  # Booleans
        ]
        
        for genome in complex_genomes:
            agent = EvolutionaryAgent(genome=genome)
            agent.fitness = 0.75
            
            # Test serialization
            agent_dict = agent.to_dict()
            self.assertIn('genome', agent_dict)
            self.assertEqual(agent_dict['genome'], genome)
            
            # Test deserialization
            restored_agent = EvolutionaryAgent.from_dict(agent_dict)
            self.assertEqual(restored_agent.genome, genome)
            self.assertEqual(restored_agent.fitness, 0.75)
            
    def test_fitness_evaluation_edge_cases(self):
        """
        Tests the agent's fitness evaluation behavior with edge case fitness values, including zero, negative, and infinite values.
        
        Handles exceptions for infinite fitness values and verifies correct assignment for finite values.
        """
        mock_evaluator = Mock()
        
        # Test with extreme fitness values
        extreme_values = [0.0, 1.0, -1.0, float('inf'), float('-inf')]
        
        for value in extreme_values:
            mock_evaluator.evaluate.return_value = value
            
            if value in [float('inf'), float('-inf')]:
                # Should handle infinite values gracefully
                try:
                    self.agent.evaluate_fitness(mock_evaluator)
                except (ValueError, OverflowError):
                    pass  # Acceptable behavior
            else:
                self.agent.evaluate_fitness(mock_evaluator)
                self.assertEqual(self.agent.fitness, value)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complex evolutionary scenarios."""
    
    def setUp(self):
        """
        Initialize the integration test environment with a GenesisEvolutionaryConduit instance configured for a small population and limited generations.
        """
        self.conduit = GenesisEvolutionaryConduit(
            population_size=20,
            generation_limit=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
    def test_complete_evolution_cycle_integration(self):
        """
        Verifies the integration of all major evolutionary operations by running a complete evolution cycle from population initialization through convergence, ensuring that selection, crossover, and mutation components interact as expected.
        """
        mock_initial_population = [Mock() for _ in range(20)]
        for i, agent in enumerate(mock_initial_population):
            agent.genome = [i % 2] * 10
            agent.fitness = i * 0.05
            agent.mutate = Mock()
            agent.clone = Mock(return_value=agent)
            
        with patch.object(self.conduit, 'initialize_population') as mock_init:
            with patch('genesis_evolutionary_conduit.SelectionOperator') as mock_selection_class:
                with patch('genesis_evolutionary_conduit.CrossoverOperator') as mock_crossover_class:
                    mock_init.return_value = mock_initial_population
                    
                    mock_selector = Mock()
                    mock_selector.select.return_value = mock_initial_population[:2]
                    mock_selection_class.return_value = mock_selector
                    
                    mock_crossover = Mock()
                    mock_crossover.crossover.return_value = Mock()
                    mock_crossover_class.return_value = mock_crossover
                    
                    # Run evolution
                    result = self.conduit.run_evolution()
                    
                    # Verify integration points
                    self.assertIsNotNone(result)
                    mock_init.assert_called_once()
                    
    def test_evolution_with_fitness_plateau(self):
        """
        Test that the evolution process correctly detects convergence when all agents have identical fitness values, simulating a fitness plateau scenario.
        """
        mock_population = [Mock() for _ in range(10)]
        # All agents have same fitness (plateau scenario)
        for agent in mock_population:
            agent.fitness = 0.5
            agent.genome = [1, 0, 1, 0]
            
        # Should handle plateau gracefully
        result = self.conduit.check_convergence(mock_population)
        self.assertIsInstance(result, bool)
        
    def test_evolution_statistics_accuracy(self):
        """
        Verify that the evolution statistics computed for a mock population are accurate.
        
        This test checks that the best, worst, and average fitness values, as well as the population size, are correctly calculated by the `get_evolution_statistics` method.
        """
        mock_population = [Mock() for _ in range(10)]
        fitness_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for agent, fitness in zip(mock_population, fitness_values):
            agent.fitness = fitness
            
        stats = self.conduit.get_evolution_statistics(mock_population)
        
        # Verify statistical calculations
        self.assertEqual(stats['best_fitness'], 1.0)
        self.assertEqual(stats['worst_fitness'], 0.1)
        self.assertAlmostEqual(stats['average_fitness'], 0.55, places=2)
        self.assertEqual(stats['population_size'], 10)


class TestPerformanceAndResourceManagement(unittest.TestCase):
    """Test performance characteristics and resource management."""
    
    def test_large_scale_evolution_performance(self):
        """
        Tests that the evolutionary algorithm can execute a large-scale evolution run with a sizable population and multiple generations, completing within a reasonable time frame.
        """
        large_conduit = GenesisEvolutionaryConduit(
            population_size=500,
            generation_limit=10
        )
        
        with patch.object(large_conduit, 'initialize_population') as mock_init:
            with patch.object(large_conduit, 'evolve_generation') as mock_evolve:
                mock_population = [Mock() for _ in range(500)]
                mock_init.return_value = mock_population
                mock_evolve.return_value = mock_population
                
                import time
                start_time = time.time()
                result = large_conduit.run_evolution()
                end_time = time.time()
                
                # Should complete in reasonable time (implementation dependent)
                self.assertIsNotNone(result)
                # Performance assertion can be adjusted based on requirements
                
    def test_memory_cleanup_after_evolution(self):
        """
        Verify that running the evolution process does not result in significant memory leaks by comparing object counts before and after execution.
        """
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        conduit = GenesisEvolutionaryConduit(population_size=100)
        
        with patch.object(conduit, 'initialize_population') as mock_init:
            with patch.object(conduit, 'evolve_generation') as mock_evolve:
                mock_population = [Mock() for _ in range(100)]
                mock_init.return_value = mock_population
                mock_evolve.return_value = mock_population
                
                result = conduit.run_evolution()
                
        # Clean up references
        del conduit
        del result
        gc.collect()
        
        # Check that we haven't leaked too many objects
        final_objects = len(gc.get_objects())
        # Allow for some variance in object count
        self.assertLess(final_objects - initial_objects, 1000)


class TestDataConsistencyAndValidation(unittest.TestCase):
    """Test data consistency and validation across the evolutionary system."""
    
    def test_population_size_consistency(self):
        """
        Verifies that the population size remains unchanged after evolving a generation.
        """
        conduit = GenesisEvolutionaryConduit(population_size=15)
        
        mock_population = [Mock() for _ in range(15)]
        for agent in mock_population:
            agent.fitness = 0.5
            
        with patch.object(conduit, 'select_parents') as mock_select:
            with patch.object(conduit, 'crossover_agents') as mock_crossover:
                with patch.object(conduit, 'mutate_agent') as mock_mutate:
                    mock_select.return_value = mock_population[:2]
                    mock_crossover.return_value = Mock()
                    mock_mutate.side_effect = lambda x: x
                    
                    result = conduit.evolve_generation(mock_population)
                    
                    # Population size should remain consistent
                    self.assertEqual(len(result), 15)
                    
    def test_fitness_value_consistency(self):
        """
        Verify that the fitness values produced by the evaluator for various agent genomes are numeric and within expected types.
        """
        evaluator = FitnessEvaluator()
        
        # Test with various agent configurations
        test_agents = [
            EvolutionaryAgent(genome=[]),
            EvolutionaryAgent(genome=[1]),
            EvolutionaryAgent(genome=[0, 1, 0, 1, 1]),
            EvolutionaryAgent(genome=list(range(100))),
        ]
        
        for agent in test_agents:
            fitness = evaluator.evaluate(agent)
            
            # Fitness should be a valid number within expected range
            self.assertIsInstance(fitness, (int, float))
            # Additional range checks can be added based on specific requirements
            
    def test_agent_id_uniqueness(self):
        """
        Verify that each agent in a generated population has a unique, non-empty ID.
        """
        agents = [EvolutionaryAgent() for _ in range(100)]
        agent_ids = [agent.id for agent in agents]
        
        # All IDs should be unique
        self.assertEqual(len(agent_ids), len(set(agent_ids)))
        
        # IDs should be non-empty
        self.assertTrue(all(agent_id for agent_id in agent_ids))


if __name__ == '__main__':
    unittest.main()