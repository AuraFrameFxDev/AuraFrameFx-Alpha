import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
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
        """Set up test fixtures before each test method."""
        self.conduit = GenesisEvolutionaryConduit()
        self.mock_population_size = 100
        self.mock_generation_limit = 50
        self.mock_mutation_rate = 0.1
        self.mock_crossover_rate = 0.8
        
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_default_parameters(self):
        """Test conduit initialization with default parameters."""
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
        """Test initialization with invalid population size."""
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(population_size=0)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(population_size=-10)
            
    def test_initialization_invalid_generation_limit(self):
        """Test initialization with invalid generation limit."""
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(generation_limit=0)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(generation_limit=-5)
            
    def test_initialization_invalid_mutation_rate(self):
        """Test initialization with invalid mutation rate."""
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(mutation_rate=-0.1)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(mutation_rate=1.1)
            
    def test_initialization_invalid_crossover_rate(self):
        """Test initialization with invalid crossover rate."""
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(crossover_rate=-0.1)
        with self.assertRaises(ConduitInitializationError):
            GenesisEvolutionaryConduit(crossover_rate=1.1)
            
    @patch('genesis_evolutionary_conduit.PopulationManager')
    def test_initialize_population_success(self, mock_population_manager):
        """Test successful population initialization."""
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
        """Test population initialization failure."""
        mock_population_manager.side_effect = Exception("Population initialization failed")
        
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.initialize_population()
            
    @patch('genesis_evolutionary_conduit.random.random')
    def test_mutate_agent_success(self, mock_random):
        """Test successful agent mutation."""
        mock_random.return_value = 0.005  # Below mutation rate
        mock_agent = Mock()
        mock_agent.mutate = Mock()
        
        result = self.conduit.mutate_agent(mock_agent)
        
        self.assertEqual(result, mock_agent)
        mock_agent.mutate.assert_called_once()
        
    @patch('genesis_evolutionary_conduit.random.random')
    def test_mutate_agent_no_mutation(self, mock_random):
        """Test agent mutation when probability is above threshold."""
        mock_random.return_value = 0.5  # Above mutation rate
        mock_agent = Mock()
        mock_agent.mutate = Mock()
        
        result = self.conduit.mutate_agent(mock_agent)
        
        self.assertEqual(result, mock_agent)
        mock_agent.mutate.assert_not_called()
        
    def test_mutate_agent_mutation_failure(self):
        """Test agent mutation when mutation fails."""
        mock_agent = Mock()
        mock_agent.mutate.side_effect = Exception("Mutation failed")
        
        with patch('genesis_evolutionary_conduit.random.random', return_value=0.005):
            with self.assertRaises(AgentEvolutionError):
                self.conduit.mutate_agent(mock_agent)
                
    def test_crossover_agents_success(self):
        """Test successful agent crossover."""
        mock_parent1 = Mock()
        mock_parent2 = Mock()
        mock_offspring = Mock()
        
        with patch('genesis_evolutionary_conduit.CrossoverOperator') as mock_crossover:
            mock_crossover.return_value.crossover.return_value = mock_offspring
            
            result = self.conduit.crossover_agents(mock_parent1, mock_parent2)
            
            self.assertEqual(result, mock_offspring)
            mock_crossover.assert_called_once()
            
    def test_crossover_agents_failure(self):
        """Test agent crossover failure."""
        mock_parent1 = Mock()
        mock_parent2 = Mock()
        
        with patch('genesis_evolutionary_conduit.CrossoverOperator') as mock_crossover:
            mock_crossover.return_value.crossover.side_effect = Exception("Crossover failed")
            
            with self.assertRaises(AgentEvolutionError):
                self.conduit.crossover_agents(mock_parent1, mock_parent2)
                
    def test_select_parents_success(self):
        """Test successful parent selection."""
        mock_population = [Mock() for _ in range(10)]
        mock_selector = Mock()
        mock_selector.select.return_value = [mock_population[0], mock_population[1]]
        
        with patch('genesis_evolutionary_conduit.SelectionOperator', return_value=mock_selector):
            result = self.conduit.select_parents(mock_population)
            
            self.assertEqual(len(result), 2)
            mock_selector.select.assert_called_once_with(mock_population, 2)
            
    def test_select_parents_empty_population(self):
        """Test parent selection with empty population."""
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.select_parents([])
            
    def test_select_parents_insufficient_population(self):
        """Test parent selection with insufficient population size."""
        mock_population = [Mock()]
        
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.select_parents(mock_population)
            
    def test_evolve_generation_success(self):
        """Test successful generation evolution."""
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
        """Test generation evolution failure."""
        mock_population = [Mock() for _ in range(10)]
        
        with patch.object(self.conduit, 'select_parents') as mock_select:
            mock_select.side_effect = Exception("Selection failed")
            
            with self.assertRaises(PopulationEvolutionError):
                self.conduit.evolve_generation(mock_population)
                
    def test_evaluate_fitness_success(self):
        """Test successful fitness evaluation."""
        mock_agent = Mock()
        mock_agent.fitness = 0.8
        
        with patch.object(self.conduit, 'fitness_evaluator') as mock_evaluator:
            mock_evaluator.evaluate.return_value = 0.8
            
            result = self.conduit.evaluate_fitness(mock_agent)
            
            self.assertEqual(result, 0.8)
            mock_evaluator.evaluate.assert_called_once_with(mock_agent)
            
    def test_evaluate_fitness_failure(self):
        """Test fitness evaluation failure."""
        mock_agent = Mock()
        
        with patch.object(self.conduit, 'fitness_evaluator') as mock_evaluator:
            mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")
            
            with self.assertRaises(AgentEvolutionError):
                self.conduit.evaluate_fitness(mock_agent)
                
    def test_run_evolution_success(self):
        """Test successful evolution run."""
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
        """Test evolution run reaching maximum generations."""
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
        """Test convergence check when population has converged."""
        mock_agent = Mock()
        mock_agent.fitness = 0.95
        mock_population = [mock_agent] * 10
        
        result = self.conduit.check_convergence(mock_population)
        
        self.assertTrue(result)
        
    def test_check_convergence_not_converged(self):
        """Test convergence check when population has not converged."""
        mock_agents = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_agents):
            agent.fitness = i * 0.1
        
        result = self.conduit.check_convergence(mock_agents)
        
        self.assertFalse(result)
        
    def test_get_best_agent_success(self):
        """Test getting best agent from population."""
        mock_agents = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_agents):
            agent.fitness = i * 0.1
        
        result = self.conduit.get_best_agent(mock_agents)
        
        self.assertEqual(result, mock_agents[-1])  # Highest fitness
        
    def test_get_best_agent_empty_population(self):
        """Test getting best agent from empty population."""
        with self.assertRaises(PopulationEvolutionError):
            self.conduit.get_best_agent([])
            
    def test_save_evolution_state_success(self):
        """Test successful evolution state saving."""
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
        """Test successful evolution state loading."""
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
        """Test loading evolution state when file doesn't exist."""
        non_existent_file = '/non/existent/file.json'
        
        with self.assertRaises(FileNotFoundError):
            self.conduit.load_evolution_state(non_existent_file)
            
    def test_reset_evolution_state(self):
        """Test resetting evolution state."""
        self.conduit.current_generation = 50
        self.conduit.best_fitness = 0.8
        
        self.conduit.reset_evolution_state()
        
        self.assertEqual(self.conduit.current_generation, 0)
        self.assertEqual(self.conduit.best_fitness, 0.0)
        
    def test_get_evolution_statistics(self):
        """Test getting evolution statistics."""
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
        """Test parameter validation with valid values."""
        # Should not raise any exceptions
        self.conduit.validate_parameters()
        
    def test_validate_parameters_invalid_population_size(self):
        """Test parameter validation with invalid population size."""
        self.conduit.population_size = -10
        
        with self.assertRaises(ConduitInitializationError):
            self.conduit.validate_parameters()
            
    def test_validate_parameters_invalid_rates(self):
        """Test parameter validation with invalid rates."""
        self.conduit.mutation_rate = 1.5
        
        with self.assertRaises(ConduitInitializationError):
            self.conduit.validate_parameters()


class TestEvolutionaryAgent(unittest.TestCase):
    """Test suite for EvolutionaryAgent class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent = EvolutionaryAgent()
        
    def test_initialization_default(self):
        """Test agent initialization with default parameters."""
        agent = EvolutionaryAgent()
        self.assertIsNotNone(agent.genome)
        self.assertEqual(agent.fitness, 0.0)
        self.assertIsNotNone(agent.id)
        
    def test_initialization_custom_genome(self):
        """Test agent initialization with custom genome."""
        custom_genome = [1, 0, 1, 1, 0]
        agent = EvolutionaryAgent(genome=custom_genome)
        self.assertEqual(agent.genome, custom_genome)
        
    def test_mutate_success(self):
        """Test successful agent mutation."""
        original_genome = self.agent.genome.copy()
        self.agent.mutate()
        
        # Genome should be modified
        self.assertNotEqual(self.agent.genome, original_genome)
        
    def test_mutate_with_rate(self):
        """Test agent mutation with specific rate."""
        original_genome = self.agent.genome.copy()
        self.agent.mutate(mutation_rate=1.0)  # 100% mutation rate
        
        # With 100% rate, genome should definitely change
        self.assertNotEqual(self.agent.genome, original_genome)
        
    def test_crossover_success(self):
        """Test successful agent crossover."""
        other_agent = EvolutionaryAgent()
        offspring = self.agent.crossover(other_agent)
        
        self.assertIsInstance(offspring, EvolutionaryAgent)
        self.assertEqual(len(offspring.genome), len(self.agent.genome))
        
    def test_crossover_invalid_agent(self):
        """Test crossover with invalid agent."""
        with self.assertRaises(AgentEvolutionError):
            self.agent.crossover(None)
            
    def test_evaluate_fitness_success(self):
        """Test successful fitness evaluation."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = 0.8
        
        self.agent.evaluate_fitness(mock_evaluator)
        
        self.assertEqual(self.agent.fitness, 0.8)
        mock_evaluator.evaluate.assert_called_once_with(self.agent)
        
    def test_clone_success(self):
        """Test successful agent cloning."""
        self.agent.fitness = 0.7
        clone = self.agent.clone()
        
        self.assertIsNot(clone, self.agent)
        self.assertEqual(clone.genome, self.agent.genome)
        self.assertEqual(clone.fitness, self.agent.fitness)
        self.assertNotEqual(clone.id, self.agent.id)
        
    def test_to_dict_success(self):
        """Test converting agent to dictionary."""
        self.agent.fitness = 0.6
        agent_dict = self.agent.to_dict()
        
        self.assertIn('id', agent_dict)
        self.assertIn('genome', agent_dict)
        self.assertIn('fitness', agent_dict)
        self.assertEqual(agent_dict['fitness'], 0.6)
        
    def test_from_dict_success(self):
        """Test creating agent from dictionary."""
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
        """Test creating agent from dictionary with missing fields."""
        agent_dict = {'genome': [1, 0, 1, 0]}
        
        with self.assertRaises(AgentEvolutionError):
            EvolutionaryAgent.from_dict(agent_dict)


class TestGeneticAlgorithm(unittest.TestCase):
    """Test suite for GeneticAlgorithm class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.algorithm = GeneticAlgorithm()
        
    def test_initialization_default(self):
        """Test genetic algorithm initialization with default parameters."""
        algorithm = GeneticAlgorithm()
        self.assertIsNotNone(algorithm.selection_operator)
        self.assertIsNotNone(algorithm.crossover_operator)
        self.assertIsNotNone(algorithm.mutation_operator)
        
    def test_run_algorithm_success(self):
        """Test successful algorithm run."""
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
        """Test algorithm run with empty population."""
        with self.assertRaises(PopulationEvolutionError):
            self.algorithm.run_algorithm([])


class TestFitnessEvaluator(unittest.TestCase):
    """Test suite for FitnessEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.evaluator = FitnessEvaluator()
        
    def test_evaluate_agent_success(self):
        """Test successful agent evaluation."""
        mock_agent = Mock()
        mock_agent.genome = [1, 0, 1, 1, 0]
        
        fitness = self.evaluator.evaluate(mock_agent)
        
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
        
    def test_evaluate_agent_none(self):
        """Test evaluation with None agent."""
        with self.assertRaises(AgentEvolutionError):
            self.evaluator.evaluate(None)
            
    def test_evaluate_population_success(self):
        """Test successful population evaluation."""
        mock_population = [Mock() for _ in range(5)]
        for agent in mock_population:
            agent.genome = [1, 0, 1, 0, 1]
            
        fitness_scores = self.evaluator.evaluate_population(mock_population)
        
        self.assertEqual(len(fitness_scores), 5)
        self.assertTrue(all(isinstance(score, float) for score in fitness_scores))
        
    def test_evaluate_population_empty(self):
        """Test evaluation with empty population."""
        fitness_scores = self.evaluator.evaluate_population([])
        self.assertEqual(len(fitness_scores), 0)


class TestMutationOperator(unittest.TestCase):
    """Test suite for MutationOperator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.operator = MutationOperator()
        
    def test_mutate_agent_success(self):
        """Test successful agent mutation."""
        mock_agent = Mock()
        mock_agent.genome = [1, 0, 1, 0, 1]
        
        result = self.operator.mutate(mock_agent)
        
        self.assertEqual(result, mock_agent)
        
    def test_mutate_agent_with_rate(self):
        """Test agent mutation with specific rate."""
        mock_agent = Mock()
        mock_agent.genome = [1, 0, 1, 0, 1]
        
        result = self.operator.mutate(mock_agent, mutation_rate=0.5)
        
        self.assertEqual(result, mock_agent)
        
    def test_mutate_agent_none(self):
        """Test mutation with None agent."""
        with self.assertRaises(AgentEvolutionError):
            self.operator.mutate(None)


class TestCrossoverOperator(unittest.TestCase):
    """Test suite for CrossoverOperator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.operator = CrossoverOperator()
        
    def test_crossover_agents_success(self):
        """Test successful agent crossover."""
        mock_parent1 = Mock()
        mock_parent2 = Mock()
        mock_parent1.genome = [1, 0, 1, 0, 1]
        mock_parent2.genome = [0, 1, 0, 1, 0]
        
        offspring = self.operator.crossover(mock_parent1, mock_parent2)
        
        self.assertIsNotNone(offspring)
        
    def test_crossover_agents_none_parents(self):
        """Test crossover with None parents."""
        with self.assertRaises(AgentEvolutionError):
            self.operator.crossover(None, None)
            
    def test_crossover_agents_single_none_parent(self):
        """Test crossover with one None parent."""
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
        """Test successful agent selection."""
        mock_population = [Mock() for _ in range(10)]
        for i, agent in enumerate(mock_population):
            agent.fitness = i * 0.1
            
        selected = self.operator.select(mock_population, 2)
        
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(agent in mock_population for agent in selected))
        
    def test_select_agents_empty_population(self):
        """Test selection with empty population."""
        with self.assertRaises(PopulationEvolutionError):
            self.operator.select([], 2)
            
    def test_select_agents_invalid_count(self):
        """Test selection with invalid count."""
        mock_population = [Mock() for _ in range(5)]
        
        with self.assertRaises(PopulationEvolutionError):
            self.operator.select(mock_population, 0)
            
    def test_select_agents_count_exceeds_population(self):
        """Test selection when count exceeds population size."""
        mock_population = [Mock() for _ in range(5)]
        
        with self.assertRaises(PopulationEvolutionError):
            self.operator.select(mock_population, 10)


class TestPopulationManager(unittest.TestCase):
    """Test suite for PopulationManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.manager = PopulationManager(size=10)
        
    def test_initialization_success(self):
        """Test successful population manager initialization."""
        manager = PopulationManager(size=20)
        self.assertEqual(manager.size, 20)
        self.assertIsNotNone(manager.fitness_evaluator)
        
    def test_initialization_invalid_size(self):
        """Test initialization with invalid size."""
        with self.assertRaises(PopulationEvolutionError):
            PopulationManager(size=0)
            
    def test_generate_population_success(self):
        """Test successful population generation."""
        population = self.manager.generate_population()
        
        self.assertEqual(len(population), 10)
        self.assertTrue(all(isinstance(agent, EvolutionaryAgent) for agent in population))
        
    def test_evaluate_population_success(self):
        """Test successful population evaluation."""
        mock_population = [Mock() for _ in range(5)]
        
        with patch.object(self.manager.fitness_evaluator, 'evaluate', return_value=0.5):
            self.manager.evaluate_population(mock_population)
            
            for agent in mock_population:
                self.assertEqual(agent.fitness, 0.5)
                
    def test_sort_population_by_fitness(self):
        """Test sorting population by fitness."""
        mock_population = [Mock() for _ in range(5)]
        for i, agent in enumerate(mock_population):
            agent.fitness = (4 - i) * 0.2  # Reverse order
            
        sorted_population = self.manager.sort_population_by_fitness(mock_population)
        
        # Should be sorted in descending order
        for i in range(len(sorted_population) - 1):
            self.assertGreaterEqual(sorted_population[i].fitness, sorted_population[i + 1].fitness)
            
    def test_get_population_statistics(self):
        """Test getting population statistics."""
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
        """Test creating ConduitInitializationError."""
        exception = ConduitInitializationError("Initialization failed")
        self.assertEqual(str(exception), "Initialization failed")
        self.assertIsInstance(exception, EvolutionaryException)
        
    def test_agent_evolution_error_creation(self):
        """Test creating AgentEvolutionError."""
        exception = AgentEvolutionError("Agent evolution failed")
        self.assertEqual(str(exception), "Agent evolution failed")
        self.assertIsInstance(exception, EvolutionaryException)
        
    def test_population_evolution_error_creation(self):
        """Test creating PopulationEvolutionError."""
        exception = PopulationEvolutionError("Population evolution failed")
        self.assertEqual(str(exception), "Population evolution failed")
        self.assertIsInstance(exception, EvolutionaryException)


if __name__ == '__main__':
    unittest.main()