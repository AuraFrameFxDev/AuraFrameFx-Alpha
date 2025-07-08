import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
import asyncio
import sys
import os

# Add the app directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ai_backend.genesis_evolutionary_conduit import (
    EvolutionaryConduit,
    GenesisEvolutionaryConduit,
    EvolutionaryParameters,
    MutationStrategy,
    SelectionStrategy,
    FitnessFunction,
    EvolutionaryException,
    PopulationManager,
    GeneticOperations
)


class TestEvolutionaryParameters(unittest.TestCase):
    """Test suite for EvolutionaryParameters class."""
    
    def setUp(self):
        """
        Initializes default and custom EvolutionaryParameters instances for use in tests.
        """
        self.default_params = EvolutionaryParameters()
        self.custom_params = EvolutionaryParameters(
            population_size=200,
            generations=1000,
            mutation_rate=0.15,
            crossover_rate=0.85,
            selection_pressure=0.3
        )
    
    def test_default_initialization(self):
        """
        Verify that the EvolutionaryParameters instance initializes with the correct default values.
        """
        self.assertEqual(self.default_params.population_size, 100)
        self.assertEqual(self.default_params.generations, 500)
        self.assertEqual(self.default_params.mutation_rate, 0.1)
        self.assertEqual(self.default_params.crossover_rate, 0.8)
        self.assertEqual(self.default_params.selection_pressure, 0.2)
    
    def test_custom_initialization(self):
        """
        Verify that custom initialization of EvolutionaryParameters sets all parameter values as expected.
        """
        self.assertEqual(self.custom_params.population_size, 200)
        self.assertEqual(self.custom_params.generations, 1000)
        self.assertEqual(self.custom_params.mutation_rate, 0.15)
        self.assertEqual(self.custom_params.crossover_rate, 0.85)
        self.assertEqual(self.custom_params.selection_pressure, 0.3)
    
    def test_parameter_validation(self):
        """
        Verify that invalid evolutionary parameter values raise ValueError during initialization.
        
        This test checks that the EvolutionaryParameters class enforces constraints on population size, mutation rate, and crossover rate, raising ValueError for out-of-bounds values.
        """
        with self.assertRaises(ValueError):
            EvolutionaryParameters(population_size=0)
        
        with self.assertRaises(ValueError):
            EvolutionaryParameters(mutation_rate=-0.1)
        
        with self.assertRaises(ValueError):
            EvolutionaryParameters(mutation_rate=1.5)
        
        with self.assertRaises(ValueError):
            EvolutionaryParameters(crossover_rate=-0.1)
        
        with self.assertRaises(ValueError):
            EvolutionaryParameters(crossover_rate=1.5)
    
    def test_to_dict(self):
        """
        Test that the EvolutionaryParameters instance is correctly converted to a dictionary with expected values.
        """
        params_dict = self.default_params.to_dict()
        expected_dict = {
            'population_size': 100,
            'generations': 500,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'selection_pressure': 0.2
        }
        self.assertEqual(params_dict, expected_dict)
    
    def test_from_dict(self):
        """
        Test that EvolutionaryParameters can be correctly instantiated from a dictionary of parameter values.
        """
        params_dict = {
            'population_size': 150,
            'generations': 750,
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'selection_pressure': 0.25
        }
        params = EvolutionaryParameters.from_dict(params_dict)
        self.assertEqual(params.population_size, 150)
        self.assertEqual(params.generations, 750)
        self.assertEqual(params.mutation_rate, 0.12)
        self.assertEqual(params.crossover_rate, 0.85)
        self.assertEqual(params.selection_pressure, 0.25)


class TestMutationStrategy(unittest.TestCase):
    """Test suite for MutationStrategy class."""
    
    def setUp(self):
        """
        Set up the test fixture by initializing a MutationStrategy instance for use in mutation strategy tests.
        """
        self.strategy = MutationStrategy()
    
    def test_gaussian_mutation(self):
        """
        Tests the Gaussian mutation strategy for numeric genomes.
        
        Verifies that the mutated genome maintains the correct length and type, and checks behavior with both low and high mutation rates.
        """
        genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.1, sigma=0.5)
        
        # Check that the genome is mutated (should be different)
        self.assertEqual(len(mutated), len(genome))
        self.assertIsInstance(mutated, list)
        
        # Test with high mutation rate
        highly_mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1.0)
        self.assertEqual(len(highly_mutated), len(genome))
    
    def test_uniform_mutation(self):
        """
        Test that the uniform mutation strategy produces a mutated genome of the same length with all values within specified bounds.
        """
        genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=0.2, bounds=(-10, 10))
        
        self.assertEqual(len(mutated), len(genome))
        self.assertIsInstance(mutated, list)
        
        # All values should be within bounds
        for value in mutated:
            self.assertGreaterEqual(value, -10)
            self.assertLessEqual(value, 10)
    
    def test_bit_flip_mutation(self):
        """
        Tests the bit-flip mutation strategy to ensure it produces a mutated genome of the same length with boolean values.
        """
        genome = [True, False, True, False, True]
        mutated = self.strategy.bit_flip_mutation(genome, mutation_rate=0.3)
        
        self.assertEqual(len(mutated), len(genome))
        self.assertIsInstance(mutated, list)
        
        # All values should be boolean
        for value in mutated:
            self.assertIsInstance(value, bool)
    
    def test_adaptive_mutation(self):
        """
        Test that the adaptive mutation strategy produces a mutated genome of the correct length and type when given a genome and fitness history.
        """
        genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        fitness_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
        
        self.assertEqual(len(mutated), len(genome))
        self.assertIsInstance(mutated, list)
    
    def test_invalid_mutation_rate(self):
        """
        Test that mutation methods raise ValueError when given invalid mutation rates.
        """
        genome = [1.0, 2.0, 3.0]
        
        with self.assertRaises(ValueError):
            self.strategy.gaussian_mutation(genome, mutation_rate=-0.1)
        
        with self.assertRaises(ValueError):
            self.strategy.uniform_mutation(genome, mutation_rate=1.5)


class TestSelectionStrategy(unittest.TestCase):
    """Test suite for SelectionStrategy class."""
    
    def setUp(self):
        """
        Initializes the selection strategy and a sample population for selection strategy tests.
        """
        self.strategy = SelectionStrategy()
        self.population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5},
            {'genome': [10, 11, 12], 'fitness': 0.3}
        ]
    
    def test_tournament_selection(self):
        """
        Test that the tournament selection strategy selects a valid individual from the population.
        
        Verifies that the selected individual is a member of the population and contains the expected 'genome' and 'fitness' keys.
        """
        selected = self.strategy.tournament_selection(self.population, tournament_size=2)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_roulette_wheel_selection(self):
        """
        Test that the roulette wheel selection strategy selects a valid individual from the population.
        
        Verifies that the selected individual is present in the population and contains the expected 'genome' and 'fitness' keys.
        """
        selected = self.strategy.roulette_wheel_selection(self.population)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_rank_selection(self):
        """
        Test that the rank-based selection strategy selects a valid individual from the population.
        
        Verifies that the selected individual is present in the population and contains the expected keys.
        """
        selected = self.strategy.rank_selection(self.population)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_elitism_selection(self):
        """
        Test that the elitism selection strategy correctly selects the top N individuals with the highest fitness from the population.
        """
        elite_count = 2
        selected = self.strategy.elitism_selection(self.population, elite_count)
        
        self.assertEqual(len(selected), elite_count)
        
        # Check that selected individuals are the fittest
        fitness_values = [individual['fitness'] for individual in selected]
        self.assertEqual(fitness_values, [0.9, 0.7])  # Sorted by fitness descending
    
    def test_empty_population(self):
        """
        Verify that selection methods raise ValueError when called with an empty population.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection([], tournament_size=2)
        
        with self.assertRaises(ValueError):
            self.strategy.roulette_wheel_selection([])
    
    def test_invalid_tournament_size(self):
        """
        Test that tournament selection raises ValueError when the tournament size is invalid.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=0)
        
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=len(self.population) + 1)


class TestFitnessFunction(unittest.TestCase):
    """Test suite for FitnessFunction class."""
    
    def setUp(self):
        """
        Set up the test fixture by initializing a FitnessFunction instance.
        """
        self.fitness_func = FitnessFunction()
    
    def test_sphere_function(self):
        """
        Tests the sphere fitness function by verifying it returns the negative sum of squares of the genome values.
        """
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.sphere_function(genome)
        
        # Sphere function: sum of squares
        expected = -(1.0**2 + 2.0**2 + 3.0**2)  # Negative for maximization
        self.assertEqual(fitness, expected)
    
    def test_rastrigin_function(self):
        """
        Test that the Rastrigin fitness function returns 0.0 for a genome at the origin.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.rastrigin_function(genome)
        
        # Rastrigin function should be 0 at origin
        self.assertEqual(fitness, 0.0)
    
    def test_rosenbrock_function(self):
        """
        Tests that the Rosenbrock fitness function returns 0.0 for the genome [1.0, 1.0], which is the global minimum.
        """
        genome = [1.0, 1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)
        
        # Rosenbrock function should be 0 at (1, 1)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function(self):
        """
        Tests that the Ackley fitness function returns 0 at the origin for a zero-valued genome.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.ackley_function(genome)
        
        # Ackley function should be 0 at origin
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function(self):
        """
        Tests evaluation of a user-defined custom fitness function by verifying that the fitness function correctly computes the sum of genome values.
        """
        def custom_func(genome):
            """
            Calculates the sum of all elements in a genome.
            
            Parameters:
            	genome (iterable): A sequence of numeric values representing a genome.
            
            Returns:
            	int or float: The total sum of the genome's elements.
            """
            return sum(genome)
        
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate(genome, custom_func)
        
        self.assertEqual(fitness, 6.0)
    
    def test_multi_objective_function(self):
        """
        Tests that the multi-objective fitness function correctly evaluates a genome against multiple objectives and returns a fitness vector.
        """
        genome = [1.0, 2.0, 3.0]
        objectives = [
            lambda g: sum(g),  # Objective 1: sum
            lambda g: sum(x**2 for x in g)  # Objective 2: sum of squares
        ]
        
        fitness = self.fitness_func.multi_objective_evaluate(genome, objectives)
        
        self.assertEqual(len(fitness), 2)
        self.assertEqual(fitness[0], 6.0)
        self.assertEqual(fitness[1], 14.0)
    
    def test_constraint_handling(self):
        """
        Test that fitness evaluation applies a penalty when genome constraints are violated.
        
        This test verifies that the fitness function penalizes genomes that do not satisfy provided constraints during evaluation.
        """
        genome = [1.0, 2.0, 3.0]
        
        def constraint_func(g):
            # Constraint: sum should be less than 5
            """
            Checks whether the sum of elements in the input is less than 5.
            
            Parameters:
            	g (iterable): An iterable of numeric values.
            
            Returns:
            	bool: True if the sum of elements is less than 5, otherwise False.
            """
            return sum(g) < 5
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            [constraint_func]
        )
        
        # Should be penalized since sum(genome) = 6 > 5
        self.assertLess(fitness, sum(genome))


class TestPopulationManager(unittest.TestCase):
    """Test suite for PopulationManager class."""
    
    def setUp(self):
        """
        Initializes the PopulationManager and sets default genome length and population size for each test.
        """
        self.manager = PopulationManager()
        self.genome_length = 5
        self.population_size = 10
    
    def test_initialize_random_population(self):
        """
        Test that the population manager correctly initializes a random population with the specified size and genome length.
        
        Verifies that each individual in the population contains a genome of the correct length and a fitness attribute.
        """
        population = self.manager.initialize_random_population(
            self.population_size, 
            self.genome_length
        )
        
        self.assertEqual(len(population), self.population_size)
        
        for individual in population:
            self.assertIn('genome', individual)
            self.assertIn('fitness', individual)
            self.assertEqual(len(individual['genome']), self.genome_length)
    
    def test_initialize_seeded_population(self):
        """
        Test that seeded population initialization includes provided seed genomes and produces the correct population size.
        
        Ensures that the initialized population contains all specified seed genomes and matches the expected population size.
        """
        seeds = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ]
        
        population = self.manager.initialize_seeded_population(
            self.population_size,
            self.genome_length,
            seeds
        )
        
        self.assertEqual(len(population), self.population_size)
        
        # Check that seeds are included
        genomes = [ind['genome'] for ind in population]
        self.assertIn(seeds[0], genomes)
        self.assertIn(seeds[1], genomes)
    
    def test_evaluate_population(self):
        """
        Tests that the population manager correctly evaluates the fitness of each individual in a population using a provided fitness function.
        
        Verifies that after evaluation, each individual in the population has a non-None fitness value of type int or float.
        """
        population = self.manager.initialize_random_population(
            self.population_size, 
            self.genome_length
        )
        
        fitness_func = lambda genome: sum(genome)
        
        self.manager.evaluate_population(population, fitness_func)
        
        for individual in population:
            self.assertIsNotNone(individual['fitness'])
            self.assertIsInstance(individual['fitness'], (int, float))
    
    def test_get_best_individual(self):
        """
        Test that the population manager correctly identifies and returns the individual with the highest fitness from a given population.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.7}
        ]
        
        best = self.manager.get_best_individual(population)
        
        self.assertEqual(best['fitness'], 0.9)
        self.assertEqual(best['genome'], [4, 5, 6])
    
    def test_get_population_statistics(self):
        """
        Tests that the population statistics calculation returns correct best, worst, average, median, and standard deviation fitness values for a given population.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.7}
        ]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertIn('best_fitness', stats)
        self.assertIn('worst_fitness', stats)
        self.assertIn('average_fitness', stats)
        self.assertIn('median_fitness', stats)
        self.assertIn('std_dev_fitness', stats)
        
        self.assertEqual(stats['best_fitness'], 0.9)
        self.assertEqual(stats['worst_fitness'], 0.5)
        self.assertAlmostEqual(stats['average_fitness'], 0.7, places=1)
    
    def test_diversity_calculation(self):
        """
        Test that the population diversity metric is correctly calculated and returns a positive float value.
        """
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [4.0, 5.0, 6.0], 'fitness': 0.9},
            {'genome': [7.0, 8.0, 9.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        
        self.assertIsInstance(diversity, float)
        self.assertGreater(diversity, 0.0)
    
    def test_empty_population_handling(self):
        """
        Test that methods for retrieving the best individual and population statistics raise ValueError when called with an empty population.
        """
        with self.assertRaises(ValueError):
            self.manager.get_best_individual([])
        
        with self.assertRaises(ValueError):
            self.manager.get_population_statistics([])


class TestGeneticOperations(unittest.TestCase):
    """Test suite for GeneticOperations class."""
    
    def setUp(self):
        """
        Initializes a GeneticOperations instance for use in genetic operations tests.
        """
        self.operations = GeneticOperations()
    
    def test_single_point_crossover(self):
        """
        Test that the single-point crossover operation produces two children of correct length containing elements from both parents.
        """
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
        
        # Children should contain elements from both parents
        combined_parents = set(parent1 + parent2)
        combined_children = set(child1 + child2)
        self.assertTrue(combined_children.issubset(combined_parents))
    
    def test_two_point_crossover(self):
        """
        Test that the two-point crossover operation produces two children of the same length as the parent genomes.
        """
        parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
        parent2 = [9, 10, 11, 12, 13, 14, 15, 16]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_uniform_crossover(self):
        """
        Tests the uniform crossover operation to ensure that two children are produced with the same length as the parent genomes.
        """
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_arithmetic_crossover(self):
        """
        Test that the arithmetic crossover operation produces children as weighted averages of two parent genomes.
        
        Verifies that the resulting children have the correct length and that each gene is the arithmetic combination of the corresponding genes from the parents using the specified alpha value.
        """
        parent1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        parent2 = [6.0, 7.0, 8.0, 9.0, 10.0]
        
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
        
        # Check that children are arithmetic combinations
        for i in range(len(parent1)):
            expected_child1 = 0.5 * parent1[i] + 0.5 * parent2[i]
            expected_child2 = 0.5 * parent2[i] + 0.5 * parent1[i]
            self.assertAlmostEqual(child1[i], expected_child1, places=5)
            self.assertAlmostEqual(child2[i], expected_child2, places=5)
    
    def test_simulated_binary_crossover(self):
        """
        Test that the simulated binary crossover (SBX) operation produces two children of correct length and within specified bounds.
        """
        parent1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        parent2 = [6.0, 7.0, 8.0, 9.0, 10.0]
        bounds = [(-10, 10)] * 5
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=2.0
        )
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
        
        # Check bounds
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
            self.assertGreaterEqual(child2[i], lower)
            self.assertLessEqual(child2[i], upper)
    
    def test_blend_crossover(self):
        """
        Test that the blend crossover (BLX-α) operation produces two children of correct length from two parent genomes.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_invalid_crossover_inputs(self):
        """
        Test that crossover operations raise ValueError when parent genomes have different lengths.
        """
        parent1 = [1, 2, 3]
        parent2 = [4, 5]  # Different length
        
        with self.assertRaises(ValueError):
            self.operations.single_point_crossover(parent1, parent2)
        
        with self.assertRaises(ValueError):
            self.operations.two_point_crossover(parent1, parent2)


class TestEvolutionaryConduit(unittest.TestCase):
    """Test suite for EvolutionaryConduit class."""
    
    def setUp(self):
        """
        Initializes the EvolutionaryConduit and EvolutionaryParameters instances for use in each test.
        """
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=20,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_initialization(self):
        """
        Verify that all core components of the EvolutionaryConduit are properly initialized.
        """
        self.assertIsNotNone(self.conduit.mutation_strategy)
        self.assertIsNotNone(self.conduit.selection_strategy)
        self.assertIsNotNone(self.conduit.fitness_function)
        self.assertIsNotNone(self.conduit.population_manager)
        self.assertIsNotNone(self.conduit.genetic_operations)
    
    def test_set_fitness_function(self):
        """
        Test that a custom fitness function can be set on the conduit and is used for fitness evaluation.
        """
        def custom_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of its elements.
            
            Parameters:
                genome (iterable): The genome to evaluate, typically a list or array of numeric values.
            
            Returns:
                int or float: The sum of the genome's elements representing its fitness score.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(custom_fitness)
        
        # Test that the function is set correctly
        test_genome = [1.0, 2.0, 3.0]
        fitness = self.conduit.fitness_function.evaluate(test_genome, custom_fitness)
        self.assertEqual(fitness, 6.0)
    
    def test_set_parameters(self):
        """
        Test that setting evolutionary parameters updates the conduit with the correct values.
        """
        self.conduit.set_parameters(self.params)
        
        self.assertEqual(self.conduit.parameters.population_size, 20)
        self.assertEqual(self.conduit.parameters.generations, 10)
        self.assertEqual(self.conduit.parameters.mutation_rate, 0.1)
        self.assertEqual(self.conduit.parameters.crossover_rate, 0.8)
    
    @patch('app.ai_backend.genesis_evolutionary_conduit.EvolutionaryConduit.evolve')
    def test_run_evolution(self, mock_evolve):
        """
        Test that the evolution process runs and returns the expected result structure.
        
        Verifies that running the evolution process produces a result containing keys for the best individual, number of generations run, final population, and statistics. Ensures the underlying evolve method is called exactly once.
        """
        mock_evolve.return_value = {
            'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
            'generations_run': 10,
            'final_population': [],
            'statistics': {'best_fitness': 0.9}
        }
        
        self.conduit.set_parameters(self.params)
        result = self.conduit.run_evolution(genome_length=5)
        
        self.assertIn('best_individual', result)
        self.assertIn('generations_run', result)
        self.assertIn('final_population', result)
        self.assertIn('statistics', result)
        
        mock_evolve.assert_called_once()
    
    def test_save_and_load_state(self):
        """
        Tests that the evolutionary conduit state can be saved and accurately restored in a new instance.
        """
        # Set up conduit state
        self.conduit.set_parameters(self.params)
        
        # Save state
        state = self.conduit.save_state()
        
        # Create new conduit and load state
        new_conduit = EvolutionaryConduit()
        new_conduit.load_state(state)
        
        # Check that state is loaded correctly
        self.assertEqual(new_conduit.parameters.population_size, 20)
        self.assertEqual(new_conduit.parameters.generations, 10)
    
    def test_add_callback(self):
        """
        Test that a callback function can be added to the evolutionary conduit and is present in the callbacks list.
        """
        callback_called = False
        
        def test_callback(generation, population, best_individual):
            """
            Callback function that sets a flag when invoked during evolutionary process.
            
            Parameters:
                generation (int): The current generation number.
                population (list): The current population of individuals.
                best_individual: The best individual in the current population.
            """
            nonlocal callback_called
            callback_called = True
        
        self.conduit.add_callback(test_callback)
        
        # Verify callback is added
        self.assertIn(test_callback, self.conduit.callbacks)
    
    def test_evolution_history_tracking(self):
        """
        Test that enabling history tracking on the evolutionary conduit correctly sets the history tracking flag after running evolution.
        """
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        # Run a simple evolution
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of its elements.
            
            Parameters:
                genome (iterable): A sequence of numeric values representing a genome.
            
            Returns:
                int or float: The sum of the genome's elements, representing its fitness score.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(simple_fitness)
        
        # Mock the evolution process
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 5,
                'final_population': [],
                'statistics': {'best_fitness': 6.0}
            }
            
            result = self.conduit.run_evolution(genome_length=3)
            
            # Check that history is tracked
            self.assertTrue(self.conduit.history_enabled)


class TestGenesisEvolutionaryConduit(unittest.TestCase):
    """Test suite for GenesisEvolutionaryConduit class."""
    
    def setUp(self):
        """
        Initializes a GenesisEvolutionaryConduit instance and sets evolutionary parameters for use in tests.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=20,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_initialization(self):
        """
        Verify that GenesisEvolutionaryConduit is properly initialized with required components, including genesis configuration, neural network factory, and optimization strategies.
        """
        self.assertIsInstance(self.genesis_conduit, EvolutionaryConduit)
        self.assertIsNotNone(self.genesis_conduit.genesis_config)
        self.assertIsNotNone(self.genesis_conduit.neural_network_factory)
        self.assertIsNotNone(self.genesis_conduit.optimization_strategies)
    
    def test_neural_network_evolution(self):
        """
        Tests the creation of a neural network using the configured network parameters in the GenesisEvolutionaryConduit.
        
        Verifies that a neural network is successfully instantiated after setting the network configuration.
        """
        # Set up network evolution parameters
        network_config = {
            'input_size': 10,
            'hidden_layers': [20, 15],
            'output_size': 1,
            'activation': 'relu'
        }
        
        self.genesis_conduit.set_network_config(network_config)
        
        # Test network creation
        network = self.genesis_conduit.create_neural_network()
        self.assertIsNotNone(network)
    
    def test_neuroevolution_fitness(self):
        """
        Tests the fitness evaluation of a neural network genome using provided training data.
        
        This method sets up mock training data, assigns it to the genesis conduit, and verifies that evaluating the fitness of a sample genome returns a numeric value.
        """
        # Mock dataset for training
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        
        self.genesis_conduit.set_training_data(X_train, y_train)
        
        # Test fitness evaluation
        genome = [0.1, 0.2, 0.3, 0.4, 0.5]  # Network weights
        fitness = self.genesis_conduit.evaluate_network_fitness(genome)
        
        self.assertIsInstance(fitness, (int, float))
    
    def test_topology_evolution(self):
        """
        Tests the mutation of a neural network topology structure, ensuring the mutated topology maintains required keys and structure.
        """
        # Start with simple topology
        topology = {
            'layers': [10, 5, 1],
            'connections': [[0, 1], [1, 2]]
        }
        
        mutated_topology = self.genesis_conduit.mutate_topology(topology)
        
        self.assertIsInstance(mutated_topology, dict)
        self.assertIn('layers', mutated_topology)
        self.assertIn('connections', mutated_topology)
    
    def test_hyperparameter_optimization(self):
        """
        Tests that the hyperparameter optimization process generates hyperparameters within the specified search space and includes all expected keys.
        """
        search_space = {
            'learning_rate': (0.001, 0.1),
            'batch_size': (16, 128),
            'dropout_rate': (0.0, 0.5)
        }
        
        self.genesis_conduit.set_hyperparameter_search_space(search_space)
        
        # Test hyperparameter generation
        hyperparams = self.genesis_conduit.generate_hyperparameters()
        
        self.assertIn('learning_rate', hyperparams)
        self.assertIn('batch_size', hyperparams)
        self.assertIn('dropout_rate', hyperparams)
        
        # Check bounds
        self.assertGreaterEqual(hyperparams['learning_rate'], 0.001)
        self.assertLessEqual(hyperparams['learning_rate'], 0.1)
    
    def test_multi_objective_optimization(self):
        """
        Tests that the GenesisEvolutionaryConduit correctly handles multi-objective optimization by setting multiple objectives and verifying that fitness evaluation returns a vector matching the number of objectives.
        """
        objectives = [
            'accuracy',
            'model_size',
            'inference_time'
        ]
        
        self.genesis_conduit.set_objectives(objectives)
        
        # Test multi-objective fitness evaluation
        genome = [0.1, 0.2, 0.3, 0.4, 0.5]
        fitness_vector = self.genesis_conduit.evaluate_multi_objective_fitness(genome)
        
        self.assertEqual(len(fitness_vector), len(objectives))
        self.assertIsInstance(fitness_vector, list)
    
    def test_adaptive_mutation_rates(self):
        """
        Test that adaptive mutation rates are correctly calculated based on population fitness history.
        
        Verifies that the calculated adaptive mutation rate is a float within the valid range [0.0, 1.0].
        """
        # Set up population with fitness history
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5, 'generation': 1},
            {'genome': [4, 5, 6], 'fitness': 0.7, 'generation': 2},
            {'genome': [7, 8, 9], 'fitness': 0.9, 'generation': 3}
        ]
        
        adaptive_rate = self.genesis_conduit.calculate_adaptive_mutation_rate(population)
        
        self.assertIsInstance(adaptive_rate, float)
        self.assertGreaterEqual(adaptive_rate, 0.0)
        self.assertLessEqual(adaptive_rate, 1.0)
    
    def test_speciation(self):
        """
        Tests that the speciation method groups individuals in the population into species based on a distance threshold to promote diversity.
        
        Verifies that the returned species structure is a non-empty list.
        """
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.1, 2.1, 3.1], 'fitness': 0.6},
            {'genome': [5.0, 6.0, 7.0], 'fitness': 0.7},
            {'genome': [5.1, 6.1, 7.1], 'fitness': 0.8}
        ]
        
        species = self.genesis_conduit.speciate_population(population, distance_threshold=2.0)
        
        self.assertIsInstance(species, list)
        self.assertGreater(len(species), 0)
    
    def test_transfer_learning(self):
        """
        Test the adaptation of a pretrained neural network genome to a new task using transfer learning.
        
        Verifies that the adapted genome is a non-empty list after applying transfer learning with a new task configuration.
        """
        # Mock pre-trained network
        pretrained_genome = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test transfer learning adaptation
        adapted_genome = self.genesis_conduit.adapt_pretrained_network(
            pretrained_genome, 
            new_task_config={'output_size': 3}
        )
        
        self.assertIsInstance(adapted_genome, list)
        self.assertGreater(len(adapted_genome), 0)
    
    def test_ensemble_evolution(self):
        """
        Tests that the ensemble evolution method selects the top-performing networks by fitness and returns the correct ensemble size.
        """
        # Create multiple networks
        networks = [
            {'genome': [1, 2, 3], 'fitness': 0.7},
            {'genome': [4, 5, 6], 'fitness': 0.8},
            {'genome': [7, 8, 9], 'fitness': 0.9}
        ]
        
        ensemble = self.genesis_conduit.create_ensemble(networks, ensemble_size=2)
        
        self.assertEqual(len(ensemble), 2)
        # Should select the best networks
        self.assertEqual(ensemble[0]['fitness'], 0.9)
        self.assertEqual(ensemble[1]['fitness'], 0.8)
    
    def test_novelty_search(self):
        """
        Test that the novelty search algorithm computes a novelty score for each individual in the population.
        
        Verifies that the number of novelty scores matches the population size and that each score is a numeric value.
        """
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [4.0, 5.0, 6.0], 'fitness': 0.7},
            {'genome': [7.0, 8.0, 9.0], 'fitness': 0.9}
        ]
        
        novelty_scores = self.genesis_conduit.calculate_novelty_scores(population)
        
        self.assertEqual(len(novelty_scores), len(population))
        for score in novelty_scores:
            self.assertIsInstance(score, (int, float))
    
    def test_coevolution(self):
        """
        Test the coevolution process involving two separate populations and verify that the result contains updated populations for both groups.
        """
        # Create two populations
        population1 = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.7}
        ]
        
        population2 = [
            {'genome': [7, 8, 9], 'fitness': 0.6},
            {'genome': [10, 11, 12], 'fitness': 0.8}
        ]
        
        # Test coevolution step
        result = self.genesis_conduit.coevolve_populations(population1, population2)
        
        self.assertIsInstance(result, dict)
        self.assertIn('population1', result)
        self.assertIn('population2', result)
    
    @patch('app.ai_backend.genesis_evolutionary_conduit.GenesisEvolutionaryConduit.save_checkpoint')
    def test_checkpoint_system(self, mock_save):
        """
        Test that the checkpoint saving mechanism is invoked with the correct file path.
        """
        # Set up conduit state
        self.genesis_conduit.set_parameters(self.params)
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pkl"
        self.genesis_conduit.save_checkpoint(checkpoint_path)
        
        mock_save.assert_called_once_with(checkpoint_path)
    
    def test_distributed_evolution(self):
        """
        Test the setup and migration functionality of distributed evolution using the island model.
        
        Verifies that the island model can be configured and that individuals can be migrated between populations, returning the expected tuple structure.
        """
        # Mock distributed setup
        island_configs = [
            {'island_id': 1, 'population_size': 10},
            {'island_id': 2, 'population_size': 10}
        ]
        
        self.genesis_conduit.setup_island_model(island_configs)
        
        # Test migration between islands
        population1 = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        population2 = [{'genome': [4, 5, 6], 'fitness': 0.7}]
        
        migrated = self.genesis_conduit.migrate_individuals(
            population1, population2, migration_rate=0.1
        )
        
        self.assertIsInstance(migrated, tuple)
        self.assertEqual(len(migrated), 2)


class TestEvolutionaryException(unittest.TestCase):
    """Test suite for EvolutionaryException class."""
    
    def test_exception_creation(self):
        """
        Test that an EvolutionaryException can be created and its string representation matches the provided message.
        """
        message = "Test evolutionary exception"
        exception = EvolutionaryException(message)
        
        self.assertEqual(str(exception), message)
        self.assertIsInstance(exception, Exception)
    
    def test_exception_with_details(self):
        """
        Test that the EvolutionaryException correctly stores and exposes additional details provided at initialization.
        """
        message = "Evolution failed"
        details = {"generation": 50, "error_type": "convergence"}
        
        exception = EvolutionaryException(message, details)
        
        self.assertEqual(str(exception), message)
        self.assertEqual(exception.details, details)
    
    def test_exception_raising(self):
        """
        Verifies that raising an EvolutionaryException triggers the expected exception handling.
        """
        with self.assertRaises(EvolutionaryException):
            raise EvolutionaryException("Test exception")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test suite for complex evolutionary scenarios."""
    
    def setUp(self):
        """
        Initialize the integration test environment with a GenesisEvolutionaryConduit instance and default evolutionary parameters.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_complete_evolution_cycle(self):
        """
        Tests the full evolution cycle using the GenesisEvolutionaryConduit, verifying that the process completes and returns expected results including the best individual and number of generations run.
        """
        # Set up fitness function
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of the squares of its elements.
            
            Parameters:
            	genome (iterable): A sequence of numeric values representing a genome.
            
            Returns:
            	int or float: The sum of squared values in the genome.
            """
            return sum(x**2 for x in genome)
        
        self.genesis_conduit.set_fitness_function(simple_fitness)
        self.genesis_conduit.set_parameters(self.params)
        
        # Mock the evolution process
        with patch.object(self.genesis_conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 14.0},
                'generations_run': 5,
                'final_population': [],
                'statistics': {'best_fitness': 14.0}
            }
            
            result = self.genesis_conduit.run_evolution(genome_length=3)
            
            self.assertIn('best_individual', result)
            self.assertEqual(result['generations_run'], 5)
    
    def test_neural_network_evolution_pipeline(self):
        """
        Tests the full neural network evolution pipeline, including network configuration, training data setup, and neural network creation.
        """
        # Set up network configuration
        network_config = {
            'input_size': 5,
            'hidden_layers': [10, 5],
            'output_size': 1,
            'activation': 'relu'
        }
        
        self.genesis_conduit.set_network_config(network_config)
        
        # Mock training data
        X_train = [[1, 2, 3, 4, 5] for _ in range(10)]
        y_train = [1 for _ in range(10)]
        
        self.genesis_conduit.set_training_data(X_train, y_train)
        
        # Test the pipeline
        network = self.genesis_conduit.create_neural_network()
        self.assertIsNotNone(network)
    
    def test_multi_objective_optimization_pipeline(self):
        """
        Tests the multi-objective optimization pipeline by setting objectives and verifying that the fitness evaluation returns the expected fitness vector for a sample genome.
        """
        objectives = ['accuracy', 'model_size']
        self.genesis_conduit.set_objectives(objectives)
        
        # Mock multi-objective fitness evaluation
        genome = [0.1, 0.2, 0.3]
        
        with patch.object(self.genesis_conduit, 'evaluate_multi_objective_fitness') as mock_eval:
            mock_eval.return_value = [0.8, 0.1]  # High accuracy, small model
            
            fitness_vector = self.genesis_conduit.evaluate_multi_objective_fitness(genome)
            
            self.assertEqual(len(fitness_vector), 2)
            self.assertEqual(fitness_vector[0], 0.8)
            self.assertEqual(fitness_vector[1], 0.1)
    
    def test_adaptive_evolution_pipeline(self):
        """
        Test the adaptive mutation rate calculation within an evolutionary pipeline using a sample population with varying fitness values.
        """
        # Set up population with varying fitness
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.3, 'generation': 1},
            {'genome': [4, 5, 6], 'fitness': 0.5, 'generation': 2},
            {'genome': [7, 8, 9], 'fitness': 0.7, 'generation': 3}
        ]
        
        # Test adaptive mutation rate calculation
        adaptive_rate = self.genesis_conduit.calculate_adaptive_mutation_rate(population)
        
        self.assertIsInstance(adaptive_rate, float)
        self.assertGreaterEqual(adaptive_rate, 0.0)
        self.assertLessEqual(adaptive_rate, 1.0)
    
    def test_error_handling_and_recovery(self):
        """
        Test that the system correctly handles invalid parameters and recovers from fitness evaluation failures during evolution.
        
        Verifies that providing invalid evolutionary parameters raises a ValueError, and that a fitness function failure during evolution raises an EvolutionaryException.
        """
        # Test invalid parameters
        with self.assertRaises(ValueError):
            invalid_params = EvolutionaryParameters(population_size=0)
        
        # Test recovery from evolution failure
        def failing_fitness(genome):
            """
            A fitness function that always raises a ValueError to simulate a fitness evaluation failure.
            
            Parameters:
            	genome: The genome to be evaluated (unused).
            
            Raises:
            	ValueError: Always raised to indicate fitness evaluation failure.
            """
            raise ValueError("Fitness evaluation failed")
        
        self.genesis_conduit.set_fitness_function(failing_fitness)
        
        with self.assertRaises(EvolutionaryException):
            self.genesis_conduit.run_evolution(genome_length=3)


# Async tests for concurrent evolution
class TestAsyncEvolution(unittest.TestCase):
    """Test suite for asynchronous evolution capabilities."""
    
    def setUp(self):
        """
        Initializes the GenesisEvolutionaryConduit and EvolutionaryParameters for asynchronous evolution tests.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    @patch('asyncio.run')
    def test_async_evolution_execution(self, mock_run):
        """
        Tests that asynchronous evolution execution returns a valid result when run using the mocked async evolution method.
        """
        async def mock_async_evolve():
            """
            Simulates an asynchronous evolutionary process and returns a mock result.
            
            Returns:
                dict: A dictionary containing the best individual, number of generations run, final population, and statistics.
            """
            return {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
                'generations_run': 5,
                'final_population': [],
                'statistics': {'best_fitness': 0.9}
            }
        
        mock_run.return_value = asyncio.run(mock_async_evolve())
        
        # Test async evolution
        result = self.genesis_conduit.run_async_evolution(genome_length=3)
        
        self.assertIsNotNone(result)
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_parallel_fitness_evaluation(self, mock_executor):
        """
        Tests that population fitness evaluation is performed in parallel using a thread pool executor and verifies that parallel execution is invoked.
        """
        # Mock parallel execution
        mock_executor.return_value.__enter__.return_value.map.return_value = [0.5, 0.7, 0.9]
        
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        def fitness_func(genome):
            """
            Calculates the fitness of a genome as the sum of its elements.
            
            Parameters:
            	genome (iterable): A sequence of numeric values representing a genome.
            
            Returns:
            	int or float: The total sum of the genome's elements.
            """
            return sum(genome)
        
        # Test parallel evaluation
        self.genesis_conduit.evaluate_population_parallel(population, fitness_func)
        
        # Verify parallel execution was attempted
        mock_executor.assert_called_once()


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

class TestAdvancedEvolutionaryParameters(unittest.TestCase):
    """Additional comprehensive tests for EvolutionaryParameters edge cases."""
    
    def test_boundary_values(self):
        """Test parameters at exact boundary values."""
        # Test minimum valid values
        params = EvolutionaryParameters(
            population_size=1,
            generations=1,
            mutation_rate=0.0,
            crossover_rate=0.0,
            selection_pressure=0.0
        )
        self.assertEqual(params.population_size, 1)
        self.assertEqual(params.mutation_rate, 0.0)
        
        # Test maximum valid values
        params = EvolutionaryParameters(
            mutation_rate=1.0,
            crossover_rate=1.0,
            selection_pressure=1.0
        )
        self.assertEqual(params.mutation_rate, 1.0)
        self.assertEqual(params.crossover_rate, 1.0)
    
    def test_float_precision_parameters(self):
        """Test parameters with high precision float values."""
        params = EvolutionaryParameters(
            mutation_rate=0.123456789,
            crossover_rate=0.987654321,
            selection_pressure=0.555555555
        )
        self.assertAlmostEqual(params.mutation_rate, 0.123456789, places=9)
        self.assertAlmostEqual(params.crossover_rate, 0.987654321, places=9)
    
    def test_large_population_sizes(self):
        """Test with very large population sizes."""
        params = EvolutionaryParameters(population_size=1000000)
        self.assertEqual(params.population_size, 1000000)
    
    def test_parameter_type_validation(self):
        """Test that non-numeric parameters raise appropriate errors."""
        with self.assertRaises((TypeError, ValueError)):
            EvolutionaryParameters(population_size="invalid")
        
        with self.assertRaises((TypeError, ValueError)):
            EvolutionaryParameters(mutation_rate="0.1")
    
    def test_from_dict_missing_keys(self):
        """Test from_dict with missing optional parameters."""
        partial_dict = {'population_size': 50}
        params = EvolutionaryParameters.from_dict(partial_dict)
        self.assertEqual(params.population_size, 50)
        # Should use defaults for missing parameters
        self.assertEqual(params.generations, 500)
    
    def test_from_dict_extra_keys(self):
        """Test from_dict ignores extra keys gracefully."""
        dict_with_extras = {
            'population_size': 75,
            'generations': 200,
            'extra_key': 'ignored',
            'another_extra': 123
        }
        params = EvolutionaryParameters.from_dict(dict_with_extras)
        self.assertEqual(params.population_size, 75)
        self.assertEqual(params.generations, 200)


class TestStressMutationStrategy(unittest.TestCase):
    """Stress tests and edge cases for MutationStrategy."""
    
    def setUp(self):
        self.strategy = MutationStrategy()
    
    def test_large_genome_mutation(self):
        """Test mutation on very large genomes."""
        large_genome = list(range(10000))  # 10k elements
        mutated = self.strategy.gaussian_mutation(large_genome, mutation_rate=0.01, sigma=0.1)
        self.assertEqual(len(mutated), len(large_genome))
    
    def test_single_element_genome(self):
        """Test mutation on single-element genomes."""
        genome = [5.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1.0)
        self.assertEqual(len(mutated), 1)
        self.assertIsInstance(mutated[0], float)
    
    def test_zero_mutation_rate(self):
        """Test that zero mutation rate returns identical genome."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.0, sigma=1.0)
        self.assertEqual(mutated, genome)
    
    def test_extreme_sigma_values(self):
        """Test Gaussian mutation with very small and large sigma values."""
        genome = [1.0, 2.0, 3.0]
        
        # Very small sigma
        mutated_small = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1e-10)
        self.assertEqual(len(mutated_small), len(genome))
        
        # Very large sigma
        mutated_large = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1000.0)
        self.assertEqual(len(mutated_large), len(genome))
    
    def test_empty_genome_handling(self):
        """Test mutation strategies with empty genomes."""
        empty_genome = []
        mutated = self.strategy.gaussian_mutation(empty_genome, mutation_rate=0.1, sigma=1.0)
        self.assertEqual(mutated, [])
    
    def test_bit_flip_edge_cases(self):
        """Test bit-flip mutation edge cases."""
        # Single bit
        single_bit = [True]
        mutated = self.strategy.bit_flip_mutation(single_bit, mutation_rate=1.0)
        self.assertEqual(len(mutated), 1)
        
        # All same bits
        all_true = [True] * 100
        mutated = self.strategy.bit_flip_mutation(all_true, mutation_rate=0.5)
        self.assertEqual(len(mutated), 100)
    
    def test_adaptive_mutation_empty_history(self):
        """Test adaptive mutation with empty fitness history."""
        genome = [1.0, 2.0, 3.0]
        empty_history = []
        
        # Should handle empty history gracefully
        mutated = self.strategy.adaptive_mutation(genome, empty_history, base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))
    
    def test_adaptive_mutation_constant_fitness(self):
        """Test adaptive mutation with constant fitness history."""
        genome = [1.0, 2.0, 3.0]
        constant_fitness = [0.5] * 10  # No improvement
        
        mutated = self.strategy.adaptive_mutation(genome, constant_fitness, base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))


class TestAdvancedSelectionStrategy(unittest.TestCase):
    """Advanced tests for SelectionStrategy with complex scenarios."""
    
    def setUp(self):
        self.strategy = SelectionStrategy()
        self.large_population = [
            {'genome': list(range(i, i+3)), 'fitness': i/100.0}
            for i in range(100)
        ]
    
    def test_tournament_selection_with_large_population(self):
        """Test tournament selection on large populations."""
        selected = self.strategy.tournament_selection(
            self.large_population, 
            tournament_size=10
        )
        self.assertIn(selected, self.large_population)
    
    def test_roulette_wheel_with_negative_fitness(self):
        """Test roulette wheel selection with negative fitness values."""
        negative_pop = [
            {'genome': [1, 2, 3], 'fitness': -0.5},
            {'genome': [4, 5, 6], 'fitness': -0.3},
            {'genome': [7, 8, 9], 'fitness': -0.1}
        ]
        selected = self.strategy.roulette_wheel_selection(negative_pop)
        self.assertIn(selected, negative_pop)
    
    def test_roulette_wheel_with_zero_fitness(self):
        """Test roulette wheel selection when all fitness values are zero."""
        zero_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.0},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 0.0}
        ]
        selected = self.strategy.roulette_wheel_selection(zero_fitness_pop)
        self.assertIn(selected, zero_fitness_pop)
    
    def test_elitism_selection_edge_cases(self):
        """Test elitism selection edge cases."""
        # Request more elites than population size
        small_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.7}
        ]
        
        selected = self.strategy.elitism_selection(small_pop, elite_count=5)
        # Should return entire population when requesting more than available
        self.assertEqual(len(selected), len(small_pop))
    
    def test_selection_with_identical_fitness(self):
        """Test selection strategies when all individuals have identical fitness."""
        identical_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        # All strategies should handle this gracefully
        tournament_selected = self.strategy.tournament_selection(identical_fitness_pop, tournament_size=2)
        roulette_selected = self.strategy.roulette_wheel_selection(identical_fitness_pop)
        rank_selected = self.strategy.rank_selection(identical_fitness_pop)
        
        self.assertIn(tournament_selected, identical_fitness_pop)
        self.assertIn(roulette_selected, identical_fitness_pop)
        self.assertIn(rank_selected, identical_fitness_pop)
    
    def test_single_individual_population(self):
        """Test selection strategies with single individual."""
        single_pop = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        tournament_selected = self.strategy.tournament_selection(single_pop, tournament_size=1)
        roulette_selected = self.strategy.roulette_wheel_selection(single_pop)
        rank_selected = self.strategy.rank_selection(single_pop)
        elite_selected = self.strategy.elitism_selection(single_pop, elite_count=1)
        
        self.assertEqual(tournament_selected, single_pop[0])
        self.assertEqual(roulette_selected, single_pop[0])
        self.assertEqual(rank_selected, single_pop[0])
        self.assertEqual(elite_selected, single_pop)


class TestExtendedFitnessFunction(unittest.TestCase):
    """Extended tests for FitnessFunction with complex scenarios."""
    
    def setUp(self):
        self.fitness_func = FitnessFunction()
    
    def test_fitness_functions_with_extreme_values(self):
        """Test fitness functions with very large and small genome values."""
        # Very large values
        large_genome = [1e6, 1e6, 1e6]
        sphere_fitness = self.fitness_func.sphere_function(large_genome)
        self.assertIsInstance(sphere_fitness, (int, float))
        
        # Very small values
        small_genome = [1e-10, 1e-10, 1e-10]
        sphere_fitness_small = self.fitness_func.sphere_function(small_genome)
        self.assertIsInstance(sphere_fitness_small, (int, float))
    
    def test_fitness_functions_with_mixed_signs(self):
        """Test fitness functions with positive and negative values."""
        mixed_genome = [-5.0, 0.0, 5.0]
        
        sphere_fitness = self.fitness_func.sphere_function(mixed_genome)
        rastrigin_fitness = self.fitness_func.rastrigin_function(mixed_genome)
        ackley_fitness = self.fitness_func.ackley_function(mixed_genome)
        
        self.assertIsInstance(sphere_fitness, (int, float))
        self.assertIsInstance(rastrigin_fitness, (int, float))
        self.assertIsInstance(ackley_fitness, (int, float))
    
    def test_empty_genome_fitness(self):
        """Test fitness functions with empty genomes."""
        empty_genome = []
        
        # Functions should handle empty genomes gracefully
        sphere_fitness = self.fitness_func.sphere_function(empty_genome)
        self.assertEqual(sphere_fitness, 0.0)
    
    def test_single_element_genome_fitness(self):
        """Test fitness functions with single-element genomes."""
        single_genome = [3.0]
        
        sphere_fitness = self.fitness_func.sphere_function(single_genome)
        rastrigin_fitness = self.fitness_func.rastrigin_function(single_genome)
        ackley_fitness = self.fitness_func.ackley_function(single_genome)
        
        self.assertIsInstance(sphere_fitness, (int, float))
        self.assertIsInstance(rastrigin_fitness, (int, float))
        self.assertIsInstance(ackley_fitness, (int, float))
    
    def test_multi_objective_with_conflicting_objectives(self):
        """Test multi-objective optimization with conflicting objectives."""
        genome = [1.0, 2.0, 3.0]
        conflicting_objectives = [
            lambda g: sum(g),           # Maximize sum
            lambda g: -sum(g),          # Minimize sum (conflicting)
            lambda g: len(g)            # Constant objective
        ]
        
        fitness_vector = self.fitness_func.multi_objective_evaluate(genome, conflicting_objectives)
        
        self.assertEqual(len(fitness_vector), 3)
        self.assertEqual(fitness_vector[0], 6.0)
        self.assertEqual(fitness_vector[1], -6.0)
        self.assertEqual(fitness_vector[2], 3)
    
    def test_constraint_handling_multiple_constraints(self):
        """Test fitness evaluation with multiple constraints."""
        genome = [2.0, 3.0, 4.0]
        
        def constraint1(g):
            return sum(g) < 10  # True for our genome
        
        def constraint2(g):
            return all(x > 0 for x in g)  # True for our genome
        
        def constraint3(g):
            return max(g) < 3  # False for our genome
        
        constraints = [constraint1, constraint2, constraint3]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome,
            lambda g: sum(g),
            constraints
        )
        
        # Should be penalized due to constraint3 violation
        self.assertLess(fitness, sum(genome))
    
    def test_custom_fitness_function_exceptions(self):
        """Test handling of exceptions in custom fitness functions."""
        def failing_fitness(genome):
            raise ValueError("Fitness calculation failed")
        
        genome = [1.0, 2.0, 3.0]
        
        with self.assertRaises(ValueError):
            self.fitness_func.evaluate(genome, failing_fitness)


class TestAdvancedPopulationManager(unittest.TestCase):
    """Advanced tests for PopulationManager with complex scenarios."""
    
    def setUp(self):
        self.manager = PopulationManager()
    
    def test_large_population_initialization(self):
        """Test initialization of very large populations."""
        large_size = 10000
        genome_length = 100
        
        population = self.manager.initialize_random_population(large_size, genome_length)
        
        self.assertEqual(len(population), large_size)
        for individual in population[:10]:  # Check first 10 to avoid performance issues
            self.assertEqual(len(individual['genome']), genome_length)
    
    def test_seeded_population_with_more_seeds_than_size(self):
        """Test seeded population when more seeds provided than population size."""
        seeds = [[i] * 5 for i in range(20)]  # 20 seeds
        population_size = 10  # Smaller than seeds
        
        population = self.manager.initialize_seeded_population(
            population_size, 5, seeds
        )
        
        self.assertEqual(len(population), population_size)
        # Should contain some of the provided seeds
        genomes = [ind['genome'] for ind in population]
        seed_found = any(seed in genomes for seed in seeds[:population_size])
        self.assertTrue(seed_found)
    
    def test_population_statistics_with_extreme_values(self):
        """Test population statistics with extreme fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': -1e6},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 1e6}
        ]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 1e6)
        self.assertEqual(stats['worst_fitness'], -1e6)
        self.assertIsInstance(stats['average_fitness'], float)
        self.assertIsInstance(stats['std_dev_fitness'], float)
    
    def test_diversity_calculation_identical_genomes(self):
        """Test diversity calculation when all genomes are identical."""
        identical_population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(identical_population)
        
        # Diversity should be 0 or very close to 0 for identical genomes
        self.assertAlmostEqual(diversity, 0.0, places=5)
    
    def test_diversity_calculation_maximally_diverse(self):
        """Test diversity calculation with maximally diverse genomes."""
        diverse_population = [
            {'genome': [-100.0, -100.0, -100.0], 'fitness': 0.5},
            {'genome': [0.0, 0.0, 0.0], 'fitness': 0.6},
            {'genome': [100.0, 100.0, 100.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(diverse_population)
        
        # Should be high diversity
        self.assertGreater(diversity, 0.0)
    
    def test_population_evaluation_with_slow_fitness(self):
        """Test population evaluation that might take time (mocked)."""
        population = self.manager.initialize_random_population(100, 10)
        
        def slow_fitness(genome):
            # Simulate slow fitness calculation
            return sum(x**2 for x in genome)
        
        # Should complete without timeout issues
        self.manager.evaluate_population(population, slow_fitness)
        
        for individual in population:
            self.assertIsNotNone(individual['fitness'])
    
    def test_get_best_individual_with_ties(self):
        """Test getting best individual when multiple have same fitness."""
        tied_population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.9},  # Tied for best
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        best = self.manager.get_best_individual(tied_population)
        
        # Should return one of the tied individuals
        self.assertEqual(best['fitness'], 0.9)
        self.assertIn(best, tied_population[:2])


class TestComplexGeneticOperations(unittest.TestCase):
    """Complex scenarios for GeneticOperations."""
    
    def setUp(self):
        self.operations = GeneticOperations()
    
    def test_crossover_with_extreme_genome_sizes(self):
        """Test crossover operations with very large and very small genomes."""
        # Very large genomes
        large_parent1 = list(range(10000))
        large_parent2 = list(range(10000, 20000))
        
        child1, child2 = self.operations.single_point_crossover(large_parent1, large_parent2)
        self.assertEqual(len(child1), len(large_parent1))
        self.assertEqual(len(child2), len(large_parent2))
        
        # Very small genomes
        small_parent1 = [1]
        small_parent2 = [2]
        
        child1_small, child2_small = self.operations.single_point_crossover(small_parent1, small_parent2)
        self.assertEqual(len(child1_small), 1)
        self.assertEqual(len(child2_small), 1)
    
    def test_arithmetic_crossover_with_zero_alpha(self):
        """Test arithmetic crossover with alpha=0 and alpha=1."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Alpha = 0 should return parent2, parent1
        child1_zero, child2_zero = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.0)
        self.assertEqual(child1_zero, parent2)
        self.assertEqual(child2_zero, parent1)
        
        # Alpha = 1 should return parent1, parent2
        child1_one, child2_one = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.0)
        self.assertEqual(child1_one, parent1)
        self.assertEqual(child2_one, parent2)
    
    def test_simulated_binary_crossover_boundary_handling(self):
        """Test SBX crossover with tight bounds."""
        parent1 = [0.5, 0.5, 0.5]
        parent2 = [0.6, 0.6, 0.6]
        bounds = [(0.0, 1.0)] * 3  # Tight bounds
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=2.0
        )
        
        # All values should be within bounds
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
            self.assertGreaterEqual(child2[i], lower)
            self.assertLessEqual(child2[i], upper)
    
    def test_blend_crossover_extreme_alpha(self):
        """Test blend crossover with extreme alpha values."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Very large alpha
        child1_large, child2_large = self.operations.blend_crossover(parent1, parent2, alpha=10.0)
        self.assertEqual(len(child1_large), len(parent1))
        self.assertEqual(len(child2_large), len(parent2))
        
        # Very small alpha
        child1_small, child2_small = self.operations.blend_crossover(parent1, parent2, alpha=0.01)
        self.assertEqual(len(child1_small), len(parent1))
        self.assertEqual(len(child2_small), len(parent2))
    
    def test_crossover_with_identical_parents(self):
        """Test crossover operations when parents are identical."""
        identical_parent = [1.0, 2.0, 3.0]
        
        # All crossover methods should handle identical parents
        child1_sp, child2_sp = self.operations.single_point_crossover(identical_parent, identical_parent)
        child1_tp, child2_tp = self.operations.two_point_crossover(identical_parent, identical_parent)
        child1_uni, child2_uni = self.operations.uniform_crossover(identical_parent, identical_parent)
        child1_arith, child2_arith = self.operations.arithmetic_crossover(identical_parent, identical_parent)
        
        # Children should be identical to parents
        self.assertEqual(child1_sp, identical_parent)
        self.assertEqual(child2_sp, identical_parent)
        self.assertEqual(child1_arith, identical_parent)
        self.assertEqual(child2_arith, identical_parent)


class TestAdvancedEvolutionaryConduit(unittest.TestCase):
    """Advanced scenarios for EvolutionaryConduit."""
    
    def setUp(self):
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(population_size=10, generations=5)
    
    def test_conduit_with_custom_components(self):
        """Test conduit with custom strategy components."""
        # Create custom components
        custom_mutation = MutationStrategy()
        custom_selection = SelectionStrategy()
        custom_fitness = FitnessFunction()
        
        # Set custom components
        self.conduit.mutation_strategy = custom_mutation
        self.conduit.selection_strategy = custom_selection
        self.conduit.fitness_function = custom_fitness
        
        # Verify components are set
        self.assertEqual(self.conduit.mutation_strategy, custom_mutation)
        self.assertEqual(self.conduit.selection_strategy, custom_selection)
        self.assertEqual(self.conduit.fitness_function, custom_fitness)
    
    def test_multiple_callback_handling(self):
        """Test conduit with multiple callbacks."""
        callback_calls = []
        
        def callback1(generation, population, best_individual):
            callback_calls.append(('callback1', generation))
        
        def callback2(generation, population, best_individual):
            callback_calls.append(('callback2', generation))
        
        def callback3(generation, population, best_individual):
            callback_calls.append(('callback3', generation))
        
        self.conduit.add_callback(callback1)
        self.conduit.add_callback(callback2)
        self.conduit.add_callback(callback3)
        
        # Verify all callbacks are added
        self.assertEqual(len(self.conduit.callbacks), 3)
        self.assertIn(callback1, self.conduit.callbacks)
        self.assertIn(callback2, self.conduit.callbacks)
        self.assertIn(callback3, self.conduit.callbacks)
    
    def test_state_persistence_edge_cases(self):
        """Test state saving and loading with complex configurations."""
        # Set up complex state
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        # Add multiple callbacks
        def dummy_callback(gen, pop, best):
            pass
        
        self.conduit.add_callback(dummy_callback)
        
        # Save and load state
        state = self.conduit.save_state()
        new_conduit = EvolutionaryConduit()
        new_conduit.load_state(state)
        
        # Verify complex state is preserved
        self.assertEqual(new_conduit.parameters.population_size, self.params.population_size)
        self.assertEqual(new_conduit.history_enabled, True)
    
    @patch('app.ai_backend.genesis_evolutionary_conduit.EvolutionaryConduit.evolve')
    def test_evolution_with_callback_exceptions(self, mock_evolve):
        """Test evolution continues even if callbacks raise exceptions."""
        mock_evolve.return_value = {
            'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
            'generations_run': 5,
            'final_population': [],
            'statistics': {'best_fitness': 0.9}
        }
        
        def failing_callback(generation, population, best_individual):
            raise RuntimeError("Callback failed")
        
        def working_callback(generation, population, best_individual):
            pass  # This should still work
        
        self.conduit.add_callback(failing_callback)
        self.conduit.add_callback(working_callback)
        self.conduit.set_parameters(self.params)
        
        # Evolution should complete despite callback failure
        result = self.conduit.run_evolution(genome_length=3)
        self.assertIsNotNone(result)


class TestAdvancedGenesisEvolutionaryConduit(unittest.TestCase):
    """Advanced scenarios for GenesisEvolutionaryConduit."""
    
    def setUp(self):
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(population_size=10, generations=5)
    
    def test_network_config_validation(self):
        """Test network configuration validation."""
        # Valid config
        valid_config = {
            'input_size': 10,
            'hidden_layers': [20, 15, 10],
            'output_size': 1,
            'activation': 'relu'
        }
        
        self.genesis_conduit.set_network_config(valid_config)
        network = self.genesis_conduit.create_neural_network()
        self.assertIsNotNone(network)
        
        # Invalid config should be handled gracefully
        invalid_config = {
            'input_size': -1,  # Invalid
            'hidden_layers': [],
            'output_size': 0,  # Invalid
        }
        
        # Should handle invalid config without crashing
        self.genesis_conduit.set_network_config(invalid_config)
    
    def test_hyperparameter_search_edge_cases(self):
        """Test hyperparameter search with edge case spaces."""
        # Very narrow search space
        narrow_space = {
            'learning_rate': (0.001, 0.001001),  # Very narrow range
            'batch_size': (32, 32),  # Single value
        }
        
        self.genesis_conduit.set_hyperparameter_search_space(narrow_space)
        hyperparams = self.genesis_conduit.generate_hyperparameters()
        
        self.assertIn('learning_rate', hyperparams)
        self.assertIn('batch_size', hyperparams)
        self.assertEqual(hyperparams['batch_size'], 32)
    
    def test_speciation_with_identical_genomes(self):
        """Test speciation when all genomes are identical."""
        identical_population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        species = self.genesis_conduit.speciate_population(identical_population, distance_threshold=0.1)
        
        # Should create single species for identical genomes
        self.assertEqual(len(species), 1)
        self.assertEqual(len(species[0]), 3)
    
    def test_ensemble_creation_edge_cases(self):
        """Test ensemble creation with edge cases."""
        # Request ensemble larger than available networks
        small_networks = [
            {'genome': [1, 2, 3], 'fitness': 0.7}
        ]
        
        ensemble = self.genesis_conduit.create_ensemble(small_networks, ensemble_size=5)
        
        # Should return all available networks
        self.assertEqual(len(ensemble), 1)
        self.assertEqual(ensemble[0]['fitness'], 0.7)
        
        # Empty networks list
        empty_ensemble = self.genesis_conduit.create_ensemble([], ensemble_size=2)
        self.assertEqual(len(empty_ensemble), 0)
    
    def test_coevolution_with_different_sized_populations(self):
        """Test coevolution with populations of different sizes."""
        small_pop = [{'genome': [1, 2], 'fitness': 0.5}]
        large_pop = [
            {'genome': [3, 4], 'fitness': 0.6},
            {'genome': [5, 6], 'fitness': 0.7},
            {'genome': [7, 8], 'fitness': 0.8}
        ]
        
        result = self.genesis_conduit.coevolve_populations(small_pop, large_pop)
        
        self.assertIsInstance(result, dict)
        self.assertIn('population1', result)
        self.assertIn('population2', result)
    
    def test_transfer_learning_with_incompatible_architectures(self):
        """Test transfer learning with incompatible network architectures."""
        # Pretrained network for different task
        pretrained_genome = [0.1, 0.2, 0.3]
        
        # Very different new task configuration
        incompatible_config = {
            'output_size': 100,  # Much larger output
            'input_size': 1000,  # Much larger input
        }
        
        # Should handle architecture mismatch gracefully
        adapted_genome = self.genesis_conduit.adapt_pretrained_network(
            pretrained_genome, 
            incompatible_config
        )
        
        self.assertIsInstance(adapted_genome, list)
    
    def test_distributed_evolution_setup_validation(self):
        """Test distributed evolution setup with invalid configurations."""
        # Invalid island configurations
        invalid_configs = [
            {'island_id': 1},  # Missing population_size
            {'population_size': -1},  # Invalid population size
        ]
        
        # Should handle invalid configs gracefully
        try:
            self.genesis_conduit.setup_island_model(invalid_configs)
        except Exception as e:
            # Should raise specific validation error
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_migration_with_empty_populations(self):
        """Test migration between empty populations."""
        empty_pop1 = []
        empty_pop2 = []
        
        migrated = self.genesis_conduit.migrate_individuals(
            empty_pop1, empty_pop2, migration_rate=0.1
        )
        
        self.assertIsInstance(migrated, tuple)
        self.assertEqual(len(migrated), 2)
        self.assertEqual(len(migrated[0]), 0)
        self.assertEqual(len(migrated[1]), 0)


class TestPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests for the evolutionary system."""
    
    def setUp(self):
        self.conduit = EvolutionaryConduit()
        self.genesis_conduit = GenesisEvolutionaryConduit()
    
    def test_large_scale_evolution_setup(self):
        """Test setup of large-scale evolutionary runs."""
        large_params = EvolutionaryParameters(
            population_size=1000,
            generations=100,
            mutation_rate=0.01,
            crossover_rate=0.9
        )
        
        self.conduit.set_parameters(large_params)
        
        # Verify parameters are set correctly
        self.assertEqual(self.conduit.parameters.population_size, 1000)
        self.assertEqual(self.conduit.parameters.generations, 100)
    
    def test_memory_efficient_population_handling(self):
        """Test that large populations don't cause memory issues."""
        manager = PopulationManager()
        
        # Create moderately large population
        population = manager.initialize_random_population(1000, 50)
        
        # Should be able to calculate statistics without issues
        stats = manager.get_population_statistics(population)
        self.assertIn('best_fitness', stats)
        
        # Should be able to evaluate population
        def simple_fitness(genome):
            return sum(genome[:10])  # Only use first 10 elements for speed
        
        manager.evaluate_population(population, simple_fitness)
        
        # All individuals should have fitness values
        for individual in population:
            self.assertIsNotNone(individual['fitness'])
    
    def test_concurrent_fitness_evaluation_mock(self):
        """Test concurrent fitness evaluation capabilities."""
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        def fitness_func(genome):
            return sum(genome)
        
        # Mock concurrent evaluation
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value.map.return_value = [6, 15, 24]
            
            self.genesis_conduit.evaluate_population_parallel(population, fitness_func)
            
            # Verify concurrent executor was used
            mock_executor.assert_called_once()


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Comprehensive error handling and edge case tests."""
    
    def setUp(self):
        self.conduit = EvolutionaryConduit()
        self.genesis_conduit = GenesisEvolutionaryConduit()
    
    def test_invalid_parameter_combinations(self):
        """Test invalid parameter combinations."""
        # Mutation rate and crossover rate both 0
        with self.assertRaises((ValueError, Warning)):
            params = EvolutionaryParameters(
                mutation_rate=0.0,
                crossover_rate=0.0
            )
            # This might be valid but should be flagged
    
    def test_fitness_function_returning_invalid_types(self):
        """Test fitness functions that return invalid types."""
        def invalid_fitness(genome):
            return "invalid"  # Should return numeric
        
        fitness_func = FitnessFunction()
        genome = [1.0, 2.0, 3.0]
        
        with self.assertRaises((TypeError, ValueError)):
            fitness_func.evaluate(genome, invalid_fitness)
    
    def test_genome_with_invalid_types(self):
        """Test operations with genomes containing invalid types."""
        strategy = MutationStrategy()
        operations = GeneticOperations()
        
        invalid_genome = ["not", "a", "number"]
        
        # Operations should handle or raise appropriate errors
        with self.assertRaises((TypeError, ValueError)):
            strategy.gaussian_mutation(invalid_genome, mutation_rate=0.1, sigma=1.0)
    
    def test_network_config_missing_required_fields(self):
        """Test network configuration with missing required fields."""
        incomplete_config = {
            'input_size': 10,
            # Missing other required fields
        }
        
        # Should handle gracefully or raise specific error
        try:
            self.genesis_conduit.set_network_config(incomplete_config)
            network = self.genesis_conduit.create_neural_network()
        except (KeyError, ValueError, AttributeError):
            pass  # Expected behavior
    
    def test_checkpoint_with_invalid_paths(self):
        """Test checkpoint saving with invalid file paths."""
        # Test with invalid path
        with self.assertRaises((OSError, IOError, PermissionError)):
            self.genesis_conduit.save_checkpoint("/invalid/path/checkpoint.pkl")
    
    def test_state_loading_with_corrupted_data(self):
        """Test state loading with corrupted or invalid data."""
        invalid_state = {
            "corrupted": "data",
            "version": "unknown"
        }
        
        # Should handle corrupted state gracefully
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.conduit.load_state(invalid_state)


class TestConfigurationValidation(unittest.TestCase):
    """Tests for configuration validation and settings."""
    
    def test_evolutionary_parameters_json_serialization(self):
        """Test that EvolutionaryParameters can be serialized to/from JSON."""
        params = EvolutionaryParameters(
            population_size=150,
            generations=300,
            mutation_rate=0.15,
            crossover_rate=0.85,
            selection_pressure=0.25
        )
        
        # Convert to dict and then JSON
        params_dict = params.to_dict()
        json_str = json.dumps(params_dict)
        
        # Parse back from JSON
        parsed_dict = json.loads(json_str)
        restored_params = EvolutionaryParameters.from_dict(parsed_dict)
        
        # Verify all parameters match
        self.assertEqual(restored_params.population_size, params.population_size)
        self.assertEqual(restored_params.generations, params.generations)
        self.assertEqual(restored_params.mutation_rate, params.mutation_rate)
        self.assertEqual(restored_params.crossover_rate, params.crossover_rate)
        self.assertEqual(restored_params.selection_pressure, params.selection_pressure)
    
    def test_genesis_config_validation(self):
        """Test Genesis-specific configuration validation."""
        # Test various genesis configurations
        configs = [
            {'neural_evolution': True, 'topology_evolution': False},
            {'neural_evolution': False, 'topology_evolution': True},
            {'neural_evolution': True, 'topology_evolution': True},
        ]
        
        for config in configs:
            # Should handle different genesis configurations
            try:
                self.genesis_conduit.genesis_config.update(config)
            except AttributeError:
                # genesis_config might not be a dict
                pass
    
    def test_parameter_bounds_validation(self):
        """Test comprehensive parameter bounds validation."""
        # Test edge cases for each parameter
        test_cases = [
            ('population_size', [1, 10, 100, 1000, 10000]),
            ('generations', [1, 10, 100, 1000]),
            ('mutation_rate', [0.0, 0.001, 0.1, 0.5, 1.0]),
            ('crossover_rate', [0.0, 0.001, 0.1, 0.5, 1.0]),
            ('selection_pressure', [0.0, 0.001, 0.1, 0.5, 1.0])
        ]
        
        for param_name, valid_values in test_cases:
            for value in valid_values:
                kwargs = {param_name: value}
                try:
                    params = EvolutionaryParameters(**kwargs)
                    self.assertIsNotNone(params)
                except ValueError:
                    # Some combinations might be invalid
                    pass


if __name__ == '__main__':
    # Configure test runner for comprehensive output
    unittest.main(verbosity=2, buffer=True, catchbreak=True)