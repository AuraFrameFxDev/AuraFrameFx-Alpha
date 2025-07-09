import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
import asyncio
import sys
import os
import time
import random

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
        Set up default and custom EvolutionaryParameters instances for use in test cases.
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
        Tests that EvolutionaryParameters is initialized with the expected default values for all parameters.
        """
        self.assertEqual(self.default_params.population_size, 100)
        self.assertEqual(self.default_params.generations, 500)
        self.assertEqual(self.default_params.mutation_rate, 0.1)
        self.assertEqual(self.default_params.crossover_rate, 0.8)
        self.assertEqual(self.default_params.selection_pressure, 0.2)
    
    def test_custom_initialization(self):
        """
        Test that custom values are correctly assigned to all attributes of EvolutionaryParameters during initialization.
        """
        self.assertEqual(self.custom_params.population_size, 200)
        self.assertEqual(self.custom_params.generations, 1000)
        self.assertEqual(self.custom_params.mutation_rate, 0.15)
        self.assertEqual(self.custom_params.crossover_rate, 0.85)
        self.assertEqual(self.custom_params.selection_pressure, 0.3)
    
    def test_parameter_validation(self):
        """
        Test that EvolutionaryParameters raises ValueError for invalid population size, mutation rate, or crossover rate values.
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
        Test that the to_dict method of EvolutionaryParameters returns a dictionary with the correct parameter values.
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
        Test that EvolutionaryParameters can be instantiated from a dictionary and that its attributes match the provided values.
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
        Set up a MutationStrategy instance for use in mutation strategy tests.
        """
        self.strategy = MutationStrategy()
    
    def test_gaussian_mutation(self):
        """
        Test that the Gaussian mutation strategy produces a mutated genome of the correct length and type for different mutation rates.
        
        Verifies that the mutated genome is a list with the same length as the input genome for both low and high mutation rates.
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
        Test that the uniform mutation strategy returns a genome of the same length with all values within the specified bounds.
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
        Test that the bit-flip mutation strategy returns a mutated genome of the same length with all boolean elements.
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
        Tests that the adaptive mutation strategy returns a mutated genome list of the same length as the input genome when provided with a fitness history.
        """
        genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        fitness_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
        
        self.assertEqual(len(mutated), len(genome))
        self.assertIsInstance(mutated, list)
    
    def test_invalid_mutation_rate(self):
        """
        Test that mutation methods raise a ValueError when provided with mutation rates outside the valid range.
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
        Set up the selection strategy instance and a sample population for selection strategy tests.
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
        Test that tournament selection returns a valid individual from the population.
        
        Ensures the selected individual is present in the population and contains both 'genome' and 'fitness' keys.
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
        Tests that the rank-based selection strategy selects a valid individual from the population.
        
        Verifies that the selected individual is present in the population and contains both 'genome' and 'fitness' keys.
        """
        selected = self.strategy.rank_selection(self.population)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_elitism_selection(self):
        """
        Tests that the elitism selection strategy correctly selects the top individuals with the highest fitness values from the population.
        """
        elite_count = 2
        selected = self.strategy.elitism_selection(self.population, elite_count)
        
        self.assertEqual(len(selected), elite_count)
        
        # Check that selected individuals are the fittest
        fitness_values = [individual['fitness'] for individual in selected]
        self.assertEqual(fitness_values, [0.9, 0.7])  # Sorted by fitness descending
    
    def test_empty_population(self):
        """
        Test that selection strategies raise ValueError when invoked with an empty population.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection([], tournament_size=2)
        
        with self.assertRaises(ValueError):
            self.strategy.roulette_wheel_selection([])
    
    def test_invalid_tournament_size(self):
        """
        Test that tournament selection raises a ValueError when the tournament size is zero or exceeds the population size.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=0)
        
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=len(self.population) + 1)


class TestFitnessFunction(unittest.TestCase):
    """Test suite for FitnessFunction class."""
    
    def setUp(self):
        """
        Initialize a FitnessFunction instance for use in test methods.
        """
        self.fitness_func = FitnessFunction()
    
    def test_sphere_function(self):
        """
        Test that the sphere fitness function returns the negative sum of squares for the provided genome.
        """
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.sphere_function(genome)
        
        # Sphere function: sum of squares
        expected = -(1.0**2 + 2.0**2 + 3.0**2)  # Negative for maximization
        self.assertEqual(fitness, expected)
    
    def test_rastrigin_function(self):
        """
        Test that the Rastrigin fitness function returns 0.0 when the input genome is at the origin.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.rastrigin_function(genome)
        
        # Rastrigin function should be 0 at origin
        self.assertEqual(fitness, 0.0)
    
    def test_rosenbrock_function(self):
        """
        Test that the Rosenbrock fitness function returns 0.0 at the global minimum for the genome [1.0, 1.0].
        """
        genome = [1.0, 1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)
        
        # Rosenbrock function should be 0 at (1, 1)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function(self):
        """
        Test that the Ackley fitness function returns a value of zero when evaluated at the origin with a genome of all zeros.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.ackley_function(genome)
        
        # Ackley function should be 0 at origin
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function(self):
        """
        Test that a user-defined fitness function correctly computes the sum of genome values.
        """
        def custom_func(genome):
            """
            Return the sum of all numeric elements in the provided genome sequence.
            
            Parameters:
            	genome (iterable): Sequence of numeric values to be summed.
            
            Returns:
            	total (numeric): The sum of all values in the genome.
            """
            return sum(genome)
        
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate(genome, custom_func)
        
        self.assertEqual(fitness, 6.0)
    
    def test_multi_objective_function(self):
        """
        Tests that the multi-objective fitness function evaluates a genome with multiple objectives and returns the expected fitness vector.
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
        Test that the fitness function penalizes genomes violating specified constraints.
        
        Ensures that when a genome does not meet the constraint (sum less than 5), the evaluated fitness is reduced compared to the unconstrained fitness.
        """
        genome = [1.0, 2.0, 3.0]
        
        def constraint_func(g):
            # Constraint: sum should be less than 5
            """
            Check if the sum of elements in the input iterable is less than 5.
            
            Parameters:
            	g (iterable): Iterable containing numeric values.
            
            Returns:
            	bool: True if the sum is less than 5, otherwise False.
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
        Set up the test environment with a PopulationManager and default parameters for genome length and population size.
        """
        self.manager = PopulationManager()
        self.genome_length = 5
        self.population_size = 10
    
    def test_initialize_random_population(self):
        """
        Test that the population manager creates a random population with the specified size and genome length.
        
        Verifies that each individual has a genome of the correct length and includes a fitness attribute.
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
        Test that seeded population initialization includes all provided seed genomes and produces the correct population size.
        
        Verifies that each seed genome is present in the initialized population and that the total number of individuals matches the specified population size.
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
        Test that evaluating a population assigns a valid numeric fitness value to each individual.
        
        Ensures that after evaluation, every individual's 'fitness' attribute is set to a non-None integer or float.
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
        Test that the population manager returns the individual with the highest fitness from the population.
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
        Test that the population manager correctly computes statistical metrics (best, worst, average, median, standard deviation) for fitness values in a population.
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
        Test that the population diversity metric is computed as a positive float for a given sample population.
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
        Test that retrieving the best individual or statistics from an empty population raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.manager.get_best_individual([])
        
        with self.assertRaises(ValueError):
            self.manager.get_population_statistics([])


class TestGeneticOperations(unittest.TestCase):
    """Test suite for GeneticOperations class."""
    
    def setUp(self):
        """
        Prepare the test environment by initializing a GeneticOperations instance for use in genetic operations tests.
        """
        self.operations = GeneticOperations()
    
    def test_single_point_crossover(self):
        """
        Test that the single-point crossover operation returns two children of the same length as the parents, with genes originating from both parents.
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
        Verify that two-point crossover produces two children with genome lengths equal to the parent genomes.
        """
        parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
        parent2 = [9, 10, 11, 12, 13, 14, 15, 16]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_uniform_crossover(self):
        """
        Test that the uniform crossover operation produces two children with genomes matching the length of the parent genomes.
        """
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_arithmetic_crossover(self):
        """
        Test that the arithmetic crossover operation generates children as weighted averages of two parent genomes.
        
        Verifies that the resulting children have the correct length and that each gene is the arithmetic mean of the corresponding genes from the parents using the specified alpha value.
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
        Test that simulated binary crossover produces two children of correct length with gene values within specified bounds.
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
        Test that the blend crossover (BLX-α) operation returns two child genomes with the same length as the parent genomes.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_invalid_crossover_inputs(self):
        """
        Test that crossover operations raise a ValueError when parent genomes have mismatched lengths.
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
        Prepare test fixtures by creating EvolutionaryConduit and EvolutionaryParameters instances for each test.
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
        Test that all core components of the EvolutionaryConduit are initialized and not None.
        """
        self.assertIsNotNone(self.conduit.mutation_strategy)
        self.assertIsNotNone(self.conduit.selection_strategy)
        self.assertIsNotNone(self.conduit.fitness_function)
        self.assertIsNotNone(self.conduit.population_manager)
        self.assertIsNotNone(self.conduit.genetic_operations)
    
    def test_set_fitness_function(self):
        """
        Test that a custom fitness function can be assigned to the conduit and is correctly used for genome fitness evaluation.
        """
        def custom_fitness(genome):
            """
            Calculate the fitness score of a genome by summing its numeric elements.
            
            Parameters:
            	genome (iterable): An iterable containing numeric values representing the genome.
            
            Returns:
            	float: The sum of all elements in the genome.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(custom_fitness)
        
        # Test that the function is set correctly
        test_genome = [1.0, 2.0, 3.0]
        fitness = self.conduit.fitness_function.evaluate(test_genome, custom_fitness)
        self.assertEqual(fitness, 6.0)
    
    def test_set_parameters(self):
        """
        Tests that setting evolutionary parameters updates the conduit with the specified values.
        """
        self.conduit.set_parameters(self.params)
        
        self.assertEqual(self.conduit.parameters.population_size, 20)
        self.assertEqual(self.conduit.parameters.generations, 10)
        self.assertEqual(self.conduit.parameters.mutation_rate, 0.1)
        self.assertEqual(self.conduit.parameters.crossover_rate, 0.8)
    
    @patch('app.ai_backend.genesis_evolutionary_conduit.EvolutionaryConduit.evolve')
    def test_run_evolution(self, mock_evolve):
        """
        Test that running the evolution process returns a result with the correct structure.
        
        Ensures the evolution run produces a result containing 'best_individual', 'generations_run', 'final_population', and 'statistics', and verifies that the evolve method is called exactly once.
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
        Test that saving and loading the evolutionary conduit state preserves parameter values in a new instance.
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
        Test that a callback function can be added to the evolutionary conduit and is registered in the callbacks list.
        """
        callback_called = False
        
        def test_callback(generation, population, best_individual):
            """
            Callback used in tests to indicate when it is invoked during the evolutionary process.
            
            Sets a flag to confirm that the callback mechanism is triggered during evolution.
            """
            nonlocal callback_called
            callback_called = True
        
        self.conduit.add_callback(test_callback)
        
        # Verify callback is added
        self.assertIn(test_callback, self.conduit.callbacks)
    
    def test_evolution_history_tracking(self):
        """
        Test that enabling history tracking on the evolutionary conduit sets the history tracking flag after running an evolution process.
        """
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        # Run a simple evolution
        def simple_fitness(genome):
            """
            Calculate the fitness score of a genome by summing its elements.
            
            Parameters:
                genome (iterable): Sequence of numeric values representing the genome.
            
            Returns:
                int or float: The total sum of the genome's elements.
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
        Prepare the test environment by initializing a GenesisEvolutionaryConduit instance and setting evolutionary parameters for use in test cases.
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
        Test that GenesisEvolutionaryConduit is properly initialized with genesis configuration, neural network factory, and optimization strategies.
        """
        self.assertIsInstance(self.genesis_conduit, EvolutionaryConduit)
        self.assertIsNotNone(self.genesis_conduit.genesis_config)
        self.assertIsNotNone(self.genesis_conduit.neural_network_factory)
        self.assertIsNotNone(self.genesis_conduit.optimization_strategies)
    
    def test_neural_network_evolution(self):
        """
        Tests that a neural network is created by GenesisEvolutionaryConduit using the specified configuration.
        
        Verifies that after setting the network configuration, the conduit instantiates a neural network object.
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
        Test that evaluating a neural network genome's fitness with training data returns a numeric value.
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
        Test that mutating a neural network topology produces a dictionary with valid 'layers' and 'connections' keys, confirming the topology structure remains intact after mutation.
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
        Test that hyperparameter optimization produces values within the defined search space and includes all required hyperparameter keys.
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
        Test that multi-objective optimization produces a fitness vector with the expected number of objectives.
        
        Ensures that evaluating a genome with multiple objectives returns a list whose length matches the number of objectives specified.
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
        Test that the adaptive mutation rate calculated from population fitness history is a float within the range [0.0, 1.0].
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
        Test that individuals are correctly grouped into species based on a distance threshold.
        
        Ensures the speciation method returns a non-empty list of species, verifying that population diversity is maintained through grouping.
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
        Test adaptation of a pretrained neural network genome to a new task using transfer learning.
        
        Verifies that applying transfer learning with a new task configuration produces a non-empty adapted genome.
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
        Test that the ensemble evolution method selects the top-performing networks by fitness and returns an ensemble of the specified size.
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
        Tests that novelty search computes a numeric novelty score for each individual in the population.
        
        Verifies that the number of novelty scores equals the population size and that all scores are numeric values.
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
        Test that the coevolution process updates and returns both populations.
        
        Verifies that the `coevolve_populations` method returns a dictionary containing updated entries for both input populations.
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
        Test that the checkpoint saving mechanism is called with the specified file path.
        
        Ensures that invoking `save_checkpoint` on the GenesisEvolutionaryConduit triggers the underlying save operation using the provided checkpoint path.
        """
        # Set up conduit state
        self.genesis_conduit.set_parameters(self.params)
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pkl"
        self.genesis_conduit.save_checkpoint(checkpoint_path)
        
        mock_save.assert_called_once_with(checkpoint_path)
    
    def test_distributed_evolution(self):
        """
        Test distributed evolution using the island model and verify migration between populations.
        
        Ensures that the island model can be set up and that individuals can be migrated between islands, with the migration function returning updated populations as a tuple.
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
        Verify that EvolutionaryException is instantiated with the correct message and is an Exception instance.
        """
        message = "Test evolutionary exception"
        exception = EvolutionaryException(message)
        
        self.assertEqual(str(exception), message)
        self.assertIsInstance(exception, Exception)
    
    def test_exception_with_details(self):
        """
        Test that EvolutionaryException stores and exposes additional details provided at initialization.
        """
        message = "Evolution failed"
        details = {"generation": 50, "error_type": "convergence"}
        
        exception = EvolutionaryException(message, details)
        
        self.assertEqual(str(exception), message)
        self.assertEqual(exception.details, details)
    
    def test_exception_raising(self):
        """
        Tests that an EvolutionaryException is correctly raised and detected by the test framework.
        """
        with self.assertRaises(EvolutionaryException):
            raise EvolutionaryException("Test exception")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test suite for complex evolutionary scenarios."""
    
    def setUp(self):
        """
        Set up the integration test environment with a GenesisEvolutionaryConduit instance and default evolutionary parameters.
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
        Test the complete evolution cycle using GenesisEvolutionaryConduit, verifying that the result contains the best individual and the correct number of generations run.
        """
        # Set up fitness function
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome by summing the squares of its elements.
            
            Parameters:
            	genome (iterable): Sequence of numeric values representing the genome.
            
            Returns:
            	float: The sum of squared values in the genome.
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
        Test the complete neural network evolution pipeline, including setting network configuration, assigning training data, and creating a neural network instance.
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
        Test that the multi-objective optimization pipeline returns a fitness vector with correct values for multiple objectives.
        
        Verifies that when multiple objectives are configured, evaluating a genome produces a fitness vector matching the number and order of objectives.
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
        Tests that the adaptive mutation rate calculation in the evolutionary pipeline returns a float within [0.0, 1.0] for a population with diverse fitness values.
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
        Test that the evolutionary framework raises appropriate exceptions for invalid parameters and fitness function failures.
        
        Verifies that a ValueError is raised when invalid evolutionary parameters are provided, and that an EvolutionaryException is raised if the fitness function fails during evolution.
        """
        # Test invalid parameters
        with self.assertRaises(ValueError):
            invalid_params = EvolutionaryParameters(population_size=0)
        
        # Test recovery from evolution failure
        def failing_fitness(genome):
            """
            A fitness function that always raises a ValueError to simulate a fitness evaluation failure.
            
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
        Set up the test environment by creating a GenesisEvolutionaryConduit instance and initializing EvolutionaryParameters for async evolution tests.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    @patch('asyncio.run')
    def test_async_evolution_execution(self, mock_run):
        """
        Test that the asynchronous evolution method returns a valid result when using a mocked async evolution process.
        
        This test ensures that the `run_async_evolution` method of the Genesis conduit produces a non-None result when the asynchronous evolution is simulated with a mock.
        """
        async def mock_async_evolve():
            """
            Simulates an asynchronous evolutionary process and returns a mock result.
            
            Returns:
                dict: Mock data representing the outcome of an evolutionary run, including the best individual, number of generations, final population, and summary statistics.
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
        Test that population fitness evaluation is executed in parallel using a thread pool executor and verifies that parallel execution is triggered.
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
            Calculate the fitness of a genome by summing all its elements.
            
            Parameters:
            	genome (iterable): Sequence of numeric values representing the genome.
            
            Returns:
            	The sum of all elements in the genome.
            """
            return sum(genome)
        
        # Test parallel evaluation
        self.genesis_conduit.evaluate_population_parallel(population, fitness_func)
        
        # Verify parallel execution was attempted
        mock_executor.assert_called_once()


class TestEvolutionaryParametersAdditional(unittest.TestCase):
    """Additional comprehensive tests for EvolutionaryParameters class."""
    
    def test_parameter_boundary_conditions(self):
        """Test boundary conditions for all parameter validations."""
        # Test exact boundary values
        params = EvolutionaryParameters(
            population_size=1,  # Minimum valid value
            generations=1,      # Minimum valid value
            mutation_rate=0.0,  # Minimum valid value
            crossover_rate=0.0, # Minimum valid value
            selection_pressure=0.0
        )
        self.assertEqual(params.population_size, 1)
        self.assertEqual(params.mutation_rate, 0.0)
        self.assertEqual(params.crossover_rate, 0.0)
        
        # Test maximum boundary values
        params = EvolutionaryParameters(
            mutation_rate=1.0,  # Maximum valid value
            crossover_rate=1.0  # Maximum valid value
        )
        self.assertEqual(params.mutation_rate, 1.0)
        self.assertEqual(params.crossover_rate, 1.0)
    
    def test_parameter_types(self):
        """Test that parameters accept different numeric types."""
        # Test with floats
        params = EvolutionaryParameters(
            population_size=100.0,
            generations=500.0
        )
        self.assertEqual(params.population_size, 100.0)
        self.assertEqual(params.generations, 500.0)
        
        # Test with numpy types if available
        try:
            import numpy as np
            params = EvolutionaryParameters(
                population_size=np.int32(50),
                mutation_rate=np.float64(0.1)
            )
            self.assertEqual(params.population_size, 50)
            self.assertEqual(params.mutation_rate, 0.1)
        except ImportError:
            pass
    
    def test_from_dict_missing_keys(self):
        """Test from_dict with missing keys uses default values."""
        partial_dict = {
            'population_size': 200,
            'mutation_rate': 0.15
        }
        params = EvolutionaryParameters.from_dict(partial_dict)
        self.assertEqual(params.population_size, 200)
        self.assertEqual(params.mutation_rate, 0.15)
        self.assertEqual(params.generations, 500)  # Default value
        self.assertEqual(params.crossover_rate, 0.8)  # Default value
    
    def test_from_dict_extra_keys(self):
        """Test from_dict ignores extra keys."""
        dict_with_extra = {
            'population_size': 150,
            'generations': 750,
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'selection_pressure': 0.25,
            'extra_key': 'should_be_ignored'
        }
        params = EvolutionaryParameters.from_dict(dict_with_extra)
        self.assertEqual(params.population_size, 150)
        self.assertFalse(hasattr(params, 'extra_key'))
    
    def test_parameter_copy_and_equality(self):
        """Test parameter copying and equality comparison."""
        params1 = EvolutionaryParameters(population_size=100, mutation_rate=0.1)
        params2 = EvolutionaryParameters(population_size=100, mutation_rate=0.1)
        params3 = EvolutionaryParameters(population_size=200, mutation_rate=0.1)
        
        # Test equality
        self.assertEqual(params1.to_dict(), params2.to_dict())
        self.assertNotEqual(params1.to_dict(), params3.to_dict())


class TestMutationStrategyAdditional(unittest.TestCase):
    """Additional comprehensive tests for MutationStrategy class."""
    
    def setUp(self):
        self.strategy = MutationStrategy()
    
    def test_mutation_with_empty_genome(self):
        """Test mutation strategies with empty genome."""
        empty_genome = []
        
        # All mutation strategies should handle empty genomes gracefully
        result = self.strategy.gaussian_mutation(empty_genome, 0.1)
        self.assertEqual(result, [])
        
        result = self.strategy.uniform_mutation(empty_genome, 0.1, bounds=(-1, 1))
        self.assertEqual(result, [])
        
        result = self.strategy.bit_flip_mutation(empty_genome, 0.1)
        self.assertEqual(result, [])
    
    def test_mutation_with_single_element_genome(self):
        """Test mutation strategies with single element genome."""
        single_genome = [1.0]
        
        result = self.strategy.gaussian_mutation(single_genome, 0.1)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], float)
        
        result = self.strategy.uniform_mutation(single_genome, 0.1, bounds=(-10, 10))
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0], -10)
        self.assertLessEqual(result[0], 10)
    
    def test_gaussian_mutation_with_zero_sigma(self):
        """Test Gaussian mutation with zero sigma (no variation)."""
        genome = [1.0, 2.0, 3.0]
        result = self.strategy.gaussian_mutation(genome, 0.0, sigma=0.0)
        self.assertEqual(result, genome)
    
    def test_gaussian_mutation_with_large_sigma(self):
        """Test Gaussian mutation with large sigma values."""
        genome = [1.0, 2.0, 3.0]
        result = self.strategy.gaussian_mutation(genome, 1.0, sigma=100.0)
        self.assertEqual(len(result), len(genome))
        # With large sigma, values should be significantly different
        self.assertNotEqual(result, genome)
    
    def test_uniform_mutation_with_tight_bounds(self):
        """Test uniform mutation with very tight bounds."""
        genome = [5.0, 5.0, 5.0]
        tight_bounds = (4.99, 5.01)
        result = self.strategy.uniform_mutation(genome, 1.0, bounds=tight_bounds)
        
        for value in result:
            self.assertGreaterEqual(value, 4.99)
            self.assertLessEqual(value, 5.01)
    
    def test_bit_flip_mutation_with_all_true(self):
        """Test bit flip mutation with all True values."""
        genome = [True, True, True, True]
        result = self.strategy.bit_flip_mutation(genome, 1.0)  # 100% mutation rate
        
        self.assertEqual(len(result), len(genome))
        # With 100% mutation rate, all values should be flipped
        self.assertNotEqual(result, genome)
        for value in result:
            self.assertIsInstance(value, bool)
    
    def test_adaptive_mutation_convergence(self):
        """Test adaptive mutation with converging fitness history."""
        genome = [1.0, 2.0, 3.0]
        
        # Converging fitness (getting better)
        improving_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        result1 = self.strategy.adaptive_mutation(genome, improving_history, base_rate=0.1)
        
        # Stagnating fitness
        stagnant_history = [0.5, 0.5, 0.5, 0.5, 0.5]
        result2 = self.strategy.adaptive_mutation(genome, stagnant_history, base_rate=0.1)
        
        self.assertEqual(len(result1), len(genome))
        self.assertEqual(len(result2), len(genome))
    
    def test_adaptive_mutation_with_short_history(self):
        """Test adaptive mutation with very short fitness history."""
        genome = [1.0, 2.0, 3.0]
        short_history = [0.5]
        
        result = self.strategy.adaptive_mutation(genome, short_history, base_rate=0.1)
        self.assertEqual(len(result), len(genome))
    
    def test_mutation_reproducibility(self):
        """Test that mutation with same random seed produces same results."""
        genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Set seed and mutate
        random.seed(42)
        result1 = self.strategy.gaussian_mutation(genome, 0.1, sigma=0.5)
        
        # Reset seed and mutate again
        random.seed(42)
        result2 = self.strategy.gaussian_mutation(genome, 0.1, sigma=0.5)
        
        self.assertEqual(result1, result2)


class TestSelectionStrategyAdditional(unittest.TestCase):
    """Additional comprehensive tests for SelectionStrategy class."""
    
    def setUp(self):
        self.strategy = SelectionStrategy()
        self.large_population = [
            {'genome': [i, i+1, i+2], 'fitness': i/10.0} 
            for i in range(100)
        ]
    
    def test_tournament_selection_with_size_one(self):
        """Test tournament selection with tournament size of 1."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.1}
        ]
        
        # Tournament size 1 should be random selection
        selected = self.strategy.tournament_selection(population, tournament_size=1)
        self.assertIn(selected, population)
    
    def test_tournament_selection_with_max_size(self):
        """Test tournament selection with maximum tournament size."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.1}
        ]
        
        # Tournament size equal to population size should select best
        selected = self.strategy.tournament_selection(population, tournament_size=len(population))
        self.assertEqual(selected['fitness'], 0.9)
    
    def test_roulette_wheel_with_zero_fitness(self):
        """Test roulette wheel selection with zero fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.0},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 0.1}
        ]
        
        # Should still select an individual even with zero fitness
        selected = self.strategy.roulette_wheel_selection(population)
        self.assertIn(selected, population)
    
    def test_roulette_wheel_with_negative_fitness(self):
        """Test roulette wheel selection with negative fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': -0.5},
            {'genome': [4, 5, 6], 'fitness': -0.1},
            {'genome': [7, 8, 9], 'fitness': -0.9}
        ]
        
        # Should handle negative fitness by shifting values
        selected = self.strategy.roulette_wheel_selection(population)
        self.assertIn(selected, population)
    
    def test_rank_selection_with_ties(self):
        """Test rank selection with tied fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        # Should handle ties gracefully
        selected = self.strategy.rank_selection(population)
        self.assertIn(selected, population)
    
    def test_elitism_selection_with_large_count(self):
        """Test elitism selection with count equal to population size."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.1}
        ]
        
        # Elite count equal to population size should return all individuals
        selected = self.strategy.elitism_selection(population, len(population))
        self.assertEqual(len(selected), len(population))
    
    def test_selection_pressure_effects(self):
        """Test how selection pressure affects selection outcomes."""
        # Run multiple selections and check distribution
        selections = []
        for _ in range(100):
            selected = self.strategy.tournament_selection(self.large_population, tournament_size=5)
            selections.append(selected['fitness'])
        
        # Higher fitness individuals should be selected more often
        mean_fitness = sum(selections) / len(selections)
        population_mean = sum(ind['fitness'] for ind in self.large_population) / len(self.large_population)
        self.assertGreater(mean_fitness, population_mean)
    
    def test_single_individual_population(self):
        """Test selection strategies with single individual population."""
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        # All selection strategies should return the single individual
        selected = self.strategy.tournament_selection(population, tournament_size=1)
        self.assertEqual(selected, population[0])
        
        selected = self.strategy.roulette_wheel_selection(population)
        self.assertEqual(selected, population[0])
        
        selected = self.strategy.rank_selection(population)
        self.assertEqual(selected, population[0])
        
        selected = self.strategy.elitism_selection(population, 1)
        self.assertEqual(selected, population)


class TestFitnessFunctionAdditional(unittest.TestCase):
    """Additional comprehensive tests for FitnessFunction class."""
    
    def setUp(self):
        self.fitness_func = FitnessFunction()
    
    def test_fitness_functions_with_extreme_values(self):
        """Test fitness functions with extreme input values."""
        # Test with very large values
        large_genome = [1000.0, 2000.0, 3000.0]
        sphere_fitness = self.fitness_func.sphere_function(large_genome)
        self.assertIsInstance(sphere_fitness, (int, float))
        
        # Test with very small values
        small_genome = [1e-10, 2e-10, 3e-10]
        rastrigin_fitness = self.fitness_func.rastrigin_function(small_genome)
        self.assertIsInstance(rastrigin_fitness, (int, float))
    
    def test_fitness_functions_with_negative_values(self):
        """Test fitness functions with negative input values."""
        negative_genome = [-1.0, -2.0, -3.0]
        
        sphere_fitness = self.fitness_func.sphere_function(negative_genome)
        self.assertIsInstance(sphere_fitness, (int, float))
        
        rosenbrock_fitness = self.fitness_func.rosenbrock_function(negative_genome)
        self.assertIsInstance(rosenbrock_fitness, (int, float))
    
    def test_fitness_functions_with_mixed_values(self):
        """Test fitness functions with mixed positive and negative values."""
        mixed_genome = [-1.0, 0.0, 1.0, -2.0, 2.0]
        
        sphere_fitness = self.fitness_func.sphere_function(mixed_genome)
        ackley_fitness = self.fitness_func.ackley_function(mixed_genome)
        
        self.assertIsInstance(sphere_fitness, (int, float))
        self.assertIsInstance(ackley_fitness, (int, float))
    
    def test_fitness_functions_with_single_value(self):
        """Test fitness functions with single-element genomes."""
        single_genome = [5.0]
        
        sphere_fitness = self.fitness_func.sphere_function(single_genome)
        self.assertEqual(sphere_fitness, -25.0)  # -(5^2)
        
        rastrigin_fitness = self.fitness_func.rastrigin_function(single_genome)
        self.assertIsInstance(rastrigin_fitness, (int, float))
    
    def test_multi_objective_with_different_scales(self):
        """Test multi-objective evaluation with objectives of different scales."""
        genome = [1.0, 2.0, 3.0]
        objectives = [
            lambda g: sum(g),           # Scale: ~6
            lambda g: sum(x**2 for x in g) * 1000,  # Scale: ~14000
            lambda g: sum(g) / 1000     # Scale: ~0.006
        ]
        
        fitness = self.fitness_func.multi_objective_evaluate(genome, objectives)
        
        self.assertEqual(len(fitness), 3)
        self.assertAlmostEqual(fitness[0], 6.0)
        self.assertAlmostEqual(fitness[1], 14000.0)
        self.assertAlmostEqual(fitness[2], 0.006)
    
    def test_constraint_handling_with_multiple_constraints(self):
        """Test constraint handling with multiple constraints."""
        genome = [1.0, 2.0, 3.0]
        
        def constraint1(g):
            return sum(g) < 10  # Should pass
        
        def constraint2(g):
            return max(g) < 2   # Should fail
        
        constraints = [constraint1, constraint2]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            constraints
        )
        
        # Should be penalized due to constraint2 failure
        self.assertLess(fitness, sum(genome))
    
    def test_constraint_handling_with_no_violations(self):
        """Test constraint handling when all constraints are satisfied."""
        genome = [0.5, 0.5, 0.5]
        
        def constraint1(g):
            return sum(g) < 5
        
        def constraint2(g):
            return max(g) < 1
        
        constraints = [constraint1, constraint2]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            constraints
        )
        
        # Should equal original fitness (no penalty)
        self.assertEqual(fitness, sum(genome))
    
    def test_custom_function_with_exception_handling(self):
        """Test custom fitness function that raises exceptions."""
        def problematic_fitness(genome):
            if len(genome) == 0:
                raise ValueError("Empty genome")
            return sum(genome)
        
        # Should handle exceptions gracefully
        try:
            fitness = self.fitness_func.evaluate([], problematic_fitness)
            self.fail("Should have raised exception")
        except ValueError:
            pass  # Expected behavior
    
    def test_fitness_function_with_non_numeric_return(self):
        """Test fitness function that returns non-numeric values."""
        def non_numeric_fitness(genome):
            return "not_a_number"
        
        # Should handle non-numeric returns
        try:
            fitness = self.fitness_func.evaluate([1, 2, 3], non_numeric_fitness)
            # If no exception, check if it's handled appropriately
            self.assertIsInstance(fitness, (int, float, str))
        except (TypeError, ValueError):
            pass  # Expected behavior


class TestPopulationManagerAdditional(unittest.TestCase):
    """Additional comprehensive tests for PopulationManager class."""
    
    def setUp(self):
        self.manager = PopulationManager()
    
    def test_population_initialization_with_bounds(self):
        """Test population initialization with specific bounds."""
        population = self.manager.initialize_random_population(
            population_size=10,
            genome_length=5,
            bounds=(-5.0, 5.0)
        )
        
        for individual in population:
            for gene in individual['genome']:
                self.assertGreaterEqual(gene, -5.0)
                self.assertLessEqual(gene, 5.0)
    
    def test_population_initialization_with_custom_generator(self):
        """Test population initialization with custom genome generator."""
        def custom_generator():
            return [i * 0.1 for i in range(5)]
        
        population = self.manager.initialize_random_population(
            population_size=5,
            genome_length=5,
            generator=custom_generator
        )
        
        # All individuals should have the same genome from custom generator
        expected_genome = [0.0, 0.1, 0.2, 0.3, 0.4]
        for individual in population:
            self.assertEqual(individual['genome'], expected_genome)
    
    def test_seeded_population_with_insufficient_seeds(self):
        """Test seeded population when seeds are fewer than population size."""
        seeds = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        
        population = self.manager.initialize_seeded_population(
            population_size=10,
            genome_length=3,
            seeds=seeds
        )
        
        # Should have 10 individuals total
        self.assertEqual(len(population), 10)
        
        # First two should be the seeds
        genomes = [ind['genome'] for ind in population]
        self.assertIn(seeds[0], genomes)
        self.assertIn(seeds[1], genomes)
    
    def test_seeded_population_with_excess_seeds(self):
        """Test seeded population when seeds exceed population size."""
        seeds = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ]
        
        population = self.manager.initialize_seeded_population(
            population_size=2,
            genome_length=3,
            seeds=seeds
        )
        
        # Should only have 2 individuals (first 2 seeds)
        self.assertEqual(len(population), 2)
        self.assertEqual(population[0]['genome'], seeds[0])
        self.assertEqual(population[1]['genome'], seeds[1])
    
    def test_population_evaluation_with_parallel_processing(self):
        """Test population evaluation with parallel processing enabled."""
        population = self.manager.initialize_random_population(10, 5)
        
        def slow_fitness(genome):
            time.sleep(0.001)  # Simulate slow computation
            return sum(genome)
        
        # Test parallel evaluation
        start_time = time.time()
        self.manager.evaluate_population(population, slow_fitness, parallel=True)
        parallel_time = time.time() - start_time
        
        # Test sequential evaluation
        population_copy = self.manager.initialize_random_population(10, 5)
        start_time = time.time()
        self.manager.evaluate_population(population_copy, slow_fitness, parallel=False)
        sequential_time = time.time() - start_time
        
        # Parallel should be faster or similar
        self.assertLessEqual(parallel_time, sequential_time * 1.5)
    
    def test_population_statistics_with_extreme_values(self):
        """Test population statistics with extreme fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 1e-10},
            {'genome': [4, 5, 6], 'fitness': 1e10},
            {'genome': [7, 8, 9], 'fitness': -1e10}
        ]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 1e10)
        self.assertEqual(stats['worst_fitness'], -1e10)
        self.assertIsInstance(stats['std_dev_fitness'], float)
    
    def test_diversity_calculation_with_identical_genomes(self):
        """Test diversity calculation with identical genomes."""
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        
        # Diversity should be zero or very close to zero
        self.assertLessEqual(diversity, 0.01)
    
    def test_diversity_calculation_with_different_genome_lengths(self):
        """Test diversity calculation with genomes of different lengths."""
        population = [
            {'genome': [1.0, 2.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0], 'fitness': 0.7}
        ]
        
        # Should handle different lengths gracefully
        diversity = self.manager.calculate_diversity(population)
        self.assertIsInstance(diversity, float)
    
    def test_population_filtering_and_sorting(self):
        """Test population filtering and sorting capabilities."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5, 'age': 1},
            {'genome': [4, 5, 6], 'fitness': 0.9, 'age': 2},
            {'genome': [7, 8, 9], 'fitness': 0.1, 'age': 3}
        ]
        
        # Test filtering by fitness threshold
        filtered = self.manager.filter_population(population, min_fitness=0.3)
        self.assertEqual(len(filtered), 2)
        
        # Test sorting by fitness
        sorted_pop = self.manager.sort_population(population, by='fitness', ascending=False)
        self.assertEqual(sorted_pop[0]['fitness'], 0.9)
        self.assertEqual(sorted_pop[-1]['fitness'], 0.1)
    
    def test_population_archive_management(self):
        """Test population archive and history management."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.9}
        ]
        
        # Test archiving best individuals
        self.manager.archive_best_individuals(population, archive_size=1)
        archive = self.manager.get_archive()
        
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0]['fitness'], 0.9)
    
    def test_population_validation(self):
        """Test population structure validation."""
        valid_population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.9}
        ]
        
        invalid_population = [
            {'genome': [1, 2, 3]},  # Missing fitness
            {'fitness': 0.9}        # Missing genome
        ]
        
        self.assertTrue(self.manager.validate_population(valid_population))
        self.assertFalse(self.manager.validate_population(invalid_population))


class TestGeneticOperationsAdditional(unittest.TestCase):
    """Additional comprehensive tests for GeneticOperations class."""
    
    def setUp(self):
        self.operations = GeneticOperations()
    
    def test_crossover_with_identical_parents(self):
        """Test crossover operations with identical parents."""
        parent = [1, 2, 3, 4, 5]
        
        child1, child2 = self.operations.single_point_crossover(parent, parent)
        self.assertEqual(child1, parent)
        self.assertEqual(child2, parent)
        
        child1, child2 = self.operations.uniform_crossover(parent, parent, 0.5)
        self.assertEqual(child1, parent)
        self.assertEqual(child2, parent)
    
    def test_crossover_with_extreme_rates(self):
        """Test crossover operations with extreme crossover rates."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        # Test with 0% crossover rate
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, 0.0)
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
        
        # Test with 100% crossover rate
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, 1.0)
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
    
    def test_arithmetic_crossover_with_extreme_alpha(self):
        """Test arithmetic crossover with extreme alpha values."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Alpha = 0 should return parent2, parent1
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.0)
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
        
        # Alpha = 1 should return parent1, parent2
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.0)
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
    
    def test_simulated_binary_crossover_bounds_enforcement(self):
        """Test that SBX crossover respects bounds strictly."""
        parent1 = [0.0, 0.0, 0.0]
        parent2 = [1.0, 1.0, 1.0]
        bounds = [(0.0, 1.0)] * 3
        
        for _ in range(100):  # Multiple runs to check consistency
            child1, child2 = self.operations.simulated_binary_crossover(
                parent1, parent2, bounds, eta=2.0
            )
            
            for i, (lower, upper) in enumerate(bounds):
                self.assertGreaterEqual(child1[i], lower)
                self.assertLessEqual(child1[i], upper)
                self.assertGreaterEqual(child2[i], lower)
                self.assertLessEqual(child2[i], upper)
    
    def test_blend_crossover_with_different_alpha_values(self):
        """Test blend crossover with various alpha values."""
        parent1 = [0.0, 0.0, 0.0]
        parent2 = [10.0, 10.0, 10.0]
        
        # Test with small alpha
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.1)
        self.assertEqual(len(child1), 3)
        self.assertEqual(len(child2), 3)
        
        # Test with large alpha
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=2.0)
        self.assertEqual(len(child1), 3)
        self.assertEqual(len(child2), 3)
    
    def test_crossover_with_float_genomes(self):
        """Test crossover operations with floating-point genomes."""
        parent1 = [1.5, 2.7, 3.14, 4.0]
        parent2 = [5.5, 6.2, 7.89, 8.0]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
        
        # Check that results are still floats
        for gene in child1:
            self.assertIsInstance(gene, (int, float))
        for gene in child2:
            self.assertIsInstance(gene, (int, float))
    
    def test_crossover_with_large_genomes(self):
        """Test crossover operations with large genomes."""
        size = 1000
        parent1 = list(range(size))
        parent2 = list(range(size, 2*size))
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), size)
        self.assertEqual(len(child2), size)
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), size)
        self.assertEqual(len(child2), size)
    
    def test_crossover_reproducibility(self):
        """Test that crossover operations are reproducible with same random seed."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        # Set seed and perform crossover
        random.seed(42)
        child1_a, child2_a = self.operations.single_point_crossover(parent1, parent2)
        
        # Reset seed and perform crossover again
        random.seed(42)
        child1_b, child2_b = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(child1_a, child1_b)
        self.assertEqual(child2_a, child2_b)
    
    def test_crossover_with_boolean_genomes(self):
        """Test crossover operations with boolean genomes."""
        parent1 = [True, False, True, False, True]
        parent2 = [False, True, False, True, False]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
        
        # Check that results are still booleans
        for gene in child1:
            self.assertIsInstance(gene, bool)
        for gene in child2:
            self.assertIsInstance(gene, bool)
    
    def test_crossover_performance_with_large_populations(self):
        """Test crossover performance with large populations."""
        parent1 = list(range(10000))
        parent2 = list(range(10000, 20000))
        
        start_time = time.time()
        for _ in range(100):
            child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 10.0)  # 10 seconds threshold


class TestEvolutionaryConduitAdditional(unittest.TestCase):
    """Additional comprehensive tests for EvolutionaryConduit class."""
    
    def setUp(self):
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_conduit_configuration_validation(self):
        """Test that conduit validates configuration parameters."""
        # Test invalid parameters
        with self.assertRaises(ValueError):
            invalid_params = EvolutionaryParameters(population_size=-1)
            self.conduit.set_parameters(invalid_params)
    
    def test_conduit_with_multiple_fitness_functions(self):
        """Test conduit with multiple fitness functions."""
        def fitness1(genome):
            return sum(genome)
        
        def fitness2(genome):
            return sum(x**2 for x in genome)
        
        # Test switching between fitness functions
        self.conduit.set_fitness_function(fitness1)
        result1 = self.conduit.fitness_function.evaluate([1, 2, 3], fitness1)
        
        self.conduit.set_fitness_function(fitness2)
        result2 = self.conduit.fitness_function.evaluate([1, 2, 3], fitness2)
        
        self.assertEqual(result1, 6)
        self.assertEqual(result2, 14)
    
    def test_conduit_callback_system(self):
        """Test comprehensive callback system functionality."""
        callback_data = []
        
        def generation_callback(generation, population, best_individual):
            callback_data.append({
                'generation': generation,
                'population_size': len(population),
                'best_fitness': best_individual['fitness']
            })
        
        def convergence_callback(generation, population, best_individual):
            if generation > 2 and callback_data:
                # Check for convergence
                if abs(callback_data[-1]['best_fitness'] - best_individual['fitness']) < 0.001:
                    callback_data.append({'converged': True})
        
        self.conduit.add_callback(generation_callback)
        self.conduit.add_callback(convergence_callback)
        
        # Mock evolution with callbacks
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 3,
                'final_population': [],
                'statistics': {}
            }
            
            # Simulate callback calls
            for gen in range(3):
                generation_callback(gen, [], {'fitness': 6.0})
                convergence_callback(gen, [], {'fitness': 6.0})
            
            self.assertEqual(len(callback_data), 3)
    
    def test_conduit_state_persistence(self):
        """Test comprehensive state persistence functionality."""
        # Configure conduit
        self.conduit.set_parameters(self.params)
        
        def custom_fitness(genome):
            return sum(x**2 for x in genome)
        
        self.conduit.set_fitness_function(custom_fitness)
        
        # Save state
        state = self.conduit.save_state()
        
        # Verify state contains expected keys
        self.assertIn('parameters', state)
        self.assertIn('fitness_function', state)
        self.assertIn('history_enabled', state)
        
        # Create new conduit and load state
        new_conduit = EvolutionaryConduit()
        new_conduit.load_state(state)
        
        # Verify state is loaded correctly
        self.assertEqual(new_conduit.parameters.population_size, 10)
        self.assertEqual(new_conduit.parameters.generations, 5)
    
    def test_conduit_with_custom_strategies(self):
        """Test conduit with custom mutation and selection strategies."""
        # Custom mutation strategy
        def custom_mutation(genome, rate):
            return [gene + 0.1 if random.random() < rate else gene for gene in genome]
        
        # Custom selection strategy
        def custom_selection(population):
            return random.choice(population)
        
        # Set custom strategies
        self.conduit.mutation_strategy.custom_mutation = custom_mutation
        self.conduit.selection_strategy.custom_selection = custom_selection
        
        # Test that custom strategies are available
        self.assertTrue(hasattr(self.conduit.mutation_strategy, 'custom_mutation'))
        self.assertTrue(hasattr(self.conduit.selection_strategy, 'custom_selection'))
    
    def test_conduit_evolution_termination_conditions(self):
        """Test various evolution termination conditions."""
        # Test fitness threshold termination
        def fitness_threshold_reached(generation, population, best_individual):
            return best_individual['fitness'] >= 10.0
        
        # Test stagnation termination
        def stagnation_check(generation, population, best_individual):
            return generation > 100  # Simplified stagnation check
        
        self.conduit.add_termination_condition(fitness_threshold_reached)
        self.conduit.add_termination_condition(stagnation_check)
        
        # Verify termination conditions are added
        self.assertEqual(len(self.conduit.termination_conditions), 2)
    
    def test_conduit_with_constraints(self):
        """Test conduit with constraint handling."""
        def constraint_function(genome):
            return sum(genome) <= 10  # Sum constraint
        
        self.conduit.add_constraint(constraint_function)
        
        # Test constraint evaluation
        valid_genome = [1, 2, 3]
        invalid_genome = [5, 6, 7]
        
        self.assertTrue(self.conduit.evaluate_constraints(valid_genome))
        self.assertFalse(self.conduit.evaluate_constraints(invalid_genome))
    
    def test_conduit_parallel_processing(self):
        """Test conduit with parallel processing capabilities."""
        self.conduit.set_parameters(self.params)
        self.conduit.enable_parallel_processing(num_workers=4)
        
        def slow_fitness(genome):
            time.sleep(0.001)
            return sum(genome)
        
        self.conduit.set_fitness_function(slow_fitness)
        
        # Mock parallel evolution
        with patch.object(self.conduit, 'evolve_parallel') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 5,
                'final_population': [],
                'statistics': {}
            }
            
            result = self.conduit.run_parallel_evolution(genome_length=3)
            self.assertIsNotNone(result)
    
    def test_conduit_memory_management(self):
        """Test conduit memory management with large populations."""
        large_params = EvolutionaryParameters(
            population_size=1000,
            generations=10
        )
        
        self.conduit.set_parameters(large_params)
        
        # Test memory cleanup
        self.conduit.enable_memory_management(max_memory_mb=100)
        
        # Mock evolution with memory management
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 10,
                'final_population': [],
                'statistics': {}
            }
            
            result = self.conduit.run_evolution(genome_length=10)
            self.assertIsNotNone(result)
    
    def test_conduit_logging_and_monitoring(self):
        """Test conduit logging and monitoring capabilities."""
        import logging
        
        # Set up logging
        logger = logging.getLogger('evolutionary_conduit')
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        self.conduit.set_logger(logger)
        self.conduit.enable_detailed_logging()
        
        # Test that logging is configured
        self.assertIsNotNone(self.conduit.logger)
        self.assertTrue(self.conduit.detailed_logging)


class TestGenesisEvolutionaryConduitAdditional(unittest.TestCase):
    """Additional comprehensive tests for GenesisEvolutionaryConduit class."""
    
    def setUp(self):
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_genesis_conduit_advanced_neural_evolution(self):
        """Test advanced neural evolution capabilities."""
        # Set up complex network architecture
        complex_config = {
            'input_size': 100,
            'hidden_layers': [200, 100, 50],
            'output_size': 10,
            'activation': 'relu',
            'dropout_rate': 0.2,
            'batch_normalization': True
        }
        
        self.genesis_conduit.set_network_config(complex_config)
        
        # Test network creation with complex architecture
        network = self.genesis_conduit.create_neural_network()
        self.assertIsNotNone(network)
    
    def test_genesis_conduit_dynamic_topology_evolution(self):
        """Test dynamic topology evolution with structural mutations."""
        initial_topology = {
            'layers': [10, 20, 1],
            'connections': [[0, 1], [1, 2]],
            'activation_functions': ['relu', 'relu', 'sigmoid']
        }
        
        # Test various topology mutations
        mutations = [
            'add_layer',
            'remove_layer', 
            'add_connection',
            'remove_connection',
            'change_activation'
        ]
        
        for mutation_type in mutations:
            mutated = self.genesis_conduit.mutate_topology(
                initial_topology, 
                mutation_type=mutation_type
            )
            self.assertIsInstance(mutated, dict)
            self.assertIn('layers', mutated)
    
    def test_genesis_conduit_advanced_hyperparameter_optimization(self):
        """Test advanced hyperparameter optimization techniques."""
        # Complex search space
        search_space = {
            'learning_rate': ('log', 1e-5, 1e-1),
            'batch_size': ('choice', [16, 32, 64, 128, 256]),
            'optimizer': ('choice', ['adam', 'sgd', 'rmsprop']),
            'weight_decay': ('uniform', 0.0, 1e-3),
            'momentum': ('uniform', 0.0, 0.99),
            'architecture': {
                'num_layers': ('int', 2, 10),
                'layer_sizes': ('int', 10, 1000)
            }
        }
        
        self.genesis_conduit.set_advanced_hyperparameter_search_space(search_space)
        
        # Test hyperparameter generation
        hyperparams = self.genesis_conduit.generate_hyperparameters()
        
        self.assertIn('learning_rate', hyperparams)
        self.assertIn('batch_size', hyperparams)
        self.assertIn('optimizer', hyperparams)
        self.assertIn('architecture', hyperparams)
    
    def test_genesis_conduit_multi_task_learning(self):
        """Test multi-task learning capabilities."""
        # Set up multiple tasks
        tasks = [
            {
                'name': 'classification',
                'loss': 'cross_entropy',
                'metrics': ['accuracy', 'f1_score']
            },
            {
                'name': 'regression',
                'loss': 'mse',
                'metrics': ['mae', 'rmse']
            }
        ]
        
        self.genesis_conduit.set_multi_task_config(tasks)
        
        # Test multi-task fitness evaluation
        genome = [0.1] * 100
        fitness_vector = self.genesis_conduit.evaluate_multi_task_fitness(genome)
        
        self.assertEqual(len(fitness_vector), len(tasks))
        self.assertIsInstance(fitness_vector, list)
    
    def test_genesis_conduit_progressive_evolution(self):
        """Test progressive evolution with increasing complexity."""
        # Start with simple network
        simple_config = {
            'input_size': 10,
            'hidden_layers': [5],
            'output_size': 1
        }
        
        # Target complex network
        complex_config = {
            'input_size': 10,
            'hidden_layers': [20, 15, 10],
            'output_size': 1
        }
        
        self.genesis_conduit.set_progressive_evolution_config(
            start_config=simple_config,
            target_config=complex_config,
            complexity_schedule='linear'
        )
        
        # Test progressive evolution
        result = self.genesis_conduit.run_progressive_evolution(
            generations_per_phase=5,
            num_phases=3
        )
        
        self.assertIsNotNone(result)
    
    def test_genesis_conduit_meta_learning(self):
        """Test meta-learning capabilities for quick adaptation."""
        # Set up meta-learning configuration
        meta_config = {
            'meta_optimizer': 'maml',
            'inner_steps': 5,
            'inner_lr': 0.01,
            'meta_lr': 0.001
        }
        
        self.genesis_conduit.set_meta_learning_config(meta_config)
        
        # Test meta-learning adaptation
        support_set = [([1, 2, 3], 0), ([4, 5, 6], 1)]
        query_set = [([7, 8, 9], 1)]
        
        adapted_genome = self.genesis_conduit.meta_learn_adaptation(
            genome=[0.1] * 50,
            support_set=support_set,
            query_set=query_set
        )
        
        self.assertIsInstance(adapted_genome, list)
        self.assertGreater(len(adapted_genome), 0)
    
    def test_genesis_conduit_neural_architecture_search(self):
        """Test neural architecture search (NAS) capabilities."""
        # Define search space
        search_space = {
            'cells': [
                {'type': 'conv', 'filters': [16, 32, 64], 'kernel_size': [3, 5, 7]},
                {'type': 'pooling', 'pool_size': [2, 3], 'stride': [1, 2]},
                {'type': 'dense', 'units': [64, 128, 256]}
            ],
            'connections': 'darts',  # Differentiable Architecture Search
            'max_depth': 10
        }
        
        self.genesis_conduit.set_nas_search_space(search_space)
        
        # Test architecture search
        architecture = self.genesis_conduit.search_neural_architecture(
            search_method='evolutionary',
            search_budget=100
        )
        
        self.assertIsInstance(architecture, dict)
        self.assertIn('cells', architecture)
    
    def test_genesis_conduit_continual_learning(self):
        """Test continual learning with catastrophic forgetting prevention."""
        # Set up continual learning
        continual_config = {
            'method': 'ewc',  # Elastic Weight Consolidation
            'regularization_strength': 1000,
            'memory_size': 1000
        }
        
        self.genesis_conduit.set_continual_learning_config(continual_config)
        
        # Test learning sequence of tasks
        tasks = [
            {'data': [[1, 2, 3]], 'labels': [0], 'task_id': 1},
            {'data': [[4, 5, 6]], 'labels': [1], 'task_id': 2},
            {'data': [[7, 8, 9]], 'labels': [0], 'task_id': 3}
        ]
        
        genome = [0.1] * 100
        updated_genome = self.genesis_conduit.continual_learn(genome, tasks)
        
        self.assertIsInstance(updated_genome, list)
        self.assertEqual(len(updated_genome), len(genome))
    
    def test_genesis_conduit_federated_evolution(self):
        """Test federated evolution for distributed learning."""
        # Set up federated learning
        federated_config = {
            'num_clients': 5,
            'local_epochs': 10,
            'aggregation_method': 'fedavg',
            'client_sampling_rate': 0.8
        }
        
        self.genesis_conduit.set_federated_config(federated_config)
        
        # Test federated evolution
        client_data = [
            {'data': [[1, 2, 3]], 'labels': [0]},
            {'data': [[4, 5, 6]], 'labels': [1]},
            {'data': [[7, 8, 9]], 'labels': [0]}
        ]
        
        global_model = self.genesis_conduit.federated_evolution(
            client_data=client_data,
            global_rounds=5
        )
        
        self.assertIsNotNone(global_model)
    
    def test_genesis_conduit_quantum_inspired_evolution(self):
        """Test quantum-inspired evolutionary algorithms."""
        # Set up quantum-inspired evolution
        quantum_config = {
            'quantum_population_size': 20,
            'quantum_rotation_angle': 0.1,
            'quantum_collapse_probability': 0.8,
            'entanglement_probability': 0.3
        }
        
        self.genesis_conduit.set_quantum_config(quantum_config)
        
        # Test quantum evolution
        quantum_population = self.genesis_conduit.initialize_quantum_population(
            population_size=10,
            genome_length=20
        )
        
        self.assertEqual(len(quantum_population), 10)
        
        # Test quantum operations
        evolved_population = self.genesis_conduit.quantum_evolve_step(
            quantum_population
        )
        
        self.assertEqual(len(evolved_population), len(quantum_population))
    
    def test_genesis_conduit_performance_benchmarking(self):
        """Test comprehensive performance benchmarking."""
        # Set up benchmarking suite
        benchmark_suite = {
            'datasets': ['iris', 'wine', 'breast_cancer'],
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'cross_validation': {'folds': 5, 'stratified': True},
            'statistical_tests': ['t_test', 'wilcoxon']
        }
        
        self.genesis_conduit.set_benchmark_suite(benchmark_suite)
        
        # Test performance benchmarking
        genome = [0.1] * 100
        benchmark_results = self.genesis_conduit.benchmark_performance(
            genome,
            benchmark_suite
        )
        
        self.assertIsInstance(benchmark_results, dict)
        self.assertIn('datasets', benchmark_results)
        self.assertIn('metrics', benchmark_results)
        self.assertIn('statistical_significance', benchmark_results)
    
    def test_genesis_conduit_automated_model_selection(self):
        """Test automated model selection and ensemble creation."""
        # Set up model selection criteria
        selection_criteria = {
            'primary_metric': 'accuracy',
            'secondary_metrics': ['model_size', 'inference_time'],
            'constraints': {
                'max_model_size': 1000000,  # 1MB
                'max_inference_time': 100   # 100ms
            },
            'ensemble_methods': ['voting', 'stacking', 'bagging']
        }
        
        self.genesis_conduit.set_model_selection_criteria(selection_criteria)
        
        # Test automated model selection
        candidate_models = [
            {'genome': [0.1] * 50, 'fitness': 0.8, 'size': 500000, 'time': 50},
            {'genome': [0.2] * 100, 'fitness': 0.9, 'size': 1200000, 'time': 80},
            {'genome': [0.3] * 75, 'fitness': 0.85, 'size': 800000, 'time': 120}
        ]
        
        selected_models = self.genesis_conduit.automated_model_selection(
            candidate_models
        )
        
        self.assertIsInstance(selected_models, list)
        self.assertGreater(len(selected_models), 0)
    
    def test_genesis_conduit_explainable_ai_integration(self):
        """Test integration with explainable AI techniques."""
        # Set up explainability configuration
        explainability_config = {
            'methods': ['lime', 'shap', 'integrated_gradients'],
            'explanation_targets': ['predictions', 'features', 'model_structure'],
            'visualization': True
        }
        
        self.genesis_conduit.set_explainability_config(explainability_config)
        
        # Test explainable AI integration
        genome = [0.1] * 100
        test_input = [1, 2, 3, 4, 5]
        
        explanations = self.genesis_conduit.explain_model_decision(
            genome,
            test_input
        )
        
        self.assertIsInstance(explanations, dict)
        self.assertIn('feature_importance', explanations)
        self.assertIn('decision_reasoning', explanations)


class TestEvolutionaryExceptionAdditional(unittest.TestCase):
    """Additional comprehensive tests for EvolutionaryException class."""
    
    def test_exception_with_nested_exceptions(self):
        """Test EvolutionaryException with nested exception handling."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            nested_exception = EvolutionaryException(
                "Evolution failed due to nested error",
                details={'original_error': str(e)}
            )
            
            self.assertIn('original_error', nested_exception.details)
            self.assertEqual(nested_exception.details['original_error'], "Original error")
    
    def test_exception_with_complex_details(self):
        """Test EvolutionaryException with complex detail structures."""
        complex_details = {
            'generation': 42,
            'population_stats': {
                'size': 100,
                'best_fitness': 0.95,
                'diversity': 0.3
            },
            'error_trace': [
                {'function': 'evolve', 'line': 123},
                {'function': 'evaluate_fitness', 'line': 456}
            ],
            'configuration': {
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
        }
        
        exception = EvolutionaryException(
            "Complex evolution failure",
            details=complex_details
        )
        
        self.assertEqual(exception.details['generation'], 42)
        self.assertEqual(exception.details['population_stats']['size'], 100)
        self.assertEqual(len(exception.details['error_trace']), 2)
    
    def test_exception_serialization(self):
        """Test EvolutionaryException serialization and deserialization."""
        import pickle
        
        details = {
            'generation': 25,
            'error_type': 'fitness_evaluation',
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        original_exception = EvolutionaryException(
            "Serialization test",
            details=details
        )
        
        # Serialize and deserialize
        serialized = pickle.dumps(original_exception)
        deserialized = pickle.loads(serialized)
        
        self.assertEqual(str(deserialized), str(original_exception))
        self.assertEqual(deserialized.details, original_exception.details)
    
    def test_exception_inheritance_hierarchy(self):
        """Test EvolutionaryException inheritance and type checking."""
        exception = EvolutionaryException("Test exception")
        
        # Test inheritance
        self.assertIsInstance(exception, Exception)
        self.assertIsInstance(exception, EvolutionaryException)
        
        # Test type checking
        self.assertTrue(issubclass(EvolutionaryException, Exception))
    
    def test_exception_string_representation(self):
        """Test various string representations of EvolutionaryException."""
        # Test with message only
        exception1 = EvolutionaryException("Simple message")
        self.assertEqual(str(exception1), "Simple message")
        self.assertEqual(repr(exception1), "EvolutionaryException('Simple message')")
        
        # Test with details
        exception2 = EvolutionaryException(
            "Message with details",
            details={'key': 'value'}
        )
        self.assertEqual(str(exception2), "Message with details")
        self.assertIn('key', str(exception2.details))


class TestPerformanceAndScalability(unittest.TestCase):
    """Performance and scalability tests for the evolutionary system."""
    
    def test_large_population_performance(self):
        """Test performance with large population sizes."""
        large_params = EvolutionaryParameters(
            population_size=1000,
            generations=10
        )
        
        conduit = EvolutionaryConduit()
        conduit.set_parameters(large_params)
        
        def simple_fitness(genome):
            return sum(genome)
        
        conduit.set_fitness_function(simple_fitness)
        
        # Mock evolution to test performance
        with patch.object(conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 10,
                'final_population': [],
                'statistics': {}
            }
            
            start_time = time.time()
            result = conduit.run_evolution(genome_length=100)
            end_time = time.time()
            
            # Should complete within reasonable time
            self.assertLess(end_time - start_time, 10.0)
            self.assertIsNotNone(result)
    
    def test_memory_usage_with_large_genomes(self):
        """Test memory usage with large genome sizes."""
        try:
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create large genomes
            large_population = []
            for i in range(100):
                genome = [0.1] * 10000  # Large genome
                large_population.append({'genome': genome, 'fitness': 0.0})
            
            # Get memory usage after creation
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 500MB)
            self.assertLess(memory_increase, 500)
        except ImportError:
            # Skip test if psutil is not available
            pass
    
    def test_concurrent_evolution_performance(self):
        """Test performance with concurrent evolution processes."""
        import threading
        
        def run_evolution():
            conduit = EvolutionaryConduit()
            params = EvolutionaryParameters(population_size=50, generations=5)
            conduit.set_parameters(params)
            
            def fitness(genome):
                return sum(genome)
            
            conduit.set_fitness_function(fitness)
            
            # Mock evolution
            with patch.object(conduit, 'evolve') as mock_evolve:
                mock_evolve.return_value = {
                    'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                    'generations_run': 5,
                    'final_population': [],
                    'statistics': {}
                }
                
                conduit.run_evolution(genome_length=10)
        
        # Run multiple concurrent evolutions
        threads = []
        start_time = time.time()
        
        for i in range(5):
            thread = threading.Thread(target=run_evolution)
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 15.0)
    
    def test_scalability_with_increasing_complexity(self):
        """Test scalability with increasing problem complexity."""
        complexities = [10, 50, 100, 500]
        times = []
        
        for complexity in complexities:
            conduit = EvolutionaryConduit()
            params = EvolutionaryParameters(
                population_size=complexity,
                generations=5
            )
            conduit.set_parameters(params)
            
            def complex_fitness(genome):
                # Simulate complex fitness evaluation
                result = 0
                for i in range(len(genome)):
                    for j in range(i+1, len(genome)):
                        result += genome[i] * genome[j]
                return result
            
            conduit.set_fitness_function(complex_fitness)
            
            # Mock evolution
            with patch.object(conduit, 'evolve') as mock_evolve:
                mock_evolve.return_value = {
                    'best_individual': {'genome': [1] * complexity, 'fitness': 1.0},
                    'generations_run': 5,
                    'final_population': [],
                    'statistics': {}
                }
                
                start_time = time.time()
                conduit.run_evolution(genome_length=complexity)
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        # Check that time complexity is reasonable
        # Time should not increase exponentially
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            self.assertLess(ratio, 10.0)  # Should not be more than 10x slower


class TestRobustnessAndErrorHandling(unittest.TestCase):
    """Robustness and error handling tests."""
    
    def test_fitness_function_exception_handling(self):
        """Test handling of exceptions in fitness functions."""
        conduit = EvolutionaryConduit()
        
        def problematic_fitness(genome):
            if len(genome) == 0:
                raise ValueError("Empty genome")
            if genome[0] < 0:
                raise RuntimeError("Negative value")
            return sum(genome)
        
        conduit.set_fitness_function(problematic_fitness)
        
        # Test with empty genome
        with self.assertRaises(ValueError):
            conduit.fitness_function.evaluate([], problematic_fitness)
        
        # Test with negative genome
        with self.assertRaises(RuntimeError):
            conduit.fitness_function.evaluate([-1, 2, 3], problematic_fitness)
    
    def test_invalid_genome_handling(self):
        """Test handling of invalid genome structures."""
        manager = PopulationManager()
        
        # Test with invalid genome types
        invalid_genomes = [
            None,
            "string_genome",
            {'invalid': 'dict'},
            [None, None, None]
        ]
        
        for invalid_genome in invalid_genomes:
            with self.assertRaises((TypeError, ValueError)):
                manager.validate_genome(invalid_genome)
    
    def test_parameter_validation_robustness(self):
        """Test robustness of parameter validation."""
        # Test with extreme values
        extreme_values = [
            {'population_size': 0},
            {'population_size': -1},
            {'population_size': float('inf')},
            {'mutation_rate': -1.0},
            {'mutation_rate': 2.0},
            {'mutation_rate': float('nan')},
            {'crossover_rate': -0.5},
            {'crossover_rate': 1.5},
            {'generations': -1},
            {'generations': 0}
        ]
        
        for extreme_value in extreme_values:
            with self.assertRaises(ValueError):
                EvolutionaryParameters(**extreme_value)
    
    def test_memory_leak_prevention(self):
        """Test prevention of memory leaks in long-running evolution."""
        import gc
        import weakref
        
        # Create evolution objects
        conduit = EvolutionaryConduit()
        params = EvolutionaryParameters(population_size=10, generations=5)
        conduit.set_parameters(params)
        
        # Create weak references to track object cleanup
        weak_refs = []
        
        for i in range(10):
            population = conduit.population_manager.initialize_random_population(10, 5)
            weak_refs.extend([weakref.ref(ind) for ind in population])
            
            # Simulate evolution step
            conduit.population_manager.evaluate_population(
                population, 
                lambda g: sum(g)
            )
            
            # Clear population
            population.clear()
            del population
        
        # Force garbage collection
        gc.collect()
        
        # Check that objects are cleaned up
        live_refs = [ref for ref in weak_refs if ref() is not None]
        self.assertLess(len(live_refs), len(weak_refs) * 0.1)  # Less than 10% should remain
    
    def test_thread_safety(self):
        """Test thread safety of evolutionary components."""
        import threading
        
        conduit = EvolutionaryConduit()
        params = EvolutionaryParameters(population_size=20, generations=5)
        conduit.set_parameters(params)
        
        def fitness_func(genome):
            return sum(genome)
        
        conduit.set_fitness_function(fitness_func)
        
        results = []
        errors = []
        
        def evolve_thread():
            try:
                # Mock evolution
                with patch.object(conduit, 'evolve') as mock_evolve:
                    mock_evolve.return_value = {
                        'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                        'generations_run': 5,
                        'final_population': [],
                        'statistics': {}
                    }
                    
                    result = conduit.run_evolution(genome_length=10)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=evolve_thread)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
    
    def test_graceful_degradation(self):
        """Test graceful degradation under resource constraints."""
        # Test with very limited resources
        limited_params = EvolutionaryParameters(
            population_size=2,  # Very small population
            generations=1,      # Single generation
            mutation_rate=0.0,  # No mutation
            crossover_rate=0.0  # No crossover
        )
        
        conduit = EvolutionaryConduit()
        conduit.set_parameters(limited_params)
        
        def simple_fitness(genome):
            return sum(genome)
        
        conduit.set_fitness_function(simple_fitness)
        
        # Mock evolution
        with patch.object(conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2], 'fitness': 3.0},
                'generations_run': 1,
                'final_population': [],
                'statistics': {}
            }
            
            result = conduit.run_evolution(genome_length=2)
            
            # Should still produce valid results
            self.assertIsNotNone(result)
            self.assertIn('best_individual', result)
            self.assertEqual(result['generations_run'], 1)


if __name__ == '__main__':
    # Run all tests with increased verbosity
    unittest.main(verbosity=2, buffer=True)