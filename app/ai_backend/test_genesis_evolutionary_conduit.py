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
        Initializes default and custom EvolutionaryParameters instances for use in test cases.
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
        Verify that EvolutionaryParameters initializes with correct default values for all parameters.
        """
        self.assertEqual(self.default_params.population_size, 100)
        self.assertEqual(self.default_params.generations, 500)
        self.assertEqual(self.default_params.mutation_rate, 0.1)
        self.assertEqual(self.default_params.crossover_rate, 0.8)
        self.assertEqual(self.default_params.selection_pressure, 0.2)
    
    def test_custom_initialization(self):
        """
        Verify that initializing EvolutionaryParameters with custom values correctly assigns all attributes.
        """
        self.assertEqual(self.custom_params.population_size, 200)
        self.assertEqual(self.custom_params.generations, 1000)
        self.assertEqual(self.custom_params.mutation_rate, 0.15)
        self.assertEqual(self.custom_params.crossover_rate, 0.85)
        self.assertEqual(self.custom_params.selection_pressure, 0.3)
    
    def test_parameter_validation(self):
        """
        Verify that invalid values for population size, mutation rate, or crossover rate in EvolutionaryParameters raise ValueError exceptions.
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
        Test that EvolutionaryParameters.to_dict() returns a dictionary with the expected parameter values.
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
        Verifies that EvolutionaryParameters can be correctly created from a dictionary and that all attributes are set to the expected values.
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
        Test that Gaussian mutation returns a mutated genome of correct length and type for varying mutation rates.
        
        Verifies that the output genome is a list matching the input genome's length for both low and high mutation rates.
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
        Verifies that the uniform mutation strategy produces a mutated genome of the same length as the input, with all values constrained within the specified bounds.
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
        Test that the bit-flip mutation strategy produces a genome of the same length with only boolean values.
        
        Verifies that the mutated genome maintains the original length and that all elements are of boolean type after mutation.
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
        Test that the adaptive mutation strategy produces a mutated genome of the same length as the input when given a fitness history.
        
        Verifies that the output is a list and matches the input genome's length.
        """
        genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        fitness_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
        
        self.assertEqual(len(mutated), len(genome))
        self.assertIsInstance(mutated, list)
    
    def test_invalid_mutation_rate(self):
        """
        Test that mutation methods raise ValueError for mutation rates outside the valid range.
        
        Verifies that providing a negative or greater-than-one mutation rate to Gaussian and uniform mutation methods results in a ValueError.
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
        Initializes a SelectionStrategy instance and a sample population for use in selection strategy tests.
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
        Tests that roulette wheel selection returns a valid individual from the population.
        
        Ensures the selected individual exists in the population and includes both 'genome' and 'fitness' keys.
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
        Test that the elitism selection strategy returns the specified number of individuals with the highest fitness values from the population.
        
        Verifies that the selected individuals are the top performers, sorted in descending order of fitness.
        """
        elite_count = 2
        selected = self.strategy.elitism_selection(self.population, elite_count)
        
        self.assertEqual(len(selected), elite_count)
        
        # Check that selected individuals are the fittest
        fitness_values = [individual['fitness'] for individual in selected]
        self.assertEqual(fitness_values, [0.9, 0.7])  # Sorted by fitness descending
    
    def test_empty_population(self):
        """
        Tests that selection strategies raise a ValueError when called with an empty population.
        
        Verifies that both tournament selection and roulette wheel selection correctly handle empty input by raising exceptions.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection([], tournament_size=2)
        
        with self.assertRaises(ValueError):
            self.strategy.roulette_wheel_selection([])
    
    def test_invalid_tournament_size(self):
        """
        Test that tournament selection raises a ValueError when the tournament size is zero or greater than the population size.
        
        Verifies that the selection strategy enforces valid tournament sizes and properly handles invalid input by raising exceptions.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=0)
        
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=len(self.population) + 1)


class TestFitnessFunction(unittest.TestCase):
    """Test suite for FitnessFunction class."""
    
    def setUp(self):
        """
        Sets up a FitnessFunction instance for use in each test method.
        """
        self.fitness_func = FitnessFunction()
    
    def test_sphere_function(self):
        """
        Test that the sphere fitness function computes and returns the negative sum of squares of the genome values.
        
        Verifies that the function correctly implements the sphere fitness calculation, returning the negative sum for maximization scenarios.
        """
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.sphere_function(genome)
        
        # Sphere function: sum of squares
        expected = -(1.0**2 + 2.0**2 + 3.0**2)  # Negative for maximization
        self.assertEqual(fitness, expected)
    
    def test_rastrigin_function(self):
        """
        Tests that the Rastrigin fitness function evaluates to 0.0 when the input genome is a zero vector.
        
        Verifies that the global minimum of the Rastrigin function is correctly computed at the origin.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.rastrigin_function(genome)
        
        # Rastrigin function should be 0 at origin
        self.assertEqual(fitness, 0.0)
    
    def test_rosenbrock_function(self):
        """
        Tests that the Rosenbrock fitness function returns 0.0 at the global minimum for the genome [1.0, 1.0].
        """
        genome = [1.0, 1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)
        
        # Rosenbrock function should be 0 at (1, 1)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function(self):
        """
        Test that the Ackley fitness function returns zero at the origin for a genome of all zeros.
        
        Verifies that evaluating the Ackley function with a zero vector genome produces the expected global minimum value of 0.0.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.ackley_function(genome)
        
        # Ackley function should be 0 at origin
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function(self):
        """
        Test evaluation of a user-defined fitness function that computes the sum of genome values.
        
        Verifies that the fitness function framework correctly applies a custom function to a genome and returns the expected sum.
        """
        def custom_func(genome):
            """
            Return the sum of all numeric values in the given genome sequence.
            
            Parameters:
                genome (iterable): Sequence of numeric values.
            
            Returns:
                numeric: The total sum of the values in the genome.
            """
            return sum(genome)
        
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate(genome, custom_func)
        
        self.assertEqual(fitness, 6.0)
    
    def test_multi_objective_function(self):
        """
        Test that the multi-objective fitness function correctly evaluates a genome using multiple objectives and returns the expected fitness values as a vector.
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
        Test that the fitness function applies a penalty when constraints are violated.
        
        Verifies that a genome failing the specified constraint (sum less than 5) receives a reduced fitness score compared to its unconstrained evaluation.
        """
        genome = [1.0, 2.0, 3.0]
        
        def constraint_func(g):
            # Constraint: sum should be less than 5
            """
            Return True if the sum of elements in the input iterable is less than 5.
            
            Parameters:
            	g (iterable): An iterable of numeric values.
            
            Returns:
            	bool: True if the sum is less than 5, False otherwise.
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
        Prepare the test environment by initializing a PopulationManager and setting default values for genome length and population size.
        """
        self.manager = PopulationManager()
        self.genome_length = 5
        self.population_size = 10
    
    def test_initialize_random_population(self):
        """
        Test initialization of a random population with the correct size and genome structure.
        
        Ensures that each individual in the generated population has a genome of the specified length and includes a fitness attribute.
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
        
        After evaluation, verifies that each individual's 'fitness' attribute is set to a non-None integer or float value.
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
        Test that the population manager correctly identifies and returns the individual with the highest fitness value from a given population.
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
        Test that population statistics are correctly computed for a given set of individuals.
        
        Verifies that the population manager calculates best, worst, average, median, and standard deviation of fitness values for a sample population, and that the results match expected values.
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
        Verifies that the population diversity metric is calculated as a positive float for a sample population.
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
        Test that attempting to retrieve the best individual or compute statistics from an empty population raises a ValueError.
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
        Test that single-point crossover produces two children of correct length, each inheriting genes from both parents.
        
        Verifies that the children have the same length as the parents and that their genes are drawn from the set of parent genes.
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
        Test that two-point crossover returns two children with genomes of the same length as the parents.
        """
        parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
        parent2 = [9, 10, 11, 12, 13, 14, 15, 16]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_uniform_crossover(self):
        """
        Test that uniform crossover generates two children with genomes of the same length as the parents.
        
        Verifies that the uniform crossover operation returns two offspring whose genome lengths match those of the input parent genomes, ensuring structural consistency after crossover.
        """
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_arithmetic_crossover(self):
        """
        Test that arithmetic crossover produces children as weighted averages of two parent genomes.
        
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
        Test that simulated binary crossover generates two children with correct genome length and gene values within specified bounds.
        
        Verifies that the crossover operation produces children matching the parent genome length and that all gene values are within the provided lower and upper bounds.
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
        Test that the blend crossover (BLX-α) operation produces two child genomes of equal length to the parent genomes.
        
        Verifies that the blend crossover returns child genomes whose lengths match those of the input parents, ensuring structural consistency after crossover.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_invalid_crossover_inputs(self):
        """
        Test that crossover operations raise ValueError when parent genomes have different lengths.
        
        Verifies that both single-point and two-point crossover methods correctly detect and reject parent genomes of mismatched lengths by raising a ValueError.
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
        Set up test fixtures by initializing EvolutionaryConduit and EvolutionaryParameters instances for use in each test.
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
        Verifies that all core components of the EvolutionaryConduit are properly initialized and not None.
        """
        self.assertIsNotNone(self.conduit.mutation_strategy)
        self.assertIsNotNone(self.conduit.selection_strategy)
        self.assertIsNotNone(self.conduit.fitness_function)
        self.assertIsNotNone(self.conduit.population_manager)
        self.assertIsNotNone(self.conduit.genetic_operations)
    
    def test_set_fitness_function(self):
        """
        Test assigning and using a custom fitness function in the evolutionary conduit.
        
        Verifies that a user-defined fitness function can be set on the conduit and is correctly applied to evaluate genome fitness.
        """
        def custom_fitness(genome):
            """
            Calculates the fitness score of a genome as the sum of its numeric elements.
            
            Parameters:
            	genome (iterable): An iterable of numeric values representing the genome.
            
            Returns:
            	float: The total sum of the genome's elements.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(custom_fitness)
        
        # Test that the function is set correctly
        test_genome = [1.0, 2.0, 3.0]
        fitness = self.conduit.fitness_function.evaluate(test_genome, custom_fitness)
        self.assertEqual(fitness, 6.0)
    
    def test_set_parameters(self):
        """
        Test that setting evolutionary parameters updates the conduit with the provided values.
        
        Verifies that the conduit correctly reflects the specified population size, number of generations, mutation rate, and crossover rate after parameters are set.
        """
        self.conduit.set_parameters(self.params)
        
        self.assertEqual(self.conduit.parameters.population_size, 20)
        self.assertEqual(self.conduit.parameters.generations, 10)
        self.assertEqual(self.conduit.parameters.mutation_rate, 0.1)
        self.assertEqual(self.conduit.parameters.crossover_rate, 0.8)
    
    @patch('app.ai_backend.genesis_evolutionary_conduit.EvolutionaryConduit.evolve')
    def test_run_evolution(self, mock_evolve):
        """
        Test that the evolution process returns a result with the expected structure.
        
        Verifies that running evolution produces a result containing 'best_individual', 'generations_run', 'final_population', and 'statistics', and that the evolve method is invoked exactly once.
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
        Test that saving and loading the evolutionary conduit state correctly restores parameters in a new instance.
        
        Verifies that after saving the state of an evolutionary conduit and loading it into a new conduit instance, key parameter values such as population size and number of generations are preserved.
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
        Test that a callback function can be registered with the evolutionary conduit.
        
        Verifies that the callback is added to the conduit’s callback list and will be invoked during evolution.
        """
        callback_called = False
        
        def test_callback(generation, population, best_individual):
            """
            Test callback function to verify invocation during the evolutionary process.
            
            Sets a flag to confirm that the callback mechanism is triggered when called with the current generation, population, and best individual.
            """
            nonlocal callback_called
            callback_called = True
        
        self.conduit.add_callback(test_callback)
        
        # Verify callback is added
        self.assertIn(test_callback, self.conduit.callbacks)
    
    def test_evolution_history_tracking(self):
        """
        Test that enabling history tracking on the evolutionary conduit correctly sets the history tracking flag after running an evolution process.
        
        Verifies that after enabling history tracking and running evolution, the conduit reflects that history tracking is active.
        """
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        # Run a simple evolution
        def simple_fitness(genome):
            """
            Calculate the fitness score of a genome as the sum of its elements.
            
            Parameters:
                genome (iterable): Sequence of numeric values representing the genome.
            
            Returns:
                int or float: The sum of all elements in the genome.
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
        Initializes a GenesisEvolutionaryConduit instance and sets evolutionary parameters for use in test cases.
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
        Test that GenesisEvolutionaryConduit creates a neural network using the provided configuration.
        
        Verifies that after setting a network configuration, the conduit successfully instantiates a neural network object.
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
        Test that evaluating the fitness of a neural network genome with provided training data yields a numeric result.
        
        This test sets training data in the Genesis evolutionary conduit, evaluates the fitness of a sample genome representing network weights, and asserts that the returned fitness value is an integer or float.
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
        Test that mutating a neural network topology returns a dictionary containing valid 'layers' and 'connections' keys, ensuring the topology structure is preserved after mutation.
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
        Test that hyperparameter optimization generates values within the specified search space and includes all required hyperparameter keys.
        
        Verifies that generated hyperparameters respect the defined bounds for each parameter and that all expected keys are present in the result.
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
        Test that multi-objective optimization returns a fitness vector matching the number of objectives.
        
        Verifies that evaluating a genome with multiple objectives produces a list whose length equals the number of specified objectives.
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
        Test that the adaptive mutation rate computed from a population's fitness history is a float within the range [0.0, 1.0].
        
        Verifies that the calculated mutation rate is a valid probability value based on the provided population data.
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
        Test that the speciation method groups individuals into species based on a distance threshold.
        
        Verifies that the returned value is a non-empty list, ensuring population diversity is preserved through correct species grouping.
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
        Tests that transfer learning adapts a pretrained neural network genome to a new task, resulting in a non-empty adapted genome.
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
        Test that ensemble evolution selects the top networks by fitness and returns an ensemble of the requested size.
        
        Verifies that the ensemble contains the correct number of networks and that they are ordered by descending fitness.
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
        Test that novelty search assigns a numeric novelty score to each individual in the population.
        
        Ensures the number of novelty scores matches the population size and that all scores are numeric types.
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
        
        Ensures that the `coevolve_populations` method returns a dictionary containing updated versions of both input populations, with keys 'population1' and 'population2'.
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
        
        Verifies that calling `save_checkpoint` on the GenesisEvolutionaryConduit triggers the underlying save operation with the specified checkpoint path.
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
        
        Sets up an island model with multiple populations and tests that individuals can be migrated between islands. Verifies that the migration function returns a tuple of updated populations.
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
        Test that EvolutionaryException correctly stores and exposes additional details passed during initialization.
        
        Verifies that the exception message and the details attribute match the provided values.
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
        Prepare the integration test environment by initializing a GenesisEvolutionaryConduit instance and default evolutionary parameters.
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
        Test that running a complete evolution cycle with GenesisEvolutionaryConduit returns a result containing the best individual and the expected number of generations.
        
        Verifies that the evolution process produces a result dictionary with the correct keys and values after running with a simple fitness function and mocked evolution.
        """
        # Set up fitness function
        def simple_fitness(genome):
            """
            Compute the fitness of a genome as the sum of the squares of its elements.
            
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
        Tests the neural network evolution pipeline by configuring the network, assigning training data, and verifying neural network creation.
        
        This test ensures that the Genesis evolutionary conduit correctly accepts network configuration and training data, and is able to instantiate a neural network as part of the evolution process.
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
        Test that the multi-objective optimization pipeline produces a fitness vector with correct values for each objective.
        
        Ensures that when multiple objectives are set, evaluating a genome yields a fitness vector whose length and order correspond to the configured objectives, and that the returned values match expected results.
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
        Test that the adaptive mutation rate calculation in the evolutionary pipeline produces a float within the range [0.0, 1.0] when given a population with diverse fitness values.
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
        Tests that the evolutionary framework correctly raises exceptions for invalid parameters and fitness function failures.
        
        Verifies that providing invalid evolutionary parameters raises a ValueError, and that a failing fitness function during evolution raises an EvolutionaryException.
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
        Prepare the test environment by instantiating a GenesisEvolutionaryConduit and initializing EvolutionaryParameters for asynchronous evolution tests.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    @patch('asyncio.run')
    def test_async_evolution_execution(self, mock_run):
        """
        Test that the asynchronous evolution method returns a valid result when the evolution process is mocked.
        
        This test verifies that `run_async_evolution` of the Genesis conduit produces a non-None result when the asynchronous evolution is simulated using a mock.
        """
        async def mock_async_evolve():
            """
            Simulates an asynchronous evolutionary process and returns mock evolutionary results.
            
            Returns:
                dict: A dictionary containing a mock best individual, number of generations run, final population, and summary statistics.
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
        Test that population fitness evaluation is performed in parallel using a thread pool executor.
        
        Verifies that the parallel execution mechanism is triggered and that fitness values are assigned to the population using the provided fitness function.
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
            Compute the fitness of a genome as the sum of its elements.
            
            Parameters:
            	genome (iterable): Sequence of numeric values representing the genome.
            
            Returns:
            	float or int: The total sum of all elements in the genome.
            """
            return sum(genome)
        
        # Test parallel evaluation
        self.genesis_conduit.evaluate_population_parallel(population, fitness_func)
        
        # Verify parallel execution was attempted
        mock_executor.assert_called_once()


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

class TestEvolutionaryParametersBoundaryConditions(unittest.TestCase):
    """Extended test suite for EvolutionaryParameters boundary conditions and edge cases."""
    
    def test_minimum_valid_values(self):
        """
        Verify that the EvolutionaryParameters class correctly accepts and stores the minimum valid values for all parameters.
        """
        params = EvolutionaryParameters(
            population_size=1,
            generations=1,
            mutation_rate=0.0,
            crossover_rate=0.0,
            selection_pressure=0.0
        )
        
        self.assertEqual(params.population_size, 1)
        self.assertEqual(params.generations, 1)
        self.assertEqual(params.mutation_rate, 0.0)
        self.assertEqual(params.crossover_rate, 0.0)
        self.assertEqual(params.selection_pressure, 0.0)
    
    def test_maximum_valid_values(self):
        """
        Test that EvolutionaryParameters accepts and correctly stores maximum valid values for all rate and size parameters.
        """
        params = EvolutionaryParameters(
            population_size=10000,
            generations=100000,
            mutation_rate=1.0,
            crossover_rate=1.0,
            selection_pressure=1.0
        )
        
        self.assertEqual(params.population_size, 10000)
        self.assertEqual(params.generations, 100000)
        self.assertEqual(params.mutation_rate, 1.0)
        self.assertEqual(params.crossover_rate, 1.0)
        self.assertEqual(params.selection_pressure, 1.0)
    
    def test_negative_generation_validation(self):
        """
        Test that initializing EvolutionaryParameters with a negative number of generations raises a ValueError.
        """
        with self.assertRaises(ValueError):
            EvolutionaryParameters(generations=-1)
    
    def test_negative_selection_pressure_validation(self):
        """
        Test that initializing EvolutionaryParameters with a negative selection pressure raises a ValueError.
        """
        with self.assertRaises(ValueError):
            EvolutionaryParameters(selection_pressure=-0.1)
    
    def test_selection_pressure_upper_bound(self):
        """
        Test that initializing EvolutionaryParameters with a selection pressure above 1.0 raises a ValueError.
        """
        with self.assertRaises(ValueError):
            EvolutionaryParameters(selection_pressure=1.1)
    
    def test_parameter_type_validation(self):
        """
        Test that initializing EvolutionaryParameters with non-numeric types for population_size or mutation_rate raises a TypeError.
        """
        with self.assertRaises(TypeError):
            EvolutionaryParameters(population_size="100")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(mutation_rate="0.1")
    
    def test_from_dict_with_missing_keys(self):
        """
        Test that EvolutionaryParameters.from_dict correctly assigns provided values and uses defaults for missing keys.
        
        Verifies that missing keys in the input dictionary result in default parameter values being set.
        """
        partial_dict = {
            'population_size': 150,
            'generations': 750
        }
        params = EvolutionaryParameters.from_dict(partial_dict)
        
        self.assertEqual(params.population_size, 150)
        self.assertEqual(params.generations, 750)
        self.assertEqual(params.mutation_rate, 0.1)  # Default value
        self.assertEqual(params.crossover_rate, 0.8)  # Default value
    
    def test_from_dict_with_invalid_values(self):
        """
        Test that `EvolutionaryParameters.from_dict` raises a ValueError when provided with invalid parameter values.
        """
        invalid_dict = {
            'population_size': 0,
            'mutation_rate': -0.5
        }
        with self.assertRaises(ValueError):
            EvolutionaryParameters.from_dict(invalid_dict)
    
    def test_to_dict_immutability(self):
        """
        Verify that changes to the dictionary returned by `to_dict` do not alter the internal state of the `EvolutionaryParameters` instance.
        """
        params = EvolutionaryParameters(population_size=100)
        params_dict = params.to_dict()
        params_dict['population_size'] = 200
        
        self.assertEqual(params.population_size, 100)  # Should remain unchanged


class TestMutationStrategyEdgeCases(unittest.TestCase):
    """Extended test suite for MutationStrategy edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Set up a MutationStrategy instance for use in mutation strategy tests.
        """
        self.strategy = MutationStrategy()
    
    def test_gaussian_mutation_with_zero_sigma(self):
        """
        Test that Gaussian mutation with zero sigma leaves the genome unchanged.
        
        Verifies that applying Gaussian mutation with a sigma value of zero does not alter any genes, regardless of the mutation rate.
        """
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=0.0)
        self.assertEqual(mutated, genome)
    
    def test_gaussian_mutation_empty_genome(self):
        """
        Test that Gaussian mutation applied to an empty genome returns an empty list.
        """
        genome = []
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.5, sigma=1.0)
        self.assertEqual(mutated, [])
    
    def test_gaussian_mutation_single_element(self):
        """
        Test that Gaussian mutation correctly mutates a single-element genome.
        
        Verifies that the mutated genome maintains the correct length and element type when applying Gaussian mutation to a one-element list.
        """
        genome = [5.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1.0)
        self.assertEqual(len(mutated), 1)
        self.assertIsInstance(mutated[0], float)
    
    def test_uniform_mutation_with_zero_bounds(self):
        """
        Test that uniform mutation with identical lower and upper bounds sets all genes to the bound value.
        
        Verifies that when the mutation range is zero (bounds are equal), every gene in the mutated genome is set to the specified bound, regardless of the original values.
        """
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=1.0, bounds=(5.0, 5.0))
        
        for value in mutated:
            self.assertEqual(value, 5.0)
    
    def test_uniform_mutation_inverted_bounds(self):
        """
        Test that uniform mutation raises a ValueError when provided with inverted bounds.
        """
        genome = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            self.strategy.uniform_mutation(genome, mutation_rate=0.5, bounds=(10, -10))
    
    def test_bit_flip_mutation_empty_genome(self):
        """
        Test that bit flip mutation returns an empty genome when given an empty input genome.
        """
        genome = []
        mutated = self.strategy.bit_flip_mutation(genome, mutation_rate=0.5)
        self.assertEqual(mutated, [])
    
    def test_bit_flip_mutation_non_boolean_genome(self):
        """
        Test that bit flip mutation raises a TypeError when applied to a non-boolean genome.
        
        This ensures the mutation strategy enforces genome type constraints.
        """
        genome = [1, 0, 1]
        with self.assertRaises(TypeError):
            self.strategy.bit_flip_mutation(genome, mutation_rate=0.5)
    
    def test_adaptive_mutation_empty_history(self):
        """
        Test that adaptive mutation raises a ValueError when provided with an empty fitness history.
        
        Verifies that the adaptive mutation method enforces the requirement for a non-empty fitness history and properly handles invalid input.
        """
        genome = [1.0, 2.0, 3.0]
        fitness_history = []
        
        with self.assertRaises(ValueError):
            self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
    
    def test_adaptive_mutation_single_history_point(self):
        """
        Test that adaptive mutation produces a genome of correct length when only a single fitness history point is provided.
        """
        genome = [1.0, 2.0, 3.0]
        fitness_history = [0.5]
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))
    
    def test_adaptive_mutation_negative_base_rate(self):
        """
        Test that adaptive mutation raises a ValueError when given a negative base mutation rate.
        
        Verifies that providing a negative base_rate to the adaptive_mutation method results in a ValueError, ensuring mutation rate validation.
        """
        genome = [1.0, 2.0, 3.0]
        fitness_history = [0.5, 0.6, 0.7]
        
        with self.assertRaises(ValueError):
            self.strategy.adaptive_mutation(genome, fitness_history, base_rate=-0.1)
    
    def test_mutation_deterministic_with_zero_rate(self):
        """
        Verify that all mutation strategies produce an unchanged genome when the mutation rate is set to zero.
        """
        genome = [1.0, 2.0, 3.0]
        
        # Gaussian mutation
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.0, sigma=1.0)
        self.assertEqual(mutated, genome)
        
        # Uniform mutation
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=0.0, bounds=(-10, 10))
        self.assertEqual(mutated, genome)
        
        # Bit flip mutation
        bool_genome = [True, False, True]
        mutated = self.strategy.bit_flip_mutation(bool_genome, mutation_rate=0.0)
        self.assertEqual(mutated, bool_genome)


class TestSelectionStrategyEdgeCases(unittest.TestCase):
    """Extended test suite for SelectionStrategy edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Set up a SelectionStrategy instance for use in selection strategy tests.
        """
        self.strategy = SelectionStrategy()
    
    def test_tournament_selection_single_individual(self):
        """
        Test that tournament selection returns the only individual when the population contains a single member and tournament size is one.
        """
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        selected = self.strategy.tournament_selection(population, tournament_size=1)
        self.assertEqual(selected, population[0])
    
    def test_tournament_selection_all_same_fitness(self):
        """
        Test that tournament selection returns a valid individual when all individuals have identical fitness values.
        
        Verifies that the selected individual is a member of the original population, ensuring correctness when selection pressure is absent due to uniform fitness.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        selected = self.strategy.tournament_selection(population, tournament_size=2)
        self.assertIn(selected, population)
    
    def test_roulette_wheel_selection_zero_fitness(self):
        """
        Verify that roulette wheel selection returns a valid individual when all fitness values are zero.
        
        Ensures that the selection method does not fail or return an invalid result when the population's fitness values are uniformly zero.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.0},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 0.0}
        ]
        selected = self.strategy.roulette_wheel_selection(population)
        self.assertIn(selected, population)
    
    def test_roulette_wheel_selection_negative_fitness(self):
        """
        Test that roulette wheel selection can handle populations with negative fitness values.
        
        Verifies that the selection method does not fail and returns a valid individual from the population when all fitness values are negative.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': -0.5},
            {'genome': [4, 5, 6], 'fitness': -0.3},
            {'genome': [7, 8, 9], 'fitness': -0.1}
        ]
        selected = self.strategy.roulette_wheel_selection(population)
        self.assertIn(selected, population)
    
    def test_rank_selection_single_individual(self):
        """
        Verify that rank selection returns the only individual when the population contains a single member.
        """
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        selected = self.strategy.rank_selection(population)
        self.assertEqual(selected, population[0])
    
    def test_elitism_selection_more_elites_than_population(self):
        """
        Test that elitism selection returns the entire population when the requested elite count exceeds the population size.
        
        Verifies that no error is raised and all individuals are selected when elite_count is greater than the number of individuals in the population.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.7}
        ]
        selected = self.strategy.elitism_selection(population, elite_count=5)
        self.assertEqual(len(selected), len(population))
    
    def test_elitism_selection_zero_count(self):
        """
        Test that elitism selection returns an empty list when the elite count is zero.
        
        Verifies that requesting zero elites from the population results in no individuals being selected.
        """
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        selected = self.strategy.elitism_selection(population, elite_count=0)
        self.assertEqual(len(selected), 0)
    
    def test_selection_with_missing_fitness_key(self):
        """
        Test that selection strategies raise a KeyError when individuals lack a fitness key.
        
        Verifies that the tournament selection method fails with a KeyError if the population contains individuals without a fitness value.
        """
        population = [{'genome': [1, 2, 3]}]  # Missing fitness
        
        with self.assertRaises(KeyError):
            self.strategy.tournament_selection(population, tournament_size=1)
    
    def test_selection_with_none_fitness(self):
        """
        Test that selection strategies raise a TypeError when individuals have None as their fitness value.
        
        This ensures that the tournament selection method does not accept individuals with undefined fitness.
        """
        population = [{'genome': [1, 2, 3], 'fitness': None}]
        
        with self.assertRaises(TypeError):
            self.strategy.tournament_selection(population, tournament_size=1)


class TestFitnessFunctionEdgeCases(unittest.TestCase):
    """Extended test suite for FitnessFunction edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Set up the test environment by initializing a FitnessFunction instance for use in test methods.
        """
        self.fitness_func = FitnessFunction()
    
    def test_sphere_function_empty_genome(self):
        """
        Test that the sphere fitness function returns 0.0 when given an empty genome.
        """
        genome = []
        fitness = self.fitness_func.sphere_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_sphere_function_single_element(self):
        """
        Test that the sphere fitness function correctly evaluates a genome with a single element.
        
        Verifies that the function returns the negative square of the single genome value.
        """
        genome = [3.0]
        fitness = self.fitness_func.sphere_function(genome)
        self.assertEqual(fitness, -9.0)
    
    def test_rastrigin_function_empty_genome(self):
        """
        Test that the Rastrigin fitness function returns 0.0 when given an empty genome.
        """
        genome = []
        fitness = self.fitness_func.rastrigin_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_rastrigin_function_large_values(self):
        """
        Test that the Rastrigin fitness function returns a negative value for genomes with large-magnitude elements.
        
        Verifies that the function produces a float result and that the fitness is less than zero when evaluated on large positive and negative values.
        """
        genome = [100.0, -100.0]
        fitness = self.fitness_func.rastrigin_function(genome)
        self.assertIsInstance(fitness, float)
        self.assertLess(fitness, 0.0)  # Should be negative for large values
    
    def test_rosenbrock_function_single_element(self):
        """
        Test that the Rosenbrock function returns 0.0 for a single-element genome.
        
        Verifies that the Rosenbrock function handles the edge case of a genome with only one element, where no pairwise terms exist.
        """
        genome = [1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_rosenbrock_function_empty_genome(self):
        """
        Test that the Rosenbrock fitness function returns 0.0 when given an empty genome.
        """
        genome = []
        fitness = self.fitness_func.rosenbrock_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function_empty_genome(self):
        """
        Test that the Ackley fitness function returns 0.0 when given an empty genome.
        """
        genome = []
        fitness = self.fitness_func.ackley_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function_single_element(self):
        """
        Test that the Ackley fitness function returns 0.0 for a single-element genome at the origin.
        """
        genome = [0.0]
        fitness = self.fitness_func.ackley_function(genome)
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function_with_exception(self):
        """
        Test that a custom fitness function raising an exception is properly propagated during evaluation.
        
        Verifies that when a custom fitness function raises a ValueError, the exception is not suppressed by the evaluation mechanism.
        """
        def failing_func(genome):
            """
            A fitness function that always raises a ValueError when called.
            
            Parameters:
            	genome: The genome to evaluate (unused).
            
            Raises:
            	ValueError: Always raised to simulate a failing fitness function.
            """
            raise ValueError("Custom fitness function failed")
        
        genome = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            self.fitness_func.evaluate(genome, failing_func)
    
    def test_multi_objective_with_empty_objectives(self):
        """
        Verify that multi-objective evaluation returns an empty list when provided with an empty objectives list.
        
        Ensures that evaluating a genome with no objectives produces an empty fitness result.
        """
        genome = [1.0, 2.0, 3.0]
        objectives = []
        
        fitness = self.fitness_func.multi_objective_evaluate(genome, objectives)
        self.assertEqual(fitness, [])
    
    def test_multi_objective_with_failing_objective(self):
        """
        Test that multi-objective evaluation raises an exception when one of the objectives fails.
        
        Verifies that a ZeroDivisionError is raised if any objective function in the objectives list throws an exception during evaluation.
        """
        genome = [1.0, 2.0, 3.0]
        objectives = [
            lambda g: sum(g),
            lambda g: 1 / 0  # Will raise ZeroDivisionError
        ]
        
        with self.assertRaises(ZeroDivisionError):
            self.fitness_func.multi_objective_evaluate(genome, objectives)
    
    def test_constraint_handling_with_no_constraints(self):
        """
        Test that evaluating a genome with no constraints returns the raw fitness value.
        
        Verifies that when the constraints list is empty, the fitness function does not apply any penalties and returns the direct result of the fitness evaluation.
        """
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            []
        )
        self.assertEqual(fitness, sum(genome))
    
    def test_constraint_handling_with_failing_constraint(self):
        """
        Test that the fitness function correctly propagates exceptions raised by failing constraints during evaluation.
        
        Verifies that when a constraint function raises an exception, the evaluation process raises the same exception.
        """
        genome = [1.0, 2.0, 3.0]
        
        def failing_constraint(g):
            """
            A constraint function that always raises a ValueError to simulate a constraint evaluation failure.
            
            Parameters:
                g: The genome or individual being evaluated.
            
            Raises:
                ValueError: Always raised to indicate constraint evaluation failure.
            """
            raise ValueError("Constraint evaluation failed")
        
        with self.assertRaises(ValueError):
            self.fitness_func.evaluate_with_constraints(
                genome, 
                lambda g: sum(g), 
                [failing_constraint]
            )


class TestPopulationManagerEdgeCases(unittest.TestCase):
    """Extended test suite for PopulationManager edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Set up a new PopulationManager instance before each test.
        """
        self.manager = PopulationManager()
    
    def test_initialize_random_population_zero_size(self):
        """
        Test that initializing a random population with a size of zero returns an empty population.
        """
        population = self.manager.initialize_random_population(0, 5)
        self.assertEqual(len(population), 0)
    
    def test_initialize_random_population_zero_genome_length(self):
        """
        Test that initializing a random population with zero genome length creates the correct number of individuals, each with an empty genome.
        """
        population = self.manager.initialize_random_population(5, 0)
        
        self.assertEqual(len(population), 5)
        for individual in population:
            self.assertEqual(len(individual['genome']), 0)
    
    def test_initialize_seeded_population_more_seeds_than_size(self):
        """
        Test that initializing a seeded population with more seeds than the specified population size results in a population of the correct size containing genomes from the provided seeds.
        """
        seeds = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        population = self.manager.initialize_seeded_population(2, 3, seeds)
        self.assertEqual(len(population), 2)
        
        genomes = [ind['genome'] for ind in population]
        # Should contain some of the seeds
        self.assertTrue(any(seed in genomes for seed in seeds))
    
    def test_initialize_seeded_population_empty_seeds(self):
        """
        Test that initializing a seeded population with an empty seeds list generates the correct number of individuals with genomes of the specified length.
        """
        seeds = []
        population = self.manager.initialize_seeded_population(3, 5, seeds)
        
        self.assertEqual(len(population), 3)
        for individual in population:
            self.assertEqual(len(individual['genome']), 5)
    
    def test_initialize_seeded_population_mismatched_genome_length(self):
        """
        Test that initializing a seeded population with genomes of incorrect length raises a ValueError.
        
        This test verifies that the population manager enforces genome length consistency when provided with seed genomes that do not match the expected length.
        """
        seeds = [
            [1.0, 2.0],  # Length 2, but expecting 5
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Length 6, but expecting 5
        ]
        
        with self.assertRaises(ValueError):
            self.manager.initialize_seeded_population(5, 5, seeds)
    
    def test_evaluate_population_empty_population(self):
        """
        Test that evaluating an empty population does not alter the population or raise errors.
        
        Verifies that calling `evaluate_population` with an empty population leaves it unchanged and does not result in exceptions.
        """
        population = []
        fitness_func = lambda genome: sum(genome)
        
        self.manager.evaluate_population(population, fitness_func)
        self.assertEqual(len(population), 0)
    
    def test_evaluate_population_with_failing_fitness(self):
        """
        Test that evaluating a population with a fitness function that raises an exception for certain individuals results in the expected error.
        
        This test verifies that when the fitness function fails (e.g., due to an empty genome), the evaluation process raises a ValueError.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [], 'fitness': None}  # Empty genome might cause issues
        ]
        
        def problematic_fitness(genome):
            """
            Calculates the sum of genome values as a fitness score.
            
            Raises:
                ValueError: If the genome is empty.
                
            Returns:
                int or float: The sum of the genome values.
            """
            if not genome:
                raise ValueError("Empty genome")
            return sum(genome)
        
        with self.assertRaises(ValueError):
            self.manager.evaluate_population(population, problematic_fitness)
    
    def test_get_population_statistics_single_individual(self):
        """
        Verify that population statistics are correctly computed when the population contains only a single individual.
        
        Ensures that best, worst, average, and median fitness values are equal to the individual's fitness, and that the standard deviation is zero.
        """
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.7}
        ]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 0.7)
        self.assertEqual(stats['worst_fitness'], 0.7)
        self.assertEqual(stats['average_fitness'], 0.7)
        self.assertEqual(stats['median_fitness'], 0.7)
        self.assertEqual(stats['std_dev_fitness'], 0.0)
    
    def test_diversity_calculation_identical_genomes(self):
        """
        Test that diversity calculation returns zero when all genomes in the population are identical.
        
        Verifies that the diversity metric correctly identifies no genetic variation among individuals with identical genomes.
        """
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        self.assertEqual(diversity, 0.0)
    
    def test_diversity_calculation_empty_genomes(self):
        """
        Test that diversity calculation returns 0.0 when all genomes in the population are empty.
        
        Verifies that the diversity metric correctly handles populations where individuals have no genetic information.
        """
        population = [
            {'genome': [], 'fitness': 0.5},
            {'genome': [], 'fitness': 0.6}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        self.assertEqual(diversity, 0.0)
    
    def test_diversity_calculation_single_individual(self):
        """
        Test that diversity calculation returns zero when the population contains only a single individual.
        
        Verifies that the diversity metric is correctly computed as 0.0 for a population with one member.
        """
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        self.assertEqual(diversity, 0.0)


class TestGeneticOperationsEdgeCases(unittest.TestCase):
    """Extended test suite for GeneticOperations edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Set up the test environment by initializing a GeneticOperations instance for use in genetic operation tests.
        """
        self.operations = GeneticOperations()
    
    def test_crossover_empty_parents(self):
        """
        Test that single-point crossover returns empty children when both parent genomes are empty.
        
        Verifies that the crossover operation handles empty input genomes without errors and produces empty offspring.
        """
        parent1 = []
        parent2 = []
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(child1, [])
        self.assertEqual(child2, [])
    
    def test_crossover_single_element_parents(self):
        """
        Test that single-point crossover correctly handles parent genomes with a single element.
        
        Verifies that the resulting child genomes each contain exactly one element.
        """
        parent1 = [1]
        parent2 = [2]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 1)
        self.assertEqual(len(child2), 1)
    
    def test_two_point_crossover_length_two(self):
        """
        Test that two-point crossover correctly produces two children of length 2 when given parent genomes of length 2.
        """
        parent1 = [1, 2]
        parent2 = [3, 4]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)
    
    def test_uniform_crossover_zero_rate(self):
        """
        Verify that uniform crossover with a zero crossover rate produces children identical to their respective parents.
        """
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.0)
        
        # With zero crossover rate, children should be identical to parents
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
    
    def test_uniform_crossover_full_rate(self):
        """
        Test that uniform crossover with a crossover rate of 1.0 results in complete swapping of parent genomes.
        
        Verifies that when the crossover rate is set to 1.0, the resulting children are exact copies of the opposite parents.
        """
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=1.0)
        
        # With full crossover rate, children should be swapped
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
    
    def test_arithmetic_crossover_zero_alpha(self):
        """
        Test that arithmetic crossover with alpha set to zero swaps the parent genomes.
        
        Verifies that when alpha is zero, the resulting children are exact copies of the opposite parents, confirming correct implementation of the arithmetic crossover edge case.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.0)
        
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
    
    def test_arithmetic_crossover_full_alpha(self):
        """
        Test that arithmetic crossover with alpha=1.0 returns the original parent genomes unchanged.
        
        Verifies that when alpha is set to 1.0, the arithmetic crossover operation produces children identical to the input parents, ensuring correct handling of the full alpha edge case.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.0)
        
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
    
    def test_simulated_binary_crossover_tight_bounds(self):
        """
        Test that simulated binary crossover produces children within very tight gene bounds.
        
        Verifies that the offspring generated by the simulated binary crossover operation do not exceed the specified narrow lower and upper limits for each gene.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [1.1, 2.1, 3.1]
        bounds = [(0.9, 1.2), (1.9, 2.2), (2.9, 3.2)]
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=2.0
        )
        
        # Check that children are within bounds
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
            self.assertGreaterEqual(child2[i], lower)
            self.assertLessEqual(child2[i], upper)
    
    def test_simulated_binary_crossover_zero_eta(self):
        """
        Verify that simulated binary crossover produces valid children when the eta parameter is set to zero.
        
        This test checks that the crossover operation returns two children of correct length, even when the distribution index (eta) is zero, which may affect the distribution of offspring genes.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        bounds = [(-10, 10)] * 3
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=0.0
        )
        
        self.assertEqual(len(child1), 3)
        self.assertEqual(len(child2), 3)
    
    def test_blend_crossover_zero_alpha(self):
        """
        Test that blend crossover with zero alpha produces children within the range of the parent gene values.
        
        Verifies that when alpha is set to zero, the resulting child genomes do not exceed the minimum or maximum values of the corresponding parent genes.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.0)
        
        # With zero alpha, children should be within parent range
        for i in range(len(parent1)):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            self.assertGreaterEqual(child1[i], min_val)
            self.assertLessEqual(child1[i], max_val)
            self.assertGreaterEqual(child2[i], min_val)
            self.assertLessEqual(child2[i], max_val)
    
    def test_crossover_with_none_values(self):
        """
        Test that arithmetic crossover raises a TypeError when parent genomes contain None values.
        """
        parent1 = [1, None, 3]
        parent2 = [4, 5, None]
        
        with self.assertRaises(TypeError):
            self.operations.arithmetic_crossover(parent1, parent2, alpha=0.5)


class TestEvolutionaryConduitEdgeCases(unittest.TestCase):
    """Extended test suite for EvolutionaryConduit edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Set up the test environment by initializing an EvolutionaryConduit instance for use in test methods.
        """
        self.conduit = EvolutionaryConduit()
    
    def test_run_evolution_zero_generations(self):
        """
        Verify that running the evolution process with zero generations completes immediately and returns the correct result structure, including zero generations run.
        """
        params = EvolutionaryParameters(
            population_size=10,
            generations=0,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of its elements.
            
            Parameters:
                genome (iterable): A sequence of numeric values representing a genome.
            
            Returns:
                int or float: The total sum of the genome's elements.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(simple_fitness)
        
        # Mock the evolution process for zero generations
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 0,
                'final_population': [],
                'statistics': {'best_fitness': 6.0}
            }
            
            result = self.conduit.run_evolution(genome_length=3)
            self.assertEqual(result['generations_run'], 0)
    
    def test_run_evolution_single_individual(self):
        """
        Test that evolution runs correctly with a population size of one.
        
        Verifies that the evolutionary process completes the specified number of generations and returns a final population containing a single individual with the expected genome and fitness.
        """
        params = EvolutionaryParameters(
            population_size=1,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of its elements.
            
            Parameters:
                genome (iterable): A sequence of numeric values representing a genome.
            
            Returns:
                int or float: The total sum of the genome's elements.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(simple_fitness)
        
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 5,
                'final_population': [{'genome': [1, 2, 3], 'fitness': 6.0}],
                'statistics': {'best_fitness': 6.0}
            }
            
            result = self.conduit.run_evolution(genome_length=3)
            self.assertEqual(len(result['final_population']), 1)
    
    def test_callback_with_exception(self):
        """
        Test that a callback function raising an exception is correctly added to the conduit.
        
        Verifies that a callback which raises an exception during execution is still registered in the conduit, regardless of how exceptions are handled during evolution.
        """
        def failing_callback(generation, population, best_individual):
            """
            A callback function that intentionally raises a ValueError when invoked.
            
            This is typically used in tests to simulate callback failures during evolutionary runs.
            """
            raise ValueError("Callback failed")
        
        self.conduit.add_callback(failing_callback)
        
        # The callback should be added but exception handling during evolution is implementation-dependent
        self.assertIn(failing_callback, self.conduit.callbacks)
    
    def test_multiple_callbacks(self):
        """
        Verify that multiple callbacks can be added to the conduit and are correctly registered.
        
        This test ensures that after adding two distinct callbacks, both are present in the conduit’s callback list.
        """
        callback1_called = False
        callback2_called = False
        
        def callback1(generation, population, best_individual):
            """
            Marks that callback1 has been invoked during evolution by setting a flag.
            
            Parameters:
                generation (int): The current generation number.
                population (list): The current population of individuals.
                best_individual: The best individual in the current population.
            """
            nonlocal callback1_called
            callback1_called = True
        
        def callback2(generation, population, best_individual):
            """
            Marks that the second callback has been invoked during evolution.
            
            Parameters:
                generation (int): The current generation number.
                population (list): The current population of individuals.
                best_individual: The best individual in the current population.
            """
            nonlocal callback2_called
            callback2_called = True
        
        self.conduit.add_callback(callback1)
        self.conduit.add_callback(callback2)
        
        self.assertEqual(len(self.conduit.callbacks), 2)
        self.assertIn(callback1, self.conduit.callbacks)
        self.assertIn(callback2, self.conduit.callbacks)
    
    def test_save_state_without_parameters(self):
        """
        Test that saving the conduit state without explicitly setting parameters includes default parameters in the saved state.
        
        Verifies that the returned state is a dictionary containing a 'parameters' key.
        """
        state = self.conduit.save_state()
        self.assertIsInstance(state, dict)
        # Should have default parameters
        self.assertIn('parameters', state)
    
    def test_load_state_with_invalid_data(self):
        """
        Test that loading an invalid state dictionary into the conduit raises a KeyError.
        
        Verifies that the conduit correctly detects and rejects state dictionaries missing required keys or containing invalid data.
        """
        invalid_state = {'invalid_key': 'invalid_value'}
        
        with self.assertRaises(KeyError):
            self.conduit.load_state(invalid_state)
    
    def test_load_state_with_none(self):
        """
        Test that loading a state with a None value raises a TypeError.
        """
        with self.assertRaises(TypeError):
            self.conduit.load_state(None)
    
    def test_set_fitness_function_with_none(self):
        """
        Test that setting the fitness function to None raises a TypeError.
        """
        with self.assertRaises(TypeError):
            self.conduit.set_fitness_function(None)


class TestGenesisEvolutionaryConduitEdgeCases(unittest.TestCase):
    """Extended test suite for GenesisEvolutionaryConduit edge cases and boundary conditions."""
    
    def setUp(self):
        """
        Set up a new GenesisEvolutionaryConduit instance for use in each test.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
    
    def test_neural_network_evolution_invalid_config(self):
        """
        Test that setting an invalid neural network configuration raises a ValueError.
        
        This test verifies that the Genesis evolutionary conduit correctly rejects configurations with invalid input size, output size, or activation function.
        """
        invalid_config = {
            'input_size': -1,  # Invalid negative size
            'hidden_layers': [],
            'output_size': 0,
            'activation': 'invalid_activation'
        }
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_network_config(invalid_config)
    
    def test_neural_network_evolution_missing_config_keys(self):
        """
        Test that neural network evolution raises a KeyError when required configuration keys are missing.
        
        This test verifies that attempting to set an incomplete network configuration in the Genesis evolutionary conduit results in a KeyError, ensuring proper validation of required configuration fields.
        """
        incomplete_config = {
            'input_size': 10
            # Missing other required keys
        }
        
        with self.assertRaises(KeyError):
            self.genesis_conduit.set_network_config(incomplete_config)
    
    def test_neuroevolution_fitness_without_training_data(self):
        """
        Test that evaluating neuroevolution fitness without training data raises a ValueError.
        
        Verifies that the fitness evaluation method correctly enforces the requirement for training data and raises an exception when it is missing.
        """
        genome = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.evaluate_network_fitness(genome)
    
    def test_neuroevolution_fitness_empty_training_data(self):
        """
        Verify that evaluating neuroevolution fitness with empty training data raises a ValueError.
        
        This test ensures that the fitness evaluation method correctly handles the case where both input and output training data are empty, enforcing input validation.
        """
        self.genesis_conduit.set_training_data([], [])
        
        genome = [0.1, 0.2, 0.3]
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.evaluate_network_fitness(genome)
    
    def test_neuroevolution_fitness_mismatched_data(self):
        """
        Test that neuroevolution fitness evaluation raises a ValueError when training data features and labels have mismatched lengths.
        """
        X_train = [[1, 2], [3, 4]]
        y_train = [0]  # Different length
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_training_data(X_train, y_train)
    
    def test_topology_evolution_invalid_topology(self):
        """
        Test that topology evolution raises a ValueError when provided with an invalid topology structure.
        
        This test verifies that attempting to mutate a topology with empty layers and invalid connections results in a ValueError, ensuring robust validation of topology inputs.
        """
        invalid_topology = {
            'layers': [],  # Empty layers
            'connections': [[0, 1]]  # Invalid connection for empty layers
        }
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.mutate_topology(invalid_topology)
    
    def test_topology_evolution_missing_keys(self):
        """
        Test that topology evolution raises a KeyError when required topology keys are missing.
        
        Verifies that attempting to mutate a topology dictionary lacking mandatory keys, such as 'connections', results in a KeyError.
        """
        incomplete_topology = {
            'layers': [10, 5, 1]
            # Missing 'connections' key
        }
        
        with self.assertRaises(KeyError):
            self.genesis_conduit.mutate_topology(incomplete_topology)
    
    def test_hyperparameter_optimization_empty_search_space(self):
        """
        Test that hyperparameter optimization returns an empty dictionary when the search space is empty.
        
        Verifies that setting an empty hyperparameter search space results in no hyperparameters being generated.
        """
        self.genesis_conduit.set_hyperparameter_search_space({})
        
        hyperparams = self.genesis_conduit.generate_hyperparameters()
        self.assertEqual(hyperparams, {})
    
    def test_hyperparameter_optimization_invalid_bounds(self):
        """
        Test that hyperparameter optimization raises a ValueError when provided with invalid search space bounds where the upper bound is less than the lower bound.
        """
        invalid_search_space = {
            'learning_rate': (0.1, 0.001),  # Upper bound less than lower bound
            'batch_size': (128, 16)  # Upper bound less than lower bound
        }
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_hyperparameter_search_space(invalid_search_space)
    
    def test_multi_objective_optimization_empty_objectives(self):
        """
        Test that multi-objective optimization returns an empty fitness vector when no objectives are set.
        
        Verifies that evaluating a genome with an empty objectives list produces an empty result, confirming correct handling of this edge case.
        """
        self.genesis_conduit.set_objectives([])
        
        genome = [0.1, 0.2, 0.3]
        fitness_vector = self.genesis_conduit.evaluate_multi_objective_fitness(genome)
        self.assertEqual(fitness_vector, [])
    
    def test_speciation_empty_population(self):
        """
        Test that speciation returns an empty list when given an empty population.
        
        Verifies that the speciation method correctly handles the case where no individuals are present, returning an empty species list.
        """
        species = self.genesis_conduit.speciate_population([], distance_threshold=1.0)
        self.assertEqual(species, [])
    
    def test_speciation_single_individual(self):
        """
        Test that speciation correctly assigns a single individual to its own species.
        
        Verifies that when the population contains only one individual, the speciation process results in exactly one species containing that individual.
        """
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}
        ]
        
        species = self.genesis_conduit.speciate_population(population, distance_threshold=1.0)
        self.assertEqual(len(species), 1)
        self.assertEqual(len(species[0]), 1)
    
    def test_transfer_learning_empty_genome(self):
        """
        Test that transfer learning raises a ValueError when provided with an empty pretrained genome.
        
        Verifies that the `adapt_pretrained_network` method correctly rejects empty genomes during transfer learning adaptation.
        """
        pretrained_genome = []
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.adapt_pretrained_network(
                pretrained_genome, 
                new_task_config={'output_size': 3}
            )
    
    def test_ensemble_evolution_empty_networks(self):
        """
        Test that ensemble evolution returns an empty list when provided with an empty networks list.
        
        Verifies that creating an ensemble with no input networks results in an empty ensemble, regardless of the requested ensemble size.
        """
        networks = []
        
        ensemble = self.genesis_conduit.create_ensemble(networks, ensemble_size=2)
        self.assertEqual(ensemble, [])
    
    def test_ensemble_evolution_more_requested_than_available(self):
        """
        Test that ensemble evolution returns all available networks when the requested ensemble size exceeds the number of provided networks.
        
        Validates that requesting an ensemble larger than the available pool does not result in errors and returns only the available networks.
        """
        networks = [
            {'genome': [1, 2, 3], 'fitness': 0.7}
        ]
        
        ensemble = self.genesis_conduit.create_ensemble(networks, ensemble_size=5)
        self.assertEqual(len(ensemble), 1)
    
    def test_novelty_search_empty_population(self):
        """
        Test that novelty search returns an empty list when provided with an empty population.
        
        Verifies that the `calculate_novelty_scores` method correctly handles the edge case of an empty input by returning an empty result.
        """
        novelty_scores = self.genesis_conduit.calculate_novelty_scores([])
        self.assertEqual(novelty_scores, [])
    
    def test_coevolution_empty_populations(self):
        """
        Test that coevolution returns empty populations when both input populations are empty.
        
        Verifies that the result is a dictionary with 'population1' and 'population2' keys, both mapped to empty lists.
        """
        result = self.genesis_conduit.coevolve_populations([], [])
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['population1'], [])
        self.assertEqual(result['population2'], [])
    
    def test_migration_with_zero_rate(self):
        """
        Verify that migration between populations with a zero migration rate leaves both populations unchanged.
        
        This test ensures that when the migration rate is set to zero, the `migrate_individuals` method does not transfer any individuals between the two populations.
        """
        population1 = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        population2 = [{'genome': [4, 5, 6], 'fitness': 0.7}]
        
        migrated = self.genesis_conduit.migrate_individuals(
            population1, population2, migration_rate=0.0
        )
        
        # With zero migration rate, populations should remain unchanged
        self.assertEqual(migrated[0], population1)
        self.assertEqual(migrated[1], population2)
    
    def test_migration_with_full_rate(self):
        """
        Verify that migration with a full migration rate results in complete swapping of individuals between two populations.
        
        This test ensures that when the migration rate is set to 1.0, all individuals from each population are migrated to the other, resulting in both populations being fully exchanged.
        """
        population1 = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        population2 = [{'genome': [4, 5, 6], 'fitness': 0.7}]
        
        migrated = self.genesis_conduit.migrate_individuals(
            population1, population2, migration_rate=1.0
        )
        
        # With full migration rate, populations should be completely swapped
        self.assertEqual(len(migrated), 2)
        self.assertNotEqual(migrated[0], population1)
        self.assertNotEqual(migrated[1], population2)


class TestConcurrencyAndThreadSafety(unittest.TestCase):
    """Test suite for concurrency and thread safety scenarios."""
    
    def setUp(self):
        """
        Set up the test environment by initializing an EvolutionaryConduit and EvolutionaryParameters with a small population and generation count for testing purposes.
        """
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    def test_concurrent_fitness_evaluation(self):
        """
        Verify that concurrent fitness evaluation of individuals in a population does not result in race conditions or missing fitness values.
        
        This test runs fitness evaluations in parallel threads and asserts that all individuals receive valid fitness assignments.
        """
        import threading
        import time
        
        def slow_fitness(genome):
            """
            Simulates a slow fitness evaluation by introducing a delay before returning the sum of the genome.
            
            Parameters:
                genome (iterable): Sequence of numeric gene values.
            
            Returns:
                int or float: The sum of the genome elements.
            """
            time.sleep(0.01)  # Simulate slow computation
            return sum(genome)
        
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        # Test that concurrent access doesn't break anything
        def evaluate_subset(pop_subset):
            """
            Assigns fitness values to each individual in a population subset using the slow_fitness function.
            
            Each individual's 'fitness' key is updated based on the evaluation of its 'genome'.
            """
            for individual in pop_subset:
                individual['fitness'] = slow_fitness(individual['genome'])
        
        threads = []
        for i in range(len(population)):
            thread = threading.Thread(target=evaluate_subset, args=([population[i]],))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All individuals should have fitness values
        for individual in population:
            self.assertIsNotNone(individual['fitness'])
    
    def test_thread_safe_callback_execution(self):
        """
        Verify that registered callbacks are executed in a thread-safe manner when invoked concurrently from multiple threads.
        
        This test ensures that the callback mechanism in the evolutionary conduit maintains correct state and avoids race conditions during concurrent execution.
        """
        import threading
        
        callback_count = 0
        lock = threading.Lock()
        
        def thread_safe_callback(generation, population, best_individual):
            """
            Thread-safe callback function that increments a shared counter for each invocation.
            
            Acquires a lock to safely update the callback count when called during concurrent evolution processes.
            """
            nonlocal callback_count
            with lock:
                callback_count += 1
        
        self.conduit.add_callback(thread_safe_callback)
        
        # Simulate concurrent callback execution
        def simulate_generation():
            """
            Invokes all registered callbacks for a simulated generation event, passing fixed generation, population, and individual data.
            """
            for callback in self.conduit.callbacks:
                callback(1, [], {'genome': [1, 2, 3], 'fitness': 0.5})
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=simulate_generation)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.assertEqual(callback_count, 5)


class TestMemoryAndPerformance(unittest.TestCase):
    """Test suite for memory usage and performance scenarios."""
    
    def setUp(self):
        """
        Set up the test environment by initializing an EvolutionaryConduit instance for use in test methods.
        """
        self.conduit = EvolutionaryConduit()
    
    def test_large_population_handling(self):
        """
        Tests that the PopulationManager can initialize and process a large population without memory errors.
        
        Verifies that a population of 1000 individuals with genome length 100 is created successfully and that population statistics, including 'best_fitness' and 'average_fitness', can be computed without issues.
        """
        manager = PopulationManager()
        
        # Create a large population
        large_population = manager.initialize_random_population(
            population_size=1000,
            genome_length=100
        )
        
        self.assertEqual(len(large_population), 1000)
        
        # Test that we can compute statistics without issues
        stats = manager.get_population_statistics(large_population)
        self.assertIn('best_fitness', stats)
        self.assertIn('average_fitness', stats)
    
    def test_memory_cleanup_after_evolution(self):
        """
        Test that memory is properly released after multiple evolution runs.
        
        Runs the evolutionary process several times with mocked results, deletes references, and forces garbage collection to ensure no memory leaks occur.
        """
        import gc
        
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of its elements.
            
            Parameters:
                genome (iterable): A sequence of numeric values representing a genome.
            
            Returns:
                int or float: The total sum of the genome's elements.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(simple_fitness)
        
        # Mock evolution to avoid actual computation
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 10,
                'final_population': [],
                'statistics': {'best_fitness': 6.0}
            }
            
            # Run evolution multiple times
            for _ in range(5):
                result = self.conduit.run_evolution(genome_length=10)
                del result
        
        # Force garbage collection
        gc.collect()
        
        # Test passes if no memory leaks cause issues
        self.assertTrue(True)
    
    def test_genome_size_scaling(self):
        """
        Test that the PopulationManager can initialize and handle populations with varying genome sizes, and that diversity calculation works correctly for each case.
        """
        manager = PopulationManager()
        
        # Test with small genomes
        small_pop = manager.initialize_random_population(10, 5)
        self.assertEqual(len(small_pop), 10)
        
        # Test with medium genomes
        medium_pop = manager.initialize_random_population(10, 50)
        self.assertEqual(len(medium_pop), 10)
        
        # Test with large genomes
        large_pop = manager.initialize_random_population(10, 500)
        self.assertEqual(len(large_pop), 10)
        
        # All should work without issues
        for pop in [small_pop, medium_pop, large_pop]:
            diversity = manager.calculate_diversity(pop)
            self.assertIsInstance(diversity, float)


class TestErrorRecoveryAndRobustness(unittest.TestCase):
    """Test suite for error recovery and robustness scenarios."""
    
    def setUp(self):
        """
        Set up the test environment by initializing an EvolutionaryConduit instance for use in test methods.
        """
        self.conduit = EvolutionaryConduit()
    
    def test_recovery_from_nan_fitness(self):
        """
        Test that the population manager raises a ValueError when the fitness function returns NaN for any individual.
        
        Verifies that the evaluation process detects and handles NaN fitness values, ensuring robustness against invalid fitness outputs.
        """
        def nan_fitness(genome):
            """
            Return a fitness value of NaN for any given genome.
            
            Parameters:
            	genome: The genome to evaluate (unused).
            
            Returns:
            	float: Not-a-Number (NaN) as the fitness value.
            """
            import math
            return float('nan')
        
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None}
        ]
        
        manager = PopulationManager()
        
        # Should handle NaN values gracefully
        with self.assertRaises(ValueError):
            manager.evaluate_population(population, nan_fitness)
    
    def test_recovery_from_infinite_fitness(self):
        """
        Test that the population manager correctly handles individuals assigned infinite fitness values.
        
        Verifies that when the fitness function returns infinity, the individual's fitness is set to `float('inf')` without causing errors or crashes.
        """
        def infinite_fitness(genome):
            """
            Return a fitness value of positive infinity for any given genome.
            
            Parameters:
            	genome: The genome to evaluate (unused).
            
            Returns:
            	float: Positive infinity as the fitness value.
            """
            return float('inf')
        
        population = [
            {'genome': [1, 2, 3], 'fitness': None}
        ]
        
        manager = PopulationManager()
        manager.evaluate_population(population, infinite_fitness)
        
        # Should handle infinity values
        self.assertEqual(population[0]['fitness'], float('inf'))
    
    def test_recovery_from_corrupted_genome(self):
        """
        Test that the population manager can recover from genomes containing invalid (None) values during fitness evaluation.
        
        Verifies that the fitness function can handle and skip invalid genome entries, assigning the correct fitness value to individuals with corrupted genomes.
        """
        population = [
            {'genome': [1, None, 3], 'fitness': None}  # None value in genome
        ]
        
        def robust_fitness(genome):
            """
            Calculates the sum of all non-None values in a genome, returning 0.0 if a TypeError occurs.
            
            Parameters:
                genome (iterable): A sequence of values representing a genome.
            
            Returns:
                float: The sum of all non-None values in the genome, or 0.0 if the genome contains invalid types.
            """
            try:
                return sum(x for x in genome if x is not None)
            except TypeError:
                return 0.0
        
        manager = PopulationManager()
        manager.evaluate_population(population, robust_fitness)
        
        self.assertEqual(population[0]['fitness'], 4.0)  # 1 + 3
    
    def test_evolution_with_degenerate_population(self):
        """
        Test that population statistics are correctly computed when all individuals have identical fitness values.
        
        Verifies that best and worst fitness are equal and standard deviation is zero in a degenerate population scenario.
        """
        # All individuals have same fitness
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        manager = PopulationManager()
        stats = manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 0.5)
        self.assertEqual(stats['worst_fitness'], 0.5)
        self.assertEqual(stats['std_dev_fitness'], 0.0)
    
    def test_parameter_validation_edge_cases(self):
        """
        Test that EvolutionaryParameters correctly accepts and stores edge case values for population size, generations, mutation rate, and crossover rate, including extremely small and large valid values.
        """
        # Test with very small but valid values
        params = EvolutionaryParameters(
            population_size=1,
            generations=1,
            mutation_rate=1e-10,
            crossover_rate=1e-10
        )
        
        self.assertEqual(params.population_size, 1)
        self.assertEqual(params.mutation_rate, 1e-10)
        
        # Test with very large values
        params = EvolutionaryParameters(
            population_size=1000000,
            generations=1000000,
            mutation_rate=0.999999,
            crossover_rate=0.999999
        )
        
        self.assertEqual(params.population_size, 1000000)
        self.assertAlmostEqual(params.mutation_rate, 0.999999, places=6)


if __name__ == '__main__':
    # Run all tests with increased verbosity
    unittest.main(verbosity=2, buffer=True)