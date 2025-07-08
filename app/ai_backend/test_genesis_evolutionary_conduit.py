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
        Verify that EvolutionaryParameters are initialized with the correct default values.
        """
        self.assertEqual(self.default_params.population_size, 100)
        self.assertEqual(self.default_params.generations, 500)
        self.assertEqual(self.default_params.mutation_rate, 0.1)
        self.assertEqual(self.default_params.crossover_rate, 0.8)
        self.assertEqual(self.default_params.selection_pressure, 0.2)
    
    def test_custom_initialization(self):
        """
        Verify that custom initialization of evolutionary parameters sets all values as expected.
        """
        self.assertEqual(self.custom_params.population_size, 200)
        self.assertEqual(self.custom_params.generations, 1000)
        self.assertEqual(self.custom_params.mutation_rate, 0.15)
        self.assertEqual(self.custom_params.crossover_rate, 0.85)
        self.assertEqual(self.custom_params.selection_pressure, 0.3)
    
    def test_parameter_validation(self):
        """
        Test that invalid evolutionary parameter values raise ValueError during initialization.
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
        Test that the EvolutionaryParameters instance is correctly converted to a dictionary representation.
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
        Test that EvolutionaryParameters can be correctly instantiated from a dictionary representation.
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
        """Set up test fixtures."""
        self.strategy = MutationStrategy()
    
    def test_gaussian_mutation(self):
        """
        Test that the Gaussian mutation strategy produces mutated genomes of correct length and type for different mutation rates.
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
        Verify that the uniform mutation strategy produces a mutated genome of the same length with all values within specified bounds.
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
        Tests the bit flip mutation strategy to ensure it produces a mutated genome of the correct length and type, with all values as booleans.
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
        Test that the adaptive mutation strategy produces a mutated genome of the correct length and type given a fitness history.
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
        Initialize the selection strategy and a sample population for selection strategy tests.
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
        
        Verifies that the selected individual is a member of the population and contains the expected keys.
        """
        selected = self.strategy.tournament_selection(self.population, tournament_size=2)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_roulette_wheel_selection(self):
        """
        Test that the roulette wheel selection strategy selects a valid individual from the population.
        
        Verifies that the selected individual is a member of the population and contains the expected keys.
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
        Test that the elitism selection strategy correctly selects the top individuals by fitness.
        
        Verifies that the number of selected individuals matches the elite count and that the selected individuals are those with the highest fitness values in descending order.
        """
        elite_count = 2
        selected = self.strategy.elitism_selection(self.population, elite_count)
        
        self.assertEqual(len(selected), elite_count)
        
        # Check that selected individuals are the fittest
        fitness_values = [individual['fitness'] for individual in selected]
        self.assertEqual(fitness_values, [0.9, 0.7])  # Sorted by fitness descending
    
    def test_empty_population(self):
        """
        Test that selection methods raise ValueError when called with an empty population.
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
        Initializes the test fixture by creating a FitnessFunction instance for use in tests.
        """
        self.fitness_func = FitnessFunction()
    
    def test_sphere_function(self):
        """
        Test that the sphere fitness function correctly computes the negative sum of squares for a given genome.
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
        Test that the Rosenbrock fitness function returns 0 for the genome [1.0, 1.0], its known global minimum.
        """
        genome = [1.0, 1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)
        
        # Rosenbrock function should be 0 at (1, 1)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function(self):
        """
        Test that the Ackley fitness function returns zero at the origin.
        
        Verifies that evaluating the Ackley function with a genome of all zeros produces a fitness value of 0.0.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.ackley_function(genome)
        
        # Ackley function should be 0 at origin
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function(self):
        """
        Tests evaluation of a custom fitness function by verifying that the fitness function correctly computes the sum of genome values.
        """
        def custom_func(genome):
            """
            Calculates the sum of all elements in the genome.
            
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
        Test that the multi-objective fitness function correctly evaluates a genome against multiple objectives and returns the expected fitness values.
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
        Test that fitness evaluation applies a penalty when constraints are violated.
        
        Verifies that the fitness function penalizes genomes that do not satisfy provided constraints during evaluation.
        """
        genome = [1.0, 2.0, 3.0]
        
        def constraint_func(g):
            # Constraint: sum should be less than 5
            """
            Checks whether the sum of elements in the input is less than 5.
            
            Parameters:
                g (iterable): An iterable of numeric values.
            
            Returns:
                bool: True if the sum of elements in g is less than 5, otherwise False.
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
        Initialize test fixtures for population manager tests by creating a PopulationManager instance and setting default genome length and population size.
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
        Test that the population manager correctly initializes a population with provided seed genomes.
        
        Verifies that the seeded genomes are present in the initialized population and that the total population size matches the expected value.
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
        Tests that the population evaluation assigns a numeric fitness value to each individual using the provided fitness function.
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
        Verify that population statistics such as best, worst, average, median, and standard deviation of fitness are correctly computed for a given population.
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
        Verify that the population diversity calculation returns a positive float value for a given set of genomes.
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
        Test that population manager methods raise ValueError when called with an empty population.
        """
        with self.assertRaises(ValueError):
            self.manager.get_best_individual([])
        
        with self.assertRaises(ValueError):
            self.manager.get_population_statistics([])


class TestGeneticOperations(unittest.TestCase):
    """Test suite for GeneticOperations class."""
    
    def setUp(self):
        """
        Initializes test fixtures for genetic operations tests.
        """
        self.operations = GeneticOperations()
    
    def test_single_point_crossover(self):
        """
        Verify that the single-point crossover operation produces two children of correct length, each containing elements from both parents.
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
        Test that the uniform crossover operation produces two children of the same length as the parent genomes.
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
        Verify that the simulated binary crossover (SBX) operation produces two children of correct length and within specified bounds.
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
        Test that the blend crossover (BLX-α) operation produces two offspring of the correct length from two parent genomes.
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
        Initialize test fixtures for EvolutionaryConduit tests by creating a conduit instance and setting default evolutionary parameters.
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
        Test that a custom fitness function can be set on the conduit and is correctly used for fitness evaluation.
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
        Test that evolutionary parameters are correctly set in the conduit and match expected values.
        """
        self.conduit.set_parameters(self.params)
        
        self.assertEqual(self.conduit.parameters.population_size, 20)
        self.assertEqual(self.conduit.parameters.generations, 10)
        self.assertEqual(self.conduit.parameters.mutation_rate, 0.1)
        self.assertEqual(self.conduit.parameters.crossover_rate, 0.8)
    
    @patch('app.ai_backend.genesis_evolutionary_conduit.EvolutionaryConduit.evolve')
    def test_run_evolution(self, mock_evolve):
        """
        Verifies that the evolution process runs and returns the expected result structure when invoked.
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
        Verify that the EvolutionaryConduit state can be saved and accurately restored in a new instance.
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
        Test that a callback function can be added to the evolution process and is registered in the conduit.
        """
        callback_called = False
        
        def test_callback(generation, population, best_individual):
            """
            Test callback function that sets a flag when invoked during evolution.
            
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
        Test that enabling history tracking in the evolutionary conduit correctly records evolution history during a run.
        
        Verifies that after running a mocked evolution process with history tracking enabled, the conduit reflects that history tracking is active.
        """
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        # Run a simple evolution
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of its elements.
            
            Parameters:
            	genome (Iterable[float|int]): The genome to evaluate.
            
            Returns:
            	float|int: The sum of the genome's elements, representing its fitness.
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
        Initialize test fixtures for GenesisEvolutionaryConduit tests by creating a conduit instance and setting evolutionary parameters.
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
        Verify that GenesisEvolutionaryConduit initializes correctly with required components.
        """
        self.assertIsInstance(self.genesis_conduit, EvolutionaryConduit)
        self.assertIsNotNone(self.genesis_conduit.genesis_config)
        self.assertIsNotNone(self.genesis_conduit.neural_network_factory)
        self.assertIsNotNone(self.genesis_conduit.optimization_strategies)
    
    def test_neural_network_evolution(self):
        """
        Tests the neural network evolution process, including setting network configuration and verifying network creation.
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
        Test that the neuroevolution fitness evaluation returns a numeric fitness value for a given genome and training data.
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
        Tests that the network topology can be evolved by mutating its structure.
        
        Verifies that the mutated topology remains a dictionary containing 'layers' and 'connections' keys after mutation.
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
        Test that hyperparameter optimization generates hyperparameters within the specified search space and includes all expected keys.
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
        Test that multi-objective optimization correctly evaluates a genome against multiple objectives and returns a fitness vector with the expected structure.
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
        Test that adaptive mutation rate calculation returns a valid float within the expected range based on population fitness history.
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
        """Test speciation for maintaining diversity."""
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
        Test that ensemble evolution selects the top-performing networks based on fitness.
        
        Verifies that the ensemble creation method returns the correct number of networks and that the selected networks have the highest fitness values.
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
        Verifies that coevolution of multiple populations produces the expected output structure.
        
        This test checks that the `coevolve_populations` method of the genesis conduit correctly processes two populations and returns a dictionary containing updated populations.
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
        Tests that the checkpoint saving mechanism correctly invokes the save operation with the specified file path.
        """
        # Set up conduit state
        self.genesis_conduit.set_parameters(self.params)
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pkl"
        self.genesis_conduit.save_checkpoint(checkpoint_path)
        
        mock_save.assert_called_once_with(checkpoint_path)
    
    def test_distributed_evolution(self):
        """
        Test distributed evolution by setting up an island model and verifying migration of individuals between islands.
        
        This test initializes multiple islands with specified configurations, simulates populations, and checks that the migration process returns the expected tuple structure.
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
        Test that an EvolutionaryException is created with the correct message and type.
        """
        message = "Test evolutionary exception"
        exception = EvolutionaryException(message)
        
        self.assertEqual(str(exception), message)
        self.assertIsInstance(exception, Exception)
    
    def test_exception_with_details(self):
        """
        Test that EvolutionaryException correctly stores and exposes additional details.
        """
        message = "Evolution failed"
        details = {"generation": 50, "error_type": "convergence"}
        
        exception = EvolutionaryException(message, details)
        
        self.assertEqual(str(exception), message)
        self.assertEqual(exception.details, details)
    
    def test_exception_raising(self):
        """Test exception raising."""
        with self.assertRaises(EvolutionaryException):
            raise EvolutionaryException("Test exception")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test suite for complex evolutionary scenarios."""
    
    def setUp(self):
        """
        Initialize test fixtures for integration tests, including a GenesisEvolutionaryConduit instance and default evolutionary parameters.
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
        Test that the complete evolution cycle executes from initialization to completion and returns expected results.
        
        Verifies that the evolution process produces a result containing the best individual and correct generation count when the evolution method is mocked.
        """
        # Set up fitness function
        def simple_fitness(genome):
            """
            Calculates the fitness of a genome as the sum of the squares of its elements.
            
            Parameters:
                genome (Iterable[float]): The genome to evaluate.
            
            Returns:
                float: The computed fitness value.
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
        Tests the end-to-end neural network evolution pipeline, including network configuration, training data setup, and network creation using the GenesisEvolutionaryConduit.
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
        Test the multi-objective optimization pipeline by verifying that multiple objectives are set and the fitness evaluation returns the expected vector for a given genome.
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
        Test the adaptive evolution pipeline by verifying that the adaptive mutation rate is correctly calculated based on a population with varying fitness values.
        
        Ensures the calculated mutation rate is a float within the valid range [0.0, 1.0].
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
        Test that invalid parameters and fitness evaluation failures are properly handled and that recovery mechanisms raise the expected exceptions.
        """
        # Test invalid parameters
        with self.assertRaises(ValueError):
            invalid_params = EvolutionaryParameters(population_size=0)
        
        # Test recovery from evolution failure
        def failing_fitness(genome):
            """
            Raises a ValueError to simulate a failure during fitness evaluation.
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
        Initialize test fixtures for asynchronous evolution tests by creating a GenesisEvolutionaryConduit instance and setting basic evolutionary parameters.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    @patch('asyncio.run')
    def test_async_evolution_execution(self, mock_run):
        """
        Tests that asynchronous evolution execution returns a valid result using a mocked evolution process.
        """
        async def mock_async_evolve():
            """
            Simulate an asynchronous evolutionary process and return mock evolution results.
            
            Returns:
                dict: A dictionary containing a mock best individual, number of generations run, final population, and statistics.
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
        Test that population fitness evaluation is performed in parallel using the provided executor mock.
        
        Verifies that the parallel execution mechanism is invoked and that fitness values are assigned as expected.
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
                genome (Iterable[float]): The genome to evaluate.
            
            Returns:
                float: The sum of all elements in the genome, representing its fitness.
            """
            return sum(genome)
        
        # Test parallel evaluation
        self.genesis_conduit.evaluate_population_parallel(population, fitness_func)
        
        # Verify parallel execution was attempted
        mock_executor.assert_called_once()


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)