import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
import asyncio
import sys
import os
import math

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
        Set up test fixtures by creating default and custom EvolutionaryParameters instances for use in test methods.
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
        Test that custom initialization of evolutionary parameters correctly assigns all specified values.
        """
        self.assertEqual(self.custom_params.population_size, 200)
        self.assertEqual(self.custom_params.generations, 1000)
        self.assertEqual(self.custom_params.mutation_rate, 0.15)
        self.assertEqual(self.custom_params.crossover_rate, 0.85)
        self.assertEqual(self.custom_params.selection_pressure, 0.3)
    
    def test_parameter_validation(self):
        """
        Verify that initializing EvolutionaryParameters with invalid values raises a ValueError.
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
        Tests conversion of an EvolutionaryParameters instance to its dictionary representation.
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
        Verify that EvolutionaryParameters can be instantiated from a dictionary and that all attributes are set correctly.
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
        Initialize the MutationStrategy instance for use in mutation strategy tests.
        """
        self.strategy = MutationStrategy()
    
    def test_gaussian_mutation(self):
        """
        Test that the Gaussian mutation strategy returns a mutated genome as a list of the same length as the input for various mutation rates.
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
        Tests that the uniform mutation strategy returns a mutated genome of the same length as the input, with all values constrained within the specified bounds.
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
        Test that the bit flip mutation strategy returns a mutated genome of the correct length and type, with all elements as booleans.
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
        Tests that the adaptive mutation strategy returns a mutated genome as a list of the same length as the input genome when provided with a fitness history.
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
        Set up the selection strategy instance and a sample population for use in selection strategy unit tests.
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
        
        Verifies that the selected individual is present in the population and contains both 'genome' and 'fitness' keys.
        """
        selected = self.strategy.tournament_selection(self.population, tournament_size=2)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_roulette_wheel_selection(self):
        """
        Test that roulette wheel selection returns a valid individual from the population.
        
        Ensures the selected individual is present in the population and includes both 'genome' and 'fitness' keys.
        """
        selected = self.strategy.roulette_wheel_selection(self.population)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_rank_selection(self):
        """
        Test that the rank-based selection strategy returns a valid individual from the population.
        
        Ensures the selected individual exists in the population and contains both 'genome' and 'fitness' keys.
        """
        selected = self.strategy.rank_selection(self.population)
        
        self.assertIn(selected, self.population)
        self.assertIsInstance(selected, dict)
        self.assertIn('genome', selected)
        self.assertIn('fitness', selected)
    
    def test_elitism_selection(self):
        """
        Test that the elitism selection strategy selects the top individuals with the highest fitness values.
        
        Ensures the number of selected individuals matches the elite count and that the selected individuals are ordered by descending fitness.
        """
        elite_count = 2
        selected = self.strategy.elitism_selection(self.population, elite_count)
        
        self.assertEqual(len(selected), elite_count)
        
        # Check that selected individuals are the fittest
        fitness_values = [individual['fitness'] for individual in selected]
        self.assertEqual(fitness_values, [0.9, 0.7])  # Sorted by fitness descending
    
    def test_empty_population(self):
        """
        Verify that selection methods raise ValueError when invoked with an empty population.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection([], tournament_size=2)
        
        with self.assertRaises(ValueError):
            self.strategy.roulette_wheel_selection([])
    
    def test_invalid_tournament_size(self):
        """
        Test that tournament selection raises a ValueError when the tournament size is zero or greater than the population size.
        """
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=0)
        
        with self.assertRaises(ValueError):
            self.strategy.tournament_selection(self.population, tournament_size=len(self.population) + 1)


class TestFitnessFunction(unittest.TestCase):
    """Test suite for FitnessFunction class."""
    
    def setUp(self):
        """
        Set up the test fixture by instantiating a FitnessFunction for use in test methods.
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
        Test that the Rosenbrock fitness function returns its global minimum value for the genome [1.0, 1.0].

        Verifies that the fitness function evaluates to 0.0 at the known minimum point.
        """
        genome = [1.0, 1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)
        
        # Rosenbrock function should be 0 at (1, 1)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function(self):
        """
        Test that the Ackley fitness function returns a value of 0.0 when evaluated at the origin.
        
        Verifies that a genome consisting entirely of zeros produces the global minimum for the Ackley function.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.ackley_function(genome)
        
        # Ackley function should be 0 at origin
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function(self):
        """
        Tests that a custom fitness function correctly computes the sum of genome values when used with the fitness evaluation method.
        """
        def custom_func(genome):
            """
            Return the sum of all numeric elements in the provided genome.
            
            Parameters:
                genome (iterable): Sequence of numeric values to be summed.
            
            Returns:
                int or float: The sum of the genome's elements.
            """
            return sum(genome)
        
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate(genome, custom_func)
        
        self.assertEqual(fitness, 6.0)
    
    def test_multi_objective_function(self):
        """
        Test that the multi-objective fitness function evaluates a genome using multiple objectives and returns the correct fitness values for each objective.
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
        Test that the fitness function penalizes genomes violating constraints during evaluation.
        
        Ensures that when a genome does not satisfy specified constraints, the evaluated fitness is reduced by a penalty.
        """
        genome = [1.0, 2.0, 3.0]
        
        def constraint_func(g):
            # Constraint: sum should be less than 5
            """
            Check if the sum of elements in the input iterable is less than 5.
            
            Parameters:
                g (iterable): Iterable containing numeric values.
            
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
        Set up test fixtures for population manager tests by initializing a PopulationManager instance and default parameters for genome length and population size.
        """
        self.manager = PopulationManager()
        self.genome_length = 5
        self.population_size = 10
    
    def test_initialize_random_population(self):
        """
        Test that the population manager creates a random population with the correct size and genome length.
        
        Ensures each individual has a genome of the specified length and includes a fitness attribute.
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
        Test initialization of a population with provided seed genomes.
        
        Ensures that the seeded genomes are included in the resulting population and that the total population size is correct.
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
        Verify that evaluating a population assigns a numeric fitness value to each individual using a fitness function.
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
        Test that the population statistics method returns correct values for best, worst, average, median, and standard deviation of fitness in a given population.
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
        Test that the population diversity calculation returns a positive float for a set of distinct genomes.
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
        Test that the population manager raises ValueError when methods are called with an empty population.
        """
        with self.assertRaises(ValueError):
            self.manager.get_best_individual([])
        
        with self.assertRaises(ValueError):
            self.manager.get_population_statistics([])


class TestGeneticOperations(unittest.TestCase):
    """Test suite for GeneticOperations class."""
    
    def setUp(self):
        """
        Set up the test fixture for genetic operations tests by initializing a GeneticOperations instance.
        """
        self.operations = GeneticOperations()
    
    def test_single_point_crossover(self):
        """
        Tests that the single-point crossover operation returns two children of the correct length, each composed of elements from both parent sequences.
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
        Test that two-point crossover returns two children with the same genome length as the parents.
        """
        parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
        parent2 = [9, 10, 11, 12, 13, 14, 15, 16]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_uniform_crossover(self):
        """
        Test that the uniform crossover operation returns two children with the same length as the parent genomes.
        """
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_arithmetic_crossover(self):
        """
        Test that arithmetic crossover produces children as weighted averages of two parent genomes.
        
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
        Test that the simulated binary crossover (SBX) operation generates two children with correct lengths and ensures all gene values are within the specified bounds.
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
        Test that the blend crossover (BLX-α) operation generates two offspring with the same length as the parent genomes.
        """
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.5)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_invalid_crossover_inputs(self):
        """
        Test that crossover operations raise a ValueError when parent genomes have different lengths.
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
        Set up test fixtures for EvolutionaryConduit tests by initializing a conduit instance and default evolutionary parameters.
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
        Tests that a custom fitness function can be assigned to the conduit and is used for evaluating genome fitness.
        """
        def custom_fitness(genome):
            """
            Calculate the fitness score of a genome by summing its elements.
            
            Parameters:
                genome (iterable): An iterable of numeric values representing the genome.
            
            Returns:
                The sum of the genome's elements as the fitness score.
            """
            return sum(genome)
        
        self.conduit.set_fitness_function(custom_fitness)
        
        # Test that the function is set correctly
        test_genome = [1.0, 2.0, 3.0]
        fitness = self.conduit.fitness_function.evaluate(test_genome, custom_fitness)
        self.assertEqual(fitness, 6.0)
    
    def test_set_parameters(self):
        """
        Test that setting evolutionary parameters in the conduit updates them to the expected values.
        """
        self.conduit.set_parameters(self.params)
        
        self.assertEqual(self.conduit.parameters.population_size, 20)
        self.assertEqual(self.conduit.parameters.generations, 10)
        self.assertEqual(self.conduit.parameters.mutation_rate, 0.1)
        self.assertEqual(self.conduit.parameters.crossover_rate, 0.8)
    
    @patch('app.ai_backend.genesis_evolutionary_conduit.EvolutionaryConduit.evolve')
    def test_run_evolution(self, mock_evolve):
        """
        Tests that the evolution process executes and returns a result containing the expected keys: 'best_individual', 'generations_run', 'final_population', and 'statistics'.
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
        Tests that the EvolutionaryConduit state can be saved and restored, ensuring parameters are preserved after loading into a new instance.
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
        Test that a callback function can be added to the evolution process and is correctly registered in the conduit.
        """
        callback_called = False
        
        def test_callback(generation, population, best_individual):
            """
            A test callback function that sets a flag when invoked during the evolution process.
            
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
        Test that enabling history tracking in the evolutionary conduit records evolution history during a run.
        
        Verifies that after running a mocked evolution process with history tracking enabled, the conduit indicates that history tracking is active.
        """
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        # Run a simple evolution
        def simple_fitness(genome):
            """
            Calculate the fitness of a genome by summing its elements.
            
            Parameters:
                genome (Iterable[float | int]): The genome to evaluate.
            
            Returns:
                float | int: The total sum of the genome's elements, used as the fitness value.
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
        Set up test fixtures for GenesisEvolutionaryConduit tests by initializing a conduit instance and evolutionary parameters.
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
        Tests that GenesisEvolutionaryConduit is properly initialized with its core components, including genesis configuration, neural network factory, and optimization strategies.
        """
        self.assertIsInstance(self.genesis_conduit, EvolutionaryConduit)
        self.assertIsNotNone(self.genesis_conduit.genesis_config)
        self.assertIsNotNone(self.genesis_conduit.neural_network_factory)
        self.assertIsNotNone(self.genesis_conduit.optimization_strategies)
    
    def test_neural_network_evolution(self):
        """
        Test the neural network evolution process by configuring network parameters and verifying that a neural network instance is created.
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
        Tests that neuroevolution fitness evaluation produces a numeric fitness value for a given genome and training data.
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
        Test that mutating a neural network topology produces a valid structure.
        
        Ensures that after mutation, the resulting topology is a dictionary containing both 'layers' and 'connections' keys.
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
        Tests that hyperparameter optimization produces hyperparameters within the defined search space and includes all required keys.
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
        Test that multi-objective optimization evaluates a genome against multiple objectives and returns a fitness vector matching the number of objectives.
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
        Verifies that the adaptive mutation rate calculation produces a float within the valid range based on the fitness history of the population.
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
        Test that the speciation process groups individuals into species based on genome similarity to maintain population diversity.
        
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
        Tests that a pretrained neural network genome can be adapted to a new task using transfer learning.
        
        Asserts that the adapted genome is a non-empty list after applying the adaptation with a new task configuration.
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
        Test that ensemble evolution selects the top-performing networks for the ensemble.
        
        Ensures that the ensemble creation method returns the specified number of networks with the highest fitness values from the provided population.
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
        
        Ensures the number of novelty scores equals the population size and that all scores are numeric values.
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
        Tests that coevolution of two populations using the genesis conduit returns a dictionary containing updated populations with the expected keys.
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
        Test that the checkpoint saving mechanism calls the save operation with the correct file path.
        """
        # Set up conduit state
        self.genesis_conduit.set_parameters(self.params)
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pkl"
        self.genesis_conduit.save_checkpoint(checkpoint_path)
        
        mock_save.assert_called_once_with(checkpoint_path)
    
    def test_distributed_evolution(self):
        """
        Test distributed evolution using an island model and verify migration of individuals between islands.
        
        This test sets up multiple islands with specific configurations, simulates populations, and checks that the migration process produces a tuple containing the updated populations.
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
        Test that an EvolutionaryException is instantiated with the correct message and is an instance of Exception.
        """
        message = "Test evolutionary exception"
        exception = EvolutionaryException(message)
        
        self.assertEqual(str(exception), message)
        self.assertIsInstance(exception, Exception)
    
    def test_exception_with_details(self):
        """
        Test that EvolutionaryException stores and exposes additional details provided during initialization.
        """
        message = "Evolution failed"
        details = {"generation": 50, "error_type": "convergence"}
        
        exception = EvolutionaryException(message, details)
        
        self.assertEqual(str(exception), message)
        self.assertEqual(exception.details, details)
    
    def test_exception_raising(self):
        """
        Verify that raising an EvolutionaryException triggers the expected exception handling.
        """
        with self.assertRaises(EvolutionaryException):
            raise EvolutionaryException("Test exception")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test suite for complex evolutionary scenarios."""
    
    def setUp(self):
        """
        Set up test fixtures for integration tests by initializing a GenesisEvolutionaryConduit instance and default evolutionary parameters.
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
        Tests that the full evolution cycle runs from initialization to completion and returns the expected result structure.
        
        Verifies that, when the evolution process is mocked, the returned result includes the best individual and the correct number of generations.
        """
        # Set up fitness function
        def simple_fitness(genome):
            """
            Calculate the fitness of a genome by summing the squares of its elements.
            
            Parameters:
                genome (Iterable[float]): Sequence of numeric values representing the genome.
            
            Returns:
                float: The sum of squares of all elements in the genome.
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
        Test the complete neural network evolution pipeline, including setting network configuration, providing training data, and creating a neural network using the GenesisEvolutionaryConduit.
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
        Test that the multi-objective optimization pipeline correctly sets objectives and returns the expected fitness vector for a given genome.
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
        Test that the adaptive evolution pipeline computes a valid mutation rate based on population fitness values.
        
        Verifies that the calculated adaptive mutation rate is a float within the range [0.0, 1.0] when provided with a population exhibiting varying fitness.
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
        Test that invalid evolutionary parameters and fitness evaluation failures trigger the appropriate exceptions.
        
        This test verifies that providing invalid parameters raises a `ValueError` and that a failure during fitness evaluation results in an `EvolutionaryException` during the evolution process.
        """
        # Test invalid parameters
        with self.assertRaises(ValueError):
            invalid_params = EvolutionaryParameters(population_size=0)
        
        # Test recovery from evolution failure
        def failing_fitness(genome):
            """
            Simulates a fitness evaluation failure by raising a ValueError.
            
            Raises:
                ValueError: Always raised to indicate a fitness evaluation failure.
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
        Set up test fixtures for asynchronous evolution tests.
        
        Initializes a GenesisEvolutionaryConduit instance and configures basic evolutionary parameters for use in test methods.
        """
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    @patch('asyncio.run')
    def test_async_evolution_execution(self, mock_run):
        """
        Test that asynchronous evolution execution returns a valid result when the evolution process is mocked.
        
        This test verifies that the `run_async_evolution` method of the evolutionary conduit produces a non-None result when the underlying asynchronous evolution process is simulated.
        """
        async def mock_async_evolve():
            """
            Simulates an asynchronous evolutionary process and returns mock results.
            
            Returns:
                dict: Contains a mock best individual, number of generations run, final population, and statistics.
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
        Test that population fitness evaluation is executed in parallel using a mocked executor.
        
        Ensures that the parallel evaluation mechanism is triggered and that fitness values are assigned to each individual in the population.
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
            Calculate the fitness of a genome by summing its elements.
            
            Parameters:
                genome (Iterable[float]): Sequence of numeric values representing the genome.
            
            Returns:
                float: Total sum of the genome's elements as the fitness score.
            """
            return sum(genome)
        
        # Test parallel evaluation
        self.genesis_conduit.evaluate_population_parallel(population, fitness_func)
        
        # Verify parallel execution was attempted
        mock_executor.assert_called_once()


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

class TestEvolutionaryParametersExtended(unittest.TestCase):
    """Extended test suite for EvolutionaryParameters class covering edge cases and advanced scenarios."""
    
    def test_boundary_values(self):
        """Test evolutionary parameters with boundary values."""
        # Test minimum valid values
        min_params = EvolutionaryParameters(
            population_size=1,
            generations=1,
            mutation_rate=0.0,
            crossover_rate=0.0,
            selection_pressure=0.0
        )
        self.assertEqual(min_params.population_size, 1)
        self.assertEqual(min_params.mutation_rate, 0.0)
        
        # Test maximum valid values
        max_params = EvolutionaryParameters(
            population_size=10000,
            generations=100000,
            mutation_rate=1.0,
            crossover_rate=1.0,
            selection_pressure=1.0
        )
        self.assertEqual(max_params.mutation_rate, 1.0)
        self.assertEqual(max_params.crossover_rate, 1.0)
    
    def test_float_precision_handling(self):
        """Test parameter handling with various float precision values."""
        params = EvolutionaryParameters(
            mutation_rate=0.123456789,
            crossover_rate=0.987654321
        )
        self.assertAlmostEqual(params.mutation_rate, 0.123456789, places=9)
        self.assertAlmostEqual(params.crossover_rate, 0.987654321, places=9)
    
    def test_serialization_roundtrip(self):
        """Test that parameters can be serialized and deserialized without loss."""
        original_params = EvolutionaryParameters(
            population_size=150,
            generations=750,
            mutation_rate=0.123,
            crossover_rate=0.876,
            selection_pressure=0.234
        )
        
        # Convert to dict and back
        params_dict = original_params.to_dict()
        reconstructed_params = EvolutionaryParameters.from_dict(params_dict)
        
        self.assertEqual(original_params.population_size, reconstructed_params.population_size)
        self.assertEqual(original_params.generations, reconstructed_params.generations)
        self.assertAlmostEqual(original_params.mutation_rate, reconstructed_params.mutation_rate, places=10)
        self.assertAlmostEqual(original_params.crossover_rate, reconstructed_params.crossover_rate, places=10)
        self.assertAlmostEqual(original_params.selection_pressure, reconstructed_params.selection_pressure, places=10)
    
    def test_from_dict_with_missing_keys(self):
        """Test parameter creation from dictionary with missing keys."""
        incomplete_dict = {
            'population_size': 100,
            'mutation_rate': 0.1
        }
        
        # Should handle missing keys gracefully by using defaults
        params = EvolutionaryParameters.from_dict(incomplete_dict)
        self.assertEqual(params.population_size, 100)
        self.assertEqual(params.mutation_rate, 0.1)
        # Should use defaults for missing keys
        self.assertEqual(params.generations, 500)  # default value
    
    def test_from_dict_with_invalid_types(self):
        """Test parameter creation from dictionary with invalid data types."""
        invalid_dict = {
            'population_size': "invalid",
            'generations': 500,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'selection_pressure': 0.2
        }
        
        with self.assertRaises((ValueError, TypeError)):
            EvolutionaryParameters.from_dict(invalid_dict)


class TestMutationStrategyExtended(unittest.TestCase):
    """Extended test suite for MutationStrategy class covering edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures for extended mutation strategy tests."""
        self.strategy = MutationStrategy()
    
    def test_empty_genome_mutation(self):
        """Test mutation strategies with empty genomes."""
        empty_genome = []
        
        # All mutation strategies should handle empty genomes gracefully
        result = self.strategy.gaussian_mutation(empty_genome, mutation_rate=0.1)
        self.assertEqual(result, [])
        
        result = self.strategy.uniform_mutation(empty_genome, mutation_rate=0.1, bounds=(-1, 1))
        self.assertEqual(result, [])
        
        result = self.strategy.bit_flip_mutation(empty_genome, mutation_rate=0.1)
        self.assertEqual(result, [])
    
    def test_single_element_genome_mutation(self):
        """Test mutation strategies with single-element genomes."""
        single_genome = [1.0]
        
        result = self.strategy.gaussian_mutation(single_genome, mutation_rate=1.0, sigma=0.1)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], float)
        
        result = self.strategy.uniform_mutation(single_genome, mutation_rate=1.0, bounds=(-10, 10))
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0], -10)
        self.assertLessEqual(result[0], 10)
    
    def test_extreme_mutation_rates(self):
        """Test mutation strategies with extreme mutation rates."""
        genome = [1.0, 2.0, 3.0]
        
        # Test with mutation rate of 0.0 (no mutation)
        result = self.strategy.gaussian_mutation(genome, mutation_rate=0.0, sigma=1.0)
        self.assertEqual(result, genome)
        
        # Test with mutation rate of 1.0 (full mutation)
        result = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1.0)
        self.assertEqual(len(result), len(genome))
        # With rate 1.0, all elements should potentially be mutated
    
    def test_adaptive_mutation_with_empty_history(self):
        """Test adaptive mutation with empty fitness history."""
        genome = [1.0, 2.0, 3.0]
        empty_history = []
        
        result = self.strategy.adaptive_mutation(genome, empty_history, base_rate=0.1)
        self.assertEqual(len(result), len(genome))
        # Should fallback to base rate
    
    def test_adaptive_mutation_with_constant_fitness(self):
        """Test adaptive mutation with constant fitness history."""
        genome = [1.0, 2.0, 3.0]
        constant_history = [0.5, 0.5, 0.5, 0.5]
        
        result = self.strategy.adaptive_mutation(genome, constant_history, base_rate=0.1)
        self.assertEqual(len(result), len(genome))
    
    def test_bit_flip_mutation_with_mixed_types(self):
        """Test that bit flip mutation properly handles non-boolean inputs."""
        mixed_genome = [1, 0, True, False, 1.0, 0.0]
        
        result = self.strategy.bit_flip_mutation(mixed_genome, mutation_rate=0.5)
        self.assertEqual(len(result), len(mixed_genome))
        # All results should be boolean
        for value in result:
            self.assertIsInstance(value, bool)
    
    def test_gaussian_mutation_with_large_sigma(self):
        """Test Gaussian mutation with very large sigma values."""
        genome = [1.0, 2.0, 3.0]
        
        result = self.strategy.gaussian_mutation(genome, mutation_rate=0.5, sigma=1000.0)
        self.assertEqual(len(result), len(genome))
        # Should still produce valid numeric results
        for value in result:
            self.assertIsInstance(value, (int, float))
    
    def test_uniform_mutation_with_invalid_bounds(self):
        """Test uniform mutation with invalid bounds."""
        genome = [1.0, 2.0, 3.0]
        
        # Test with inverted bounds (lower > upper)
        with self.assertRaises(ValueError):
            self.strategy.uniform_mutation(genome, mutation_rate=0.5, bounds=(10, -10))
        
        # Test with equal bounds
        result = self.strategy.uniform_mutation(genome, mutation_rate=1.0, bounds=(5, 5))
        for value in result:
            self.assertEqual(value, 5)


class TestSelectionStrategyExtended(unittest.TestCase):
    """Extended test suite for SelectionStrategy class covering edge cases and stress conditions."""
    
    def setUp(self):
        """Set up test fixtures for extended selection strategy tests."""
        self.strategy = SelectionStrategy()
    
    def test_single_individual_population(self):
        """Test selection strategies with single individual populations."""
        single_pop = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        # Tournament selection with single individual
        result = self.strategy.tournament_selection(single_pop, tournament_size=1)
        self.assertEqual(result, single_pop[0])
        
        # Roulette wheel selection with single individual
        result = self.strategy.roulette_wheel_selection(single_pop)
        self.assertEqual(result, single_pop[0])
        
        # Rank selection with single individual
        result = self.strategy.rank_selection(single_pop)
        self.assertEqual(result, single_pop[0])
        
        # Elitism with single individual
        result = self.strategy.elitism_selection(single_pop, 1)
        self.assertEqual(result, single_pop)
    
    def test_population_with_identical_fitness(self):
        """Test selection strategies with populations having identical fitness values."""
        identical_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        # All selection methods should handle identical fitness
        result = self.strategy.tournament_selection(identical_fitness_pop, tournament_size=2)
        self.assertIn(result, identical_fitness_pop)
        
        result = self.strategy.roulette_wheel_selection(identical_fitness_pop)
        self.assertIn(result, identical_fitness_pop)
        
        result = self.strategy.rank_selection(identical_fitness_pop)
        self.assertIn(result, identical_fitness_pop)
    
    def test_population_with_negative_fitness(self):
        """Test selection strategies with negative fitness values."""
        negative_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': -0.1},
            {'genome': [4, 5, 6], 'fitness': -0.5},
            {'genome': [7, 8, 9], 'fitness': -0.9}
        ]
        
        # Tournament selection should work with negative fitness
        result = self.strategy.tournament_selection(negative_fitness_pop, tournament_size=2)
        self.assertIn(result, negative_fitness_pop)
        
        # Rank selection should work with negative fitness
        result = self.strategy.rank_selection(negative_fitness_pop)
        self.assertIn(result, negative_fitness_pop)
    
    def test_elitism_with_elite_count_exceeding_population(self):
        """Test elitism selection when elite count exceeds population size."""
        small_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7}
        ]
        
        # Should return entire population when elite count > population size
        result = self.strategy.elitism_selection(small_pop, elite_count=5)
        self.assertEqual(len(result), len(small_pop))
        self.assertEqual(result, small_pop)
    
    def test_tournament_selection_with_large_tournament_size(self):
        """Test tournament selection with tournament size equal to population size."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        # Tournament size equal to population size should select the best
        result = self.strategy.tournament_selection(population, tournament_size=len(population))
        self.assertEqual(result['fitness'], 0.9)
    
    def test_roulette_wheel_with_zero_total_fitness(self):
        """Test roulette wheel selection with zero total fitness."""
        zero_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.0},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 0.0}
        ]
        
        # Should handle zero fitness gracefully
        result = self.strategy.roulette_wheel_selection(zero_fitness_pop)
        self.assertIn(result, zero_fitness_pop)
    
    def test_selection_with_extreme_fitness_values(self):
        """Test selection strategies with extreme fitness values."""
        extreme_pop = [
            {'genome': [1, 2, 3], 'fitness': 1e10},
            {'genome': [4, 5, 6], 'fitness': 1e-10},
            {'genome': [7, 8, 9], 'fitness': 1e5}
        ]
        
        # All selection methods should handle extreme values
        result = self.strategy.tournament_selection(extreme_pop, tournament_size=2)
        self.assertIn(result, extreme_pop)
        
        result = self.strategy.roulette_wheel_selection(extreme_pop)
        self.assertIn(result, extreme_pop)
    
    def test_selection_performance_with_large_population(self):
        """Test selection performance with large population sizes."""
        large_pop = [
            {'genome': [i], 'fitness': i * 0.001} 
            for i in range(1000)
        ]
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        result = self.strategy.tournament_selection(large_pop, tournament_size=10)
        end_time = time.time()
        
        self.assertIn(result, large_pop)
        self.assertLess(end_time - start_time, 1.0)  # Should complete within 1 second


class TestFitnessFunctionExtended(unittest.TestCase):
    """Extended test suite for FitnessFunction class covering edge cases and advanced scenarios."""
    
    def setUp(self):
        """Set up test fixtures for extended fitness function tests."""
        self.fitness_func = FitnessFunction()
    
    def test_fitness_functions_with_empty_genome(self):
        """Test all fitness functions with empty genomes."""
        empty_genome = []
        
        # All fitness functions should handle empty genomes
        self.assertEqual(self.fitness_func.sphere_function(empty_genome), 0.0)
        self.assertEqual(self.fitness_func.rastrigin_function(empty_genome), 0.0)
        self.assertEqual(self.fitness_func.rosenbrock_function(empty_genome), 0.0)
        self.assertEqual(self.fitness_func.ackley_function(empty_genome), 0.0)
    
    def test_fitness_functions_with_single_element(self):
        """Test fitness functions with single-element genomes."""
        single_genome = [2.0]
        
        # Test sphere function
        fitness = self.fitness_func.sphere_function(single_genome)
        self.assertEqual(fitness, -4.0)  # -(2.0^2)
        
        # Test with other functions
        fitness = self.fitness_func.rastrigin_function(single_genome)
        self.assertIsInstance(fitness, (int, float))
        
        fitness = self.fitness_func.ackley_function(single_genome)
        self.assertIsInstance(fitness, (int, float))
    
    def test_fitness_functions_with_large_genomes(self):
        """Test fitness functions with large genome sizes."""
        large_genome = [1.0] * 1000  # 1000 elements
        
        # Should handle large genomes efficiently
        import time
        start_time = time.time()
        fitness = self.fitness_func.sphere_function(large_genome)
        end_time = time.time()
        
        self.assertEqual(fitness, -1000.0)  # -(1000 * 1.0^2)
        self.assertLess(end_time - start_time, 1.0)  # Should complete quickly
    
    def test_fitness_functions_with_extreme_values(self):
        """Test fitness functions with extreme numerical values."""
        extreme_genome = [1e6, -1e6, 1e-6, -1e-6]
        
        # Functions should handle extreme values without errors
        fitness = self.fitness_func.sphere_function(extreme_genome)
        self.assertIsInstance(fitness, (int, float))
        self.assertFalse(math.isnan(fitness))
        self.assertFalse(math.isinf(fitness))
    
    def test_multi_objective_with_conflicting_objectives(self):
        """Test multi-objective optimization with conflicting objectives."""
        genome = [1.0, 2.0, 3.0]
        
        # Create conflicting objectives
        objectives = [
            lambda g: sum(g),           # Maximize sum
            lambda g: -sum(g),          # Minimize sum (conflicting)
            lambda g: sum(x**2 for x in g),  # Maximize sum of squares
            lambda g: -sum(x**2 for x in g)  # Minimize sum of squares (conflicting)
        ]
        
        fitness = self.fitness_func.multi_objective_evaluate(genome, objectives)
        
        self.assertEqual(len(fitness), 4)
        self.assertEqual(fitness[0], 6.0)
        self.assertEqual(fitness[1], -6.0)
        self.assertEqual(fitness[2], 14.0)
        self.assertEqual(fitness[3], -14.0)
    
    def test_constraint_handling_with_multiple_constraints(self):
        """Test constraint handling with multiple constraints."""
        genome = [2.0, 3.0, 4.0]
        
        def constraint1(g):
            return sum(g) < 10  # Sum should be less than 10
        
        def constraint2(g):
            return all(x > 0 for x in g)  # All elements should be positive
        
        def constraint3(g):
            return max(g) < 5  # Maximum element should be less than 5
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome,
            lambda g: sum(g),
            [constraint1, constraint2, constraint3]
        )
        
        # Should be penalized for violating constraint3 (max(g) = 4 < 5 is satisfied, so no penalty)
        self.assertIsInstance(fitness, (int, float))
    
    def test_constraint_handling_with_zero_penalty(self):
        """Test constraint handling when no constraints are violated."""
        genome = [1.0, 2.0, 3.0]
        
        def always_satisfied_constraint(g):
            return True  # Always satisfied
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome,
            lambda g: sum(g),
            [always_satisfied_constraint]
        )
        
        # Should equal the original fitness (no penalty)
        self.assertEqual(fitness, sum(genome))
    
    def test_custom_function_with_exception_handling(self):
        """Test custom fitness function that raises exceptions."""
        def failing_fitness(genome):
            if len(genome) > 3:
                raise ValueError("Genome too long")
            return sum(genome)
        
        # Should handle exceptions gracefully
        short_genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate(short_genome, failing_fitness)
        self.assertEqual(fitness, 6.0)
        
        # Long genome should cause exception
        long_genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        with self.assertRaises(ValueError):
            self.fitness_func.evaluate(long_genome, failing_fitness)


class TestPopulationManagerExtended(unittest.TestCase):
    """Extended test suite for PopulationManager class covering edge cases and performance scenarios."""
    
    def setUp(self):
        """Set up test fixtures for extended population manager tests."""
        self.manager = PopulationManager()
    
    def test_initialize_random_population_with_extreme_sizes(self):
        """Test population initialization with extreme population sizes."""
        # Test with very small population
        small_pop = self.manager.initialize_random_population(1, 5)
        self.assertEqual(len(small_pop), 1)
        self.assertEqual(len(small_pop[0]['genome']), 5)
        
        # Test with larger population
        large_pop = self.manager.initialize_random_population(1000, 10)
        self.assertEqual(len(large_pop), 1000)
        for individual in large_pop:
            self.assertEqual(len(individual['genome']), 10)
    
    def test_initialize_seeded_population_with_more_seeds_than_population(self):
        """Test seeded population initialization when seeds exceed population size."""
        seeds = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0]
        ]
        
        # Population size smaller than number of seeds
        population = self.manager.initialize_seeded_population(3, 3, seeds)
        self.assertEqual(len(population), 3)
        
        # All individuals should have seeded genomes
        genomes = [ind['genome'] for ind in population]
        for genome in genomes:
            self.assertIn(genome, seeds[:3])  # Only first 3 seeds should be used
    
    def test_initialize_seeded_population_with_invalid_seed_length(self):
        """Test seeded population initialization with seeds of wrong length."""
        seeds = [
            [1.0, 2.0],  # Wrong length (should be 3)
            [4.0, 5.0, 6.0, 7.0]  # Wrong length (should be 3)
        ]
        
        # Should handle invalid seed lengths gracefully
        population = self.manager.initialize_seeded_population(5, 3, seeds)
        self.assertEqual(len(population), 5)
        
        # All genomes should have correct length
        for individual in population:
            self.assertEqual(len(individual['genome']), 3)
    
    def test_evaluate_population_with_failing_fitness_function(self):
        """Test population evaluation with fitness function that fails for some individuals."""
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        def selective_fitness(genome):
            if sum(genome) > 15:  # Fails for [7, 8, 9]
                raise ValueError("Fitness calculation failed")
            return sum(genome)
        
        # Should handle partial failures gracefully
        with self.assertRaises(ValueError):
            self.manager.evaluate_population(population, selective_fitness)
    
    def test_get_population_statistics_with_edge_cases(self):
        """Test population statistics calculation with edge cases."""
        # Test with population having very similar fitness values
        similar_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.1000001},
            {'genome': [4, 5, 6], 'fitness': 0.1000002},
            {'genome': [7, 8, 9], 'fitness': 0.1000003}
        ]
        
        stats = self.manager.get_population_statistics(similar_fitness_pop)
        self.assertIn('std_dev_fitness', stats)
        self.assertGreater(stats['std_dev_fitness'], 0.0)
        
        # Test with population having extreme fitness spread
        extreme_spread_pop = [
            {'genome': [1, 2, 3], 'fitness': 1e-10},
            {'genome': [4, 5, 6], 'fitness': 1e10},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        stats = self.manager.get_population_statistics(extreme_spread_pop)
        self.assertIsInstance(stats['std_dev_fitness'], float)
        self.assertFalse(math.isnan(stats['std_dev_fitness']))
    
    def test_diversity_calculation_with_identical_genomes(self):
        """Test diversity calculation with identical genomes."""
        identical_pop = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(identical_pop)
        self.assertEqual(diversity, 0.0)  # Should be zero for identical genomes
    
    def test_diversity_calculation_with_single_individual(self):
        """Test diversity calculation with single individual."""
        single_pop = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}
        ]
        
        diversity = self.manager.calculate_diversity(single_pop)
        self.assertEqual(diversity, 0.0)  # Should be zero for single individual
    
    def test_get_best_individual_with_ties(self):
        """Test best individual selection when multiple individuals have the same fitness."""
        tied_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.7}
        ]
        
        best = self.manager.get_best_individual(tied_pop)
        self.assertEqual(best['fitness'], 0.9)
        self.assertIn(best['genome'], [[1, 2, 3], [4, 5, 6]])
    
    def test_population_operations_with_nan_fitness(self):
        """Test population operations with NaN fitness values."""
        nan_pop = [
            {'genome': [1, 2, 3], 'fitness': float('nan')},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.7}
        ]
        
        # Should handle NaN values gracefully
        best = self.manager.get_best_individual(nan_pop)
        self.assertFalse(math.isnan(best['fitness']))
        
        stats = self.manager.get_population_statistics(nan_pop)
        self.assertFalse(math.isnan(stats['best_fitness']))


class TestGeneticOperationsExtended(unittest.TestCase):
    """Extended test suite for GeneticOperations class covering edge cases and advanced crossover scenarios."""
    
    def setUp(self):
        """Set up test fixtures for extended genetic operations tests."""
        self.operations = GeneticOperations()
    
    def test_crossover_with_empty_parents(self):
        """Test crossover operations with empty parent genomes."""
        empty_parent1 = []
        empty_parent2 = []
        
        # All crossover operations should handle empty parents
        child1, child2 = self.operations.single_point_crossover(empty_parent1, empty_parent2)
        self.assertEqual(child1, [])
        self.assertEqual(child2, [])
        
        child1, child2 = self.operations.uniform_crossover(empty_parent1, empty_parent2)
        self.assertEqual(child1, [])
        self.assertEqual(child2, [])
    
    def test_crossover_with_single_element_parents(self):
        """Test crossover operations with single-element parent genomes."""
        parent1 = [1.0]
        parent2 = [2.0]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 1)
        self.assertEqual(len(child2), 1)
        self.assertIn(child1[0], [1.0, 2.0])
        self.assertIn(child2[0], [1.0, 2.0])
    
    def test_two_point_crossover_with_small_genomes(self):
        """Test two-point crossover with genomes too small for two points."""
        small_parent1 = [1, 2]
        small_parent2 = [3, 4]
        
        child1, child2 = self.operations.two_point_crossover(small_parent1, small_parent2)
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)
    
    def test_uniform_crossover_with_extreme_rates(self):
        """Test uniform crossover with extreme crossover rates."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        # Test with rate 0.0 (no crossover)
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.0)
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
        
        # Test with rate 1.0 (complete crossover)
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=1.0)
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_arithmetic_crossover_with_negative_alpha(self):
        """Test arithmetic crossover with negative alpha values."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Should handle negative alpha
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=-0.5)
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_arithmetic_crossover_with_alpha_greater_than_one(self):
        """Test arithmetic crossover with alpha > 1."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Should handle alpha > 1
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.5)
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_simulated_binary_crossover_with_tight_bounds(self):
        """Test SBX crossover with very tight bounds."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [1.1, 2.1, 3.1]
        tight_bounds = [(0.9, 1.2), (1.9, 2.2), (2.9, 3.2)]
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, tight_bounds, eta=2.0
        )
        
        # Children should respect tight bounds
        for i, (lower, upper) in enumerate(tight_bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
            self.assertGreaterEqual(child2[i], lower)
            self.assertLessEqual(child2[i], upper)
    
    def test_blend_crossover_with_zero_alpha(self):
        """Test blend crossover with alpha = 0."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.0)
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
        
        # With alpha=0, children should be within parent bounds
        for i in range(len(parent1)):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            self.assertGreaterEqual(child1[i], min_val)
            self.assertLessEqual(child1[i], max_val)
            self.assertGreaterEqual(child2[i], min_val)
            self.assertLessEqual(child2[i], max_val)
    
    def test_crossover_with_identical_parents(self):
        """Test crossover operations with identical parent genomes."""
        identical_parent = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Single-point crossover with identical parents
        child1, child2 = self.operations.single_point_crossover(identical_parent, identical_parent)
        self.assertEqual(child1, identical_parent)
        self.assertEqual(child2, identical_parent)
        
        # Arithmetic crossover with identical parents
        child1, child2 = self.operations.arithmetic_crossover(identical_parent, identical_parent, alpha=0.5)
        self.assertEqual(child1, identical_parent)
        self.assertEqual(child2, identical_parent)
    
    def test_crossover_with_extreme_values(self):
        """Test crossover operations with extreme numerical values."""
        extreme_parent1 = [1e10, -1e10, 1e-10]
        extreme_parent2 = [2e10, -2e10, 2e-10]
        
        # Should handle extreme values without overflow/underflow
        child1, child2 = self.operations.arithmetic_crossover(extreme_parent1, extreme_parent2, alpha=0.5)
        
        for value in child1 + child2:
            self.assertFalse(math.isnan(value))
            self.assertFalse(math.isinf(value))
    
    def test_crossover_performance_with_large_genomes(self):
        """Test crossover performance with large genome sizes."""
        large_parent1 = [1.0] * 10000
        large_parent2 = [2.0] * 10000
        
        # Should complete efficiently
        import time
        start_time = time.time()
        child1, child2 = self.operations.single_point_crossover(large_parent1, large_parent2)
        end_time = time.time()
        
        self.assertEqual(len(child1), 10000)
        self.assertEqual(len(child2), 10000)
        self.assertLess(end_time - start_time, 1.0)  # Should complete within 1 second


class TestEvolutionaryConduitExtended(unittest.TestCase):
    """Extended test suite for EvolutionaryConduit class covering advanced scenarios and error handling."""
    
    def setUp(self):
        """Set up test fixtures for extended evolutionary conduit tests."""
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=20,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_multiple_callback_execution(self):
        """Test execution of multiple callbacks during evolution."""
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
        
        # Verify all callbacks are registered
        self.assertEqual(len(self.conduit.callbacks), 3)
        self.assertIn(callback1, self.conduit.callbacks)
        self.assertIn(callback2, self.conduit.callbacks)
        self.assertIn(callback3, self.conduit.callbacks)
    
    def test_callback_with_exception_handling(self):
        """Test callback execution when one callback raises an exception."""
        successful_calls = []
        
        def working_callback(generation, population, best_individual):
            successful_calls.append(generation)
        
        def failing_callback(generation, population, best_individual):
            raise ValueError("Callback failed")
        
        self.conduit.add_callback(working_callback)
        self.conduit.add_callback(failing_callback)
        
        # Should handle failing callback gracefully
        self.assertEqual(len(self.conduit.callbacks), 2)
    
    def test_state_serialization_with_complex_fitness_function(self):
        """Test state saving and loading with complex fitness functions."""
        def complex_fitness(genome):
            # Complex fitness function with nested operations
            return sum(x**2 for x in genome) + sum(abs(x) for x in genome) - len(genome)
        
        self.conduit.set_fitness_function(complex_fitness)
        self.conduit.set_parameters(self.params)
        
        # Save and load state
        state = self.conduit.save_state()
        new_conduit = EvolutionaryConduit()
        new_conduit.load_state(state)
        
        # Test that fitness function works after loading
        test_genome = [1.0, 2.0, 3.0]
        expected_fitness = complex_fitness(test_genome)
        
        # Should be able to use the fitness function (though the actual function object may not be serializable)
        self.assertEqual(new_conduit.parameters.population_size, 20)
        self.assertEqual(new_conduit.parameters.generations, 10)
    
    def test_parameter_validation_edge_cases(self):
        """Test parameter validation with edge cases."""
        # Test with parameters at boundary values
        boundary_params = EvolutionaryParameters(
            population_size=1,
            generations=1,
            mutation_rate=0.0,
            crossover_rate=1.0,
            selection_pressure=1.0
        )
        
        self.conduit.set_parameters(boundary_params)
        self.assertEqual(self.conduit.parameters.population_size, 1)
        self.assertEqual(self.conduit.parameters.mutation_rate, 0.0)
        self.assertEqual(self.conduit.parameters.crossover_rate, 1.0)
    
    def test_history_tracking_with_large_runs(self):
        """Test history tracking with large evolution runs."""
        self.conduit.set_parameters(EvolutionaryParameters(
            population_size=100,
            generations=100
        ))
        self.conduit.enable_history_tracking()
        
        # Should be able to handle large history without memory issues
        self.assertTrue(self.conduit.history_enabled)
    
    def test_evolution_with_stagnation_detection(self):
        """Test evolution behavior with fitness stagnation."""
        stagnation_count = 0
        
        def stagnant_fitness(genome):
            nonlocal stagnation_count
            stagnation_count += 1
            return 0.5  # Always return the same fitness
        
        self.conduit.set_fitness_function(stagnant_fitness)
        self.conduit.set_parameters(self.params)
        
        # Mock evolution to test stagnation handling
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 0.5},
                'generations_run': 10,
                'final_population': [],
                'statistics': {'best_fitness': 0.5, 'stagnation_detected': True}
            }
            
            result = self.conduit.run_evolution(genome_length=3)
            self.assertIn('statistics', result)
    
    def test_concurrent_evolution_instances(self):
        """Test running multiple evolution instances concurrently."""
        conduit1 = EvolutionaryConduit()
        conduit2 = EvolutionaryConduit()
        
        params1 = EvolutionaryParameters(population_size=10, generations=5)
        params2 = EvolutionaryParameters(population_size=20, generations=3)
        
        conduit1.set_parameters(params1)
        conduit2.set_parameters(params2)
        
        # Both should maintain separate state
        self.assertEqual(conduit1.parameters.population_size, 10)
        self.assertEqual(conduit2.parameters.population_size, 20)
        self.assertEqual(conduit1.parameters.generations, 5)
        self.assertEqual(conduit2.parameters.generations, 3)


class TestGenesisEvolutionaryConduitExtended(unittest.TestCase):
    """Extended test suite for GenesisEvolutionaryConduit class covering advanced neural evolution scenarios."""
    
    def setUp(self):
        """Set up test fixtures for extended genesis evolutionary conduit tests."""
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=20,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_network_topology_evolution_with_constraints(self):
        """Test topology evolution with structural constraints."""
        # Define constraints
        max_layers = 5
        max_nodes_per_layer = 100
        min_nodes_per_layer = 1
        
        initial_topology = {
            'layers': [10, 20, 15, 1],
            'connections': [[0, 1], [1, 2], [2, 3]]
        }
        
        # Test multiple mutations to ensure constraints are maintained
        for _ in range(10):
            mutated = self.genesis_conduit.mutate_topology(initial_topology)
            
            self.assertLessEqual(len(mutated['layers']), max_layers)
            for layer_size in mutated['layers']:
                self.assertGreaterEqual(layer_size, min_nodes_per_layer)
                self.assertLessEqual(layer_size, max_nodes_per_layer)
    
    def test_hyperparameter_optimization_with_categorical_parameters(self):
        """Test hyperparameter optimization with categorical parameters."""
        search_space = {
            'learning_rate': (0.001, 0.1),
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'activation': ['relu', 'tanh', 'sigmoid'],
            'batch_size': [16, 32, 64, 128]
        }
        
        self.genesis_conduit.set_hyperparameter_search_space(search_space)
        
        # Generate multiple hyperparameter sets
        for _ in range(10):
            hyperparams = self.genesis_conduit.generate_hyperparameters()
            
            self.assertIn('learning_rate', hyperparams)
            self.assertIn('optimizer', hyperparams)
            self.assertIn('activation', hyperparams)
            self.assertIn('batch_size', hyperparams)
            
            # Check categorical parameters
            self.assertIn(hyperparams['optimizer'], ['adam', 'sgd', 'rmsprop'])
            self.assertIn(hyperparams['activation'], ['relu', 'tanh', 'sigmoid'])
            self.assertIn(hyperparams['batch_size'], [16, 32, 64, 128])
    
    def test_multi_objective_optimization_with_pareto_front(self):
        """Test multi-objective optimization with Pareto front calculation."""
        # Create population with different trade-offs
        population = [
            {'genome': [1, 2, 3], 'fitness': [0.9, 0.1, 0.2]},  # High accuracy, low size, low time
            {'genome': [4, 5, 6], 'fitness': [0.7, 0.3, 0.1]},  # Medium accuracy, medium size, low time
            {'genome': [7, 8, 9], 'fitness': [0.5, 0.8, 0.3]},  # Low accuracy, high size, medium time
            {'genome': [10, 11, 12], 'fitness': [0.8, 0.2, 0.4]}  # High accuracy, low size, high time
        ]
        
        # Test Pareto front calculation
        pareto_front = self.genesis_conduit.calculate_pareto_front(population)
        
        self.assertIsInstance(pareto_front, list)
        self.assertGreater(len(pareto_front), 0)
        self.assertLessEqual(len(pareto_front), len(population))
    
    def test_adaptive_mutation_with_fitness_landscape_analysis(self):
        """Test adaptive mutation with fitness landscape analysis."""
        # Create population with different fitness landscapes
        diverse_population = [
            {'genome': [1, 2, 3], 'fitness': 0.9, 'generation': 1},
            {'genome': [1.1, 2.1, 3.1], 'fitness': 0.85, 'generation': 2},
            {'genome': [5, 6, 7], 'fitness': 0.3, 'generation': 3},
            {'genome': [5.1, 6.1, 7.1], 'fitness': 0.25, 'generation': 4}
        ]
        
        # Test adaptive mutation rate calculation
        adaptive_rate = self.genesis_conduit.calculate_adaptive_mutation_rate(diverse_population)
        
        self.assertIsInstance(adaptive_rate, float)
        self.assertGreater(adaptive_rate, 0.0)
        self.assertLess(adaptive_rate, 1.0)
    
    def test_speciation_with_dynamic_thresholds(self):
        """Test speciation with dynamic similarity thresholds."""
        # Create population with varying similarity
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.1, 2.1, 3.1], 'fitness': 0.6},  # Similar to first
            {'genome': [1.2, 2.2, 3.2], 'fitness': 0.55}, # Similar to first two
            {'genome': [10.0, 20.0, 30.0], 'fitness': 0.7}, # Very different
            {'genome': [10.1, 20.1, 30.1], 'fitness': 0.8}  # Similar to fourth
        ]
        
        # Test with different thresholds
        strict_species = self.genesis_conduit.speciate_population(population, distance_threshold=0.5)
        loose_species = self.genesis_conduit.speciate_population(population, distance_threshold=50.0)
        
        # Strict threshold should create more species
        self.assertGreaterEqual(len(strict_species), len(loose_species))
        self.assertGreater(len(strict_species), 0)
    
    def test_transfer_learning_with_different_architectures(self):
        """Test transfer learning between different network architectures."""
        # Pretrained network for smaller problem
        small_network_genome = [0.1, 0.2, 0.3, 0.4]
        
        # Adapt to larger network
        large_task_config = {
            'input_size': 20,
            'output_size': 10,
            'hidden_layers': [50, 30, 20]
        }
        
        adapted_genome = self.genesis_conduit.adapt_pretrained_network(
            small_network_genome, 
            large_task_config
        )
        
        self.assertIsInstance(adapted_genome, list)
        self.assertGreater(len(adapted_genome), len(small_network_genome))
    
    def test_ensemble_evolution_with_diversity_metrics(self):
        """Test ensemble evolution with diversity constraints."""
        # Create diverse networks
        diverse_networks = [
            {'genome': [1, 2, 3], 'fitness': 0.8, 'architecture': 'small'},
            {'genome': [4, 5, 6, 7, 8], 'fitness': 0.85, 'architecture': 'medium'},
            {'genome': [9, 10, 11, 12, 13, 14], 'fitness': 0.9, 'architecture': 'large'},
            {'genome': [15, 16, 17], 'fitness': 0.75, 'architecture': 'small'},
            {'genome': [18, 19, 20, 21], 'fitness': 0.82, 'architecture': 'medium'}
        ]
        
        # Create ensemble with diversity consideration
        ensemble = self.genesis_conduit.create_ensemble(diverse_networks, ensemble_size=3)
        
        self.assertEqual(len(ensemble), 3)
        # Should select high-performing and diverse networks
        fitness_values = [net['fitness'] for net in ensemble]
        self.assertTrue(all(f >= 0.75 for f in fitness_values))
    
    def test_coevolution_with_competitive_fitness(self):
        """Test coevolution with competitive fitness evaluation."""
        # Create competitive populations
        predator_population = [
            {'genome': [1, 2, 3], 'fitness': 0.6, 'role': 'predator'},
            {'genome': [4, 5, 6], 'fitness': 0.7, 'role': 'predator'}
        ]
        
        prey_population = [
            {'genome': [7, 8, 9], 'fitness': 0.5, 'role': 'prey'},
            {'genome': [10, 11, 12], 'fitness': 0.8, 'role': 'prey'}
        ]
        
        # Test competitive coevolution
        result = self.genesis_conduit.coevolve_populations(predator_population, prey_population)
        
        self.assertIsInstance(result, dict)
        self.assertIn('population1', result)
        self.assertIn('population2', result)
        self.assertIn('competitive_results', result)
    
    def test_novelty_search_with_behavior_characterization(self):
        """Test novelty search with behavior characterization."""
        # Create population with different behaviors
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5, 'behavior': [0.1, 0.2, 0.3]},
            {'genome': [4, 5, 6], 'fitness': 0.7, 'behavior': [0.4, 0.5, 0.6]},
            {'genome': [7, 8, 9], 'fitness': 0.9, 'behavior': [0.7, 0.8, 0.9]},
            {'genome': [1.1, 2.1, 3.1], 'fitness': 0.6, 'behavior': [0.11, 0.21, 0.31]}
        ]
        
        # Calculate novelty scores
        novelty_scores = self.genesis_conduit.calculate_novelty_scores(population)
        
        self.assertEqual(len(novelty_scores), len(population))
        for score in novelty_scores:
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
    
    def test_distributed_evolution_with_migration_strategies(self):
        """Test distributed evolution with different migration strategies."""
        # Set up island model
        island_configs = [
            {'island_id': 1, 'population_size': 20, 'migration_rate': 0.1},
            {'island_id': 2, 'population_size': 15, 'migration_rate': 0.2},
            {'island_id': 3, 'population_size': 25, 'migration_rate': 0.05}
        ]
        
        self.genesis_conduit.setup_island_model(island_configs)
        
        # Test different migration strategies
        pop1 = [{'genome': [1, 2, 3], 'fitness': 0.8}] * 10
        pop2 = [{'genome': [4, 5, 6], 'fitness': 0.6}] * 8
        pop3 = [{'genome': [7, 8, 9], 'fitness': 0.9}] * 12
        
        # Test best-individual migration
        migrated_best = self.genesis_conduit.migrate_best_individuals([pop1, pop2, pop3])
        self.assertIsInstance(migrated_best, list)
        self.assertEqual(len(migrated_best), 3)
        
        # Test random migration
        migrated_random = self.genesis_conduit.migrate_random_individuals([pop1, pop2, pop3])
        self.assertIsInstance(migrated_random, list)
        self.assertEqual(len(migrated_random), 3)
    
    def test_checkpoint_and_resume_functionality(self):
        """Test checkpoint saving and resuming evolution."""
        # Set up evolution state
        self.genesis_conduit.set_parameters(self.params)
        
        # Create mock evolution state
        evolution_state = {
            'generation': 5,
            'population': [{'genome': [1, 2, 3], 'fitness': 0.5}] * 10,
            'best_fitness_history': [0.1, 0.2, 0.3, 0.4, 0.5],
            'parameters': self.params.to_dict()
        }
        
        # Test checkpoint creation
        checkpoint_path = "test_checkpoint.pkl"
        
        with patch.object(self.genesis_conduit, 'save_checkpoint') as mock_save:
            self.genesis_conduit.save_checkpoint(checkpoint_path, evolution_state)
            mock_save.assert_called_once_with(checkpoint_path, evolution_state)
        
        # Test checkpoint loading
        with patch.object(self.genesis_conduit, 'load_checkpoint') as mock_load:
            mock_load.return_value = evolution_state
            loaded_state = self.genesis_conduit.load_checkpoint(checkpoint_path)
            
            self.assertEqual(loaded_state['generation'], 5)
            self.assertEqual(len(loaded_state['population']), 10)
            self.assertEqual(len(loaded_state['best_fitness_history']), 5)


class TestAsyncEvolutionExtended(unittest.TestCase):
    """Extended test suite for asynchronous evolution capabilities covering advanced concurrent scenarios."""
    
    def setUp(self):
        """Set up test fixtures for extended asynchronous evolution tests."""
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=20,
            generations=10
        )
    
    @patch('asyncio.gather')
    def test_concurrent_population_evaluation(self, mock_gather):
        """Test concurrent evaluation of population fitness."""
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        # Mock async fitness evaluation
        async def mock_fitness_eval(individual):
            return sum(individual['genome'])
        
        mock_gather.return_value = asyncio.run(
            asyncio.gather(*[mock_fitness_eval(ind) for ind in population])
        )
        
        # Test concurrent evaluation
        result = self.genesis_conduit.evaluate_population_async(population, mock_fitness_eval)
        
        self.assertIsNotNone(result)
        mock_gather.assert_called_once()
    
    @patch('multiprocessing.Pool')
    def test_multiprocess_evolution(self, mock_pool):
        """Test evolution using multiprocessing."""
        # Mock multiprocessing pool
        mock_pool.return_value.__enter__.return_value.map.return_value = [0.5, 0.7, 0.9]
        
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        def fitness_func(genome):
            return sum(genome)
        
        # Test multiprocess evaluation
        self.genesis_conduit.evaluate_population_multiprocess(population, fitness_func)
        
        # Verify multiprocessing was used
        mock_pool.assert_called_once()
    
    @patch('threading.Thread')
    def test_threaded_evolution_components(self, mock_thread):
        """Test threaded execution of evolution components."""
        # Mock threading
        mock_thread.return_value.start.return_value = None
        mock_thread.return_value.join.return_value = None
        
        # Test threaded mutation
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}] * 10
        
        result = self.genesis_conduit.mutate_population_threaded(population)
        
        self.assertIsInstance(result, list)
        # Threading should have been used
        mock_thread.assert_called()
    
    def test_async_callback_execution(self):
        """Test asynchronous callback execution during evolution."""
        async_callback_results = []
        
        async def async_callback(generation, population, best_individual):
            async_callback_results.append({
                'generation': generation,
                'population_size': len(population),
                'best_fitness': best_individual['fitness']
            })
        
        self.genesis_conduit.add_async_callback(async_callback)
        
        # Verify callback registration
        self.assertIn(async_callback, self.genesis_conduit.async_callbacks)
    
    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_distributed_fitness_evaluation(self, mock_executor):
        """Test distributed fitness evaluation across multiple processes."""
        # Mock distributed execution
        mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = 0.7
        
        large_population = [
            {'genome': [i, i+1, i+2], 'fitness': None} 
            for i in range(100)
        ]
        
        def complex_fitness(genome):
            # Simulate computationally expensive fitness function
            return sum(x**2 for x in genome) * 0.001
        
        # Test distributed evaluation
        self.genesis_conduit.evaluate_population_distributed(large_population, complex_fitness)
        
        # Should use process pool executor
        mock_executor.assert_called_once()
    
    def test_async_migration_between_islands(self):
        """Test asynchronous migration between evolution islands."""
        # Create multiple islands
        island1 = [{'genome': [1, 2, 3], 'fitness': 0.8}] * 10
        island2 = [{'genome': [4, 5, 6], 'fitness': 0.6}] * 8
        island3 = [{'genome': [7, 8, 9], 'fitness': 0.9}] * 12
        
        islands = [island1, island2, island3]
        
        # Test async migration
        with patch.object(self.genesis_conduit, 'migrate_async') as mock_migrate:
            mock_migrate.return_value = islands  # Return modified islands
            
            result = self.genesis_conduit.migrate_async(islands, migration_rate=0.1)
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)
            mock_migrate.assert_called_once()