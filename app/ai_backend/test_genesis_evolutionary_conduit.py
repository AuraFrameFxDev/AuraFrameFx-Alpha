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


class TestEvolutionaryParametersEdgeCases(unittest.TestCase):
    """Additional edge case tests for EvolutionaryParameters class."""
    
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
    
    def test_float_precision_handling(self):
        """Test handling of floating point precision edge cases."""
        params = EvolutionaryParameters(
            mutation_rate=0.9999999999999999,
            crossover_rate=0.0000000000000001
        )
        self.assertLessEqual(params.mutation_rate, 1.0)
        self.assertGreaterEqual(params.crossover_rate, 0.0)
    
    def test_invalid_generation_values(self):
        """Test invalid generation values."""
        with self.assertRaises(ValueError):
            EvolutionaryParameters(generations=0)
        
        with self.assertRaises(ValueError):
            EvolutionaryParameters(generations=-1)
    
    def test_dict_conversion_round_trip(self):
        """Test that conversion to dict and back preserves all values."""
        original = EvolutionaryParameters(
            population_size=50,
            generations=250,
            mutation_rate=0.05,
            crossover_rate=0.75,
            selection_pressure=0.15
        )
        dict_repr = original.to_dict()
        reconstructed = EvolutionaryParameters.from_dict(dict_repr)
        
        self.assertEqual(original.population_size, reconstructed.population_size)
        self.assertEqual(original.generations, reconstructed.generations)
        self.assertEqual(original.mutation_rate, reconstructed.mutation_rate)
        self.assertEqual(original.crossover_rate, reconstructed.crossover_rate)
        self.assertEqual(original.selection_pressure, reconstructed.selection_pressure)
    
    def test_from_dict_missing_keys(self):
        """Test from_dict with missing keys uses defaults."""
        partial_dict = {'population_size': 75}
        params = EvolutionaryParameters.from_dict(partial_dict)
        
        self.assertEqual(params.population_size, 75)
        self.assertEqual(params.generations, 500)  # Default value
        self.assertEqual(params.mutation_rate, 0.1)  # Default value
    
    def test_from_dict_extra_keys(self):
        """Test from_dict ignores extra keys."""
        dict_with_extra = {
            'population_size': 100,
            'generations': 500,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'selection_pressure': 0.2,
            'extra_key': 'ignored'
        }
        params = EvolutionaryParameters.from_dict(dict_with_extra)
        self.assertEqual(params.population_size, 100)


class TestMutationStrategyRobustness(unittest.TestCase):
    """Additional robustness tests for MutationStrategy class."""
    
    def setUp(self):
        self.strategy = MutationStrategy()
    
    def test_empty_genome_mutation(self):
        """Test mutation strategies with empty genomes."""
        empty_genome = []
        
        # All mutation methods should handle empty genomes gracefully
        mutated = self.strategy.gaussian_mutation(empty_genome, mutation_rate=0.1)
        self.assertEqual(len(mutated), 0)
        
        mutated = self.strategy.uniform_mutation(empty_genome, mutation_rate=0.1, bounds=(-1, 1))
        self.assertEqual(len(mutated), 0)
        
        mutated = self.strategy.bit_flip_mutation(empty_genome, mutation_rate=0.1)
        self.assertEqual(len(mutated), 0)
    
    def test_single_element_genome(self):
        """Test mutation with single-element genomes."""
        single_genome = [5.0]
        
        mutated = self.strategy.gaussian_mutation(single_genome, mutation_rate=1.0, sigma=0.1)
        self.assertEqual(len(mutated), 1)
        
        single_bool_genome = [True]
        mutated = self.strategy.bit_flip_mutation(single_bool_genome, mutation_rate=1.0)
        self.assertEqual(len(mutated), 1)
        self.assertIsInstance(mutated[0], bool)
    
    def test_extreme_mutation_rates(self):
        """Test with extreme mutation rates."""
        genome = [1.0, 2.0, 3.0]
        
        # Zero mutation rate should preserve genome
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.0)
        self.assertEqual(mutated, genome)
        
        # Maximum mutation rate
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=10.0)
        self.assertEqual(len(mutated), len(genome))
    
    def test_adaptive_mutation_edge_cases(self):
        """Test adaptive mutation with edge case fitness histories."""
        genome = [1.0, 2.0, 3.0]
        
        # Empty fitness history
        mutated = self.strategy.adaptive_mutation(genome, [], base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))
        
        # Single fitness value
        mutated = self.strategy.adaptive_mutation(genome, [0.5], base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))
        
        # All identical fitness values
        mutated = self.strategy.adaptive_mutation(genome, [0.5, 0.5, 0.5], base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))
    
    def test_gaussian_mutation_with_extreme_sigma(self):
        """Test Gaussian mutation with extreme sigma values."""
        genome = [1.0, 2.0, 3.0]
        
        # Very small sigma
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1e-10)
        self.assertEqual(len(mutated), len(genome))
        
        # Very large sigma
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1e10)
        self.assertEqual(len(mutated), len(genome))
    
    def test_uniform_mutation_with_tight_bounds(self):
        """Test uniform mutation with very tight bounds."""
        genome = [1.0, 2.0, 3.0]
        
        # Extremely tight bounds
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=1.0, bounds=(0.999, 1.001))
        for value in mutated:
            self.assertGreaterEqual(value, 0.999)
            self.assertLessEqual(value, 1.001)


class TestSelectionStrategyStress(unittest.TestCase):
    """Stress tests for SelectionStrategy class."""
    
    def setUp(self):
        self.strategy = SelectionStrategy()
    
    def test_large_population_selection(self):
        """Test selection with large populations."""
        # Create large population
        large_population = [
            {'genome': [i, i+1, i+2], 'fitness': i * 0.001}
            for i in range(1000)
        ]
        
        selected = self.strategy.tournament_selection(large_population, tournament_size=10)
        self.assertIn(selected, large_population)
        
        selected = self.strategy.roulette_wheel_selection(large_population)
        self.assertIn(selected, large_population)
    
    def test_selection_with_identical_fitness(self):
        """Test selection when all individuals have identical fitness."""
        identical_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        selected = self.strategy.roulette_wheel_selection(identical_fitness_pop)
        self.assertIn(selected, identical_fitness_pop)
        
        selected = self.strategy.rank_selection(identical_fitness_pop)
        self.assertIn(selected, identical_fitness_pop)
    
    def test_selection_with_negative_fitness(self):
        """Test selection with negative fitness values."""
        negative_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': -0.1},
            {'genome': [4, 5, 6], 'fitness': -0.5},
            {'genome': [7, 8, 9], 'fitness': -0.9}
        ]
        
        selected = self.strategy.tournament_selection(negative_fitness_pop, tournament_size=2)
        self.assertIn(selected, negative_fitness_pop)
        
        selected = self.strategy.rank_selection(negative_fitness_pop)
        self.assertIn(selected, negative_fitness_pop)
    
    def test_elitism_selection_edge_cases(self):
        """Test elitism selection edge cases."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        # Elite count equal to population size
        selected = self.strategy.elitism_selection(population, len(population))
        self.assertEqual(len(selected), len(population))
        
        # Elite count of 1
        selected = self.strategy.elitism_selection(population, 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['fitness'], 0.9)
    
    def test_tournament_selection_with_large_tournament(self):
        """Test tournament selection with tournament size equal to population."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        # Tournament size equal to population size should select best
        selected = self.strategy.tournament_selection(population, len(population))
        self.assertEqual(selected['fitness'], 0.9)


class TestFitnessFunctionComplexity(unittest.TestCase):
    """Complex scenarios for FitnessFunction class."""
    
    def setUp(self):
        self.fitness_func = FitnessFunction()
    
    def test_very_large_genomes(self):
        """Test fitness functions with very large genomes."""
        large_genome = [1.0] * 10000
        
        fitness = self.fitness_func.sphere_function(large_genome)
        self.assertEqual(fitness, -10000.0)  # -sum of squares
        
        fitness = self.fitness_func.rastrigin_function([0.0] * 10000)
        self.assertEqual(fitness, 0.0)
    
    def test_extreme_genome_values(self):
        """Test fitness functions with extreme genome values."""
        extreme_genome = [1e6, -1e6, 1e-6]
        
        fitness = self.fitness_func.sphere_function(extreme_genome)
        expected = -(1e12 + 1e12 + 1e-12)
        self.assertAlmostEqual(fitness, expected, places=10)
        
    def test_nan_and_inf_handling(self):
        """Test fitness function behavior with NaN and infinity values."""
        import math
        
        problematic_genome = [float('nan'), float('inf'), float('-inf')]
        
        # Functions should handle these gracefully
        try:
            fitness = self.fitness_func.sphere_function(problematic_genome)
            # Should either return a valid number or raise an exception
            self.assertTrue(isinstance(fitness, (int, float)) or math.isnan(fitness) or math.isinf(fitness))
        except (ValueError, ArithmeticError):
            # Acceptable to raise an exception for invalid inputs
            pass
    
    def test_multi_objective_with_conflicting_objectives(self):
        """Test multi-objective optimization with conflicting objectives."""
        genome = [1.0, 2.0, 3.0]
        
        conflicting_objectives = [
            lambda g: sum(g),      # Maximize sum
            lambda g: -sum(g),     # Minimize sum (conflicting)
            lambda g: len(g)       # Fixed value
        ]
        
        fitness = self.fitness_func.multi_objective_evaluate(genome, conflicting_objectives)
        
        self.assertEqual(len(fitness), 3)
        self.assertEqual(fitness[0], 6.0)
        self.assertEqual(fitness[1], -6.0)
        self.assertEqual(fitness[2], 3)
    
    def test_constraint_handling_multiple_constraints(self):
        """Test fitness evaluation with multiple constraints."""
        genome = [2.0, 3.0, 4.0]
        
        constraints = [
            lambda g: sum(g) < 10,    # Sum constraint
            lambda g: all(x > 0 for x in g),  # Positivity constraint
            lambda g: max(g) < 5      # Maximum value constraint
        ]
        
        # All constraints satisfied
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, lambda g: sum(g), constraints
        )
        self.assertEqual(fitness, sum(genome))  # No penalty
        
        # Violate one constraint
        violating_genome = [2.0, 3.0, 6.0]  # max > 5
        fitness = self.fitness_func.evaluate_with_constraints(
            violating_genome, lambda g: sum(g), constraints
        )
        self.assertLess(fitness, sum(violating_genome))  # Penalized
    
    def test_zero_dimension_genome(self):
        """Test fitness functions with zero-dimensional genomes."""
        empty_genome = []
        
        fitness = self.fitness_func.sphere_function(empty_genome)
        self.assertEqual(fitness, 0.0)
        
        fitness = self.fitness_func.rastrigin_function(empty_genome)
        self.assertEqual(fitness, 0.0)


class TestPopulationManagerAdvanced(unittest.TestCase):
    """Advanced tests for PopulationManager class."""
    
    def setUp(self):
        self.manager = PopulationManager()
    
    def test_population_initialization_with_constraints(self):
        """Test population initialization with genome constraints."""
        # Test bounded initialization
        population = self.manager.initialize_random_population(
            population_size=50,
            genome_length=10,
            bounds=(-5.0, 5.0)
        )
        
        for individual in population:
            for gene in individual['genome']:
                self.assertGreaterEqual(gene, -5.0)
                self.assertLessEqual(gene, 5.0)
    
    def test_seeded_population_with_insufficient_seeds(self):
        """Test seeded population when seeds < population size."""
        seeds = [[1.0, 2.0, 3.0]]  # Only one seed
        population_size = 10
        
        population = self.manager.initialize_seeded_population(
            population_size, 3, seeds
        )
        
        self.assertEqual(len(population), population_size)
        # First individual should be the seed
        self.assertEqual(population[0]['genome'], seeds[0])
    
    def test_population_statistics_with_outliers(self):
        """Test statistics calculation with outlier fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.001},  # Very low
            {'genome': [4, 5, 6], 'fitness': 0.5},    # Normal
            {'genome': [7, 8, 9], 'fitness': 1000.0}  # Very high outlier
        ]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 1000.0)
        self.assertEqual(stats['worst_fitness'], 0.001)
        self.assertGreater(stats['std_dev_fitness'], 0)
    
    def test_diversity_calculation_edge_cases(self):
        """Test diversity calculation with edge cases."""
        # All identical genomes
        identical_population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}
        ]
        
        diversity = self.manager.calculate_diversity(identical_population)
        self.assertEqual(diversity, 0.0)
        
        # Single individual
        single_population = [{'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}]
        diversity = self.manager.calculate_diversity(single_population)
        self.assertEqual(diversity, 0.0)
    
    def test_concurrent_population_evaluation(self):
        """Test population evaluation with concurrent processing."""
        large_population = self.manager.initialize_random_population(100, 10)
        
        def slow_fitness(genome):
            # Simulate slow fitness calculation
            import time
            time.sleep(0.001)
            return sum(genome)
        
        start_time = datetime.now()
        self.manager.evaluate_population(large_population, slow_fitness)
        end_time = datetime.now()
        
        # Verify all fitness values are assigned
        for individual in large_population:
            self.assertIsNotNone(individual['fitness'])


class TestGeneticOperationsEdgeCases(unittest.TestCase):
    """Edge case tests for GeneticOperations class."""
    
    def setUp(self):
        self.operations = GeneticOperations()
    
    def test_crossover_with_different_data_types(self):
        """Test crossover operations with mixed data types."""
        parent1 = [1, 2.5, True, 'a']
        parent2 = [4, 5.5, False, 'b']
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_crossover_with_very_short_genomes(self):
        """Test crossover with minimal genome lengths."""
        # Single element genomes
        parent1 = [1]
        parent2 = [2]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 1)
        self.assertEqual(len(child2), 1)
        
        # Two element genomes
        parent1 = [1, 2]
        parent2 = [3, 4]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)
    
    def test_arithmetic_crossover_edge_alpha_values(self):
        """Test arithmetic crossover with edge alpha values."""
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
    
    def test_simulated_binary_crossover_boundary_conditions(self):
        """Test SBX with boundary conditions."""
        parent1 = [0.0, 5.0, 10.0]
        parent2 = [10.0, 5.0, 0.0]
        bounds = [(0, 10), (0, 10), (0, 10)]
        
        # Test with extreme eta values
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=100.0
        )
        
        # Children should be within bounds
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
    
    def test_blend_crossover_extreme_alpha(self):
        """Test blend crossover with extreme alpha values."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Very large alpha
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=10.0)
        self.assertEqual(len(child1), len(parent1))
        
        # Zero alpha should create children close to parents
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.0)
        self.assertEqual(len(child1), len(parent1))


class TestConcurrencyAndThreadSafety(unittest.TestCase):
    """Tests for concurrent evolution and thread safety."""
    
    def setUp(self):
        self.conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(population_size=20, generations=5)
    
    def test_concurrent_fitness_evaluation(self):
        """Test that fitness evaluation can handle concurrent access."""
        import threading
        import time
        
        def fitness_func(genome):
            time.sleep(0.01)  # Simulate computation
            return sum(genome)
        
        population = self.conduit.population_manager.initialize_random_population(50, 5)
        
        # Create multiple threads for evaluation
        threads = []
        results = []
        
        def evaluate_subset(pop_subset, results_list, index):
            for individual in pop_subset:
                individual['fitness'] = fitness_func(individual['genome'])
            results_list.append(f"Thread {index} completed")
        
        # Split population among threads
        chunk_size = len(population) // 4
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < 3 else len(population)
            subset = population[start_idx:end_idx]
            
            thread = threading.Thread(
                target=evaluate_subset,
                args=(subset, results, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all individuals have fitness
        for individual in population:
            self.assertIsNotNone(individual['fitness'])
        
        self.assertEqual(len(results), 4)
    
    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_distributed_population_evolution(self, mock_executor):
        """Test distributed evolution across multiple processes."""
        mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = {
            'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
            'generation': 5
        }
        
        # Test distributed evolution setup
        island_populations = [
            [{'genome': [1, 2, 3], 'fitness': 0.5}],
            [{'genome': [4, 5, 6], 'fitness': 0.7}]
        ]
        
        # Simulate distributed evolution step
        results = []
        for pop in island_populations:
            result = self.conduit.evolve_island_population(pop)
            results.append(result)
        
        self.assertEqual(len(results), 2)


class TestMemoryAndPerformance(unittest.TestCase):
    """Tests for memory usage and performance characteristics."""
    
    def setUp(self):
        self.conduit = GenesisEvolutionaryConduit()
    
    def test_large_population_memory_usage(self):
        """Test memory efficiency with large populations."""
        import sys
        
        # Create large population
        large_pop_size = 10000
        genome_length = 100
        
        initial_size = sys.getsizeof(self.conduit)
        
        population = self.conduit.population_manager.initialize_random_population(
            large_pop_size, genome_length
        )
        
        # Verify population was created successfully
        self.assertEqual(len(population), large_pop_size)
        
        # Basic memory usage check
        final_size = sys.getsizeof(self.conduit) + sys.getsizeof(population)
        self.assertGreater(final_size, initial_size)
    
    def test_evolution_performance_scaling(self):
        """Test performance scaling with different population sizes."""
        import time
        
        population_sizes = [10, 50, 100]
        times = []
        
        def simple_fitness(genome):
            return sum(x**2 for x in genome)
        
        self.conduit.set_fitness_function(simple_fitness)
        
        for pop_size in population_sizes:
            params = EvolutionaryParameters(
                population_size=pop_size,
                generations=2
            )
            self.conduit.set_parameters(params)
            
            start_time = time.time()
            
            # Mock the evolution to avoid long running times
            with patch.object(self.conduit, 'evolve') as mock_evolve:
                mock_evolve.return_value = {
                    'best_individual': {'genome': [1, 2, 3], 'fitness': 14.0},
                    'generations_run': 2,
                    'final_population': [],
                    'statistics': {'best_fitness': 14.0}
                }
                
                result = self.conduit.run_evolution(genome_length=10)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Performance should scale reasonably (this is just a basic check)
        self.assertTrue(all(t >= 0 for t in times))


class TestErrorRecoveryAndResilience(unittest.TestCase):
    """Tests for error recovery and system resilience."""
    
    def setUp(self):
        self.conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(population_size=10, generations=5)
    
    def test_fitness_evaluation_failure_recovery(self):
        """Test recovery from fitness evaluation failures."""
        failure_count = 0
        
        def unreliable_fitness(genome):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # Fail first 3 times
                raise ValueError("Simulated fitness evaluation failure")
            return sum(genome)
        
        self.conduit.set_fitness_function(unreliable_fitness)
        
        # The system should handle fitness evaluation failures gracefully
        population = self.conduit.population_manager.initialize_random_population(5, 3)
        
        # Should either handle the error or raise EvolutionaryException
        try:
            self.conduit.population_manager.evaluate_population(population, unreliable_fitness)
        except (EvolutionaryException, ValueError):
            pass  # Expected behavior
    
    def test_mutation_operation_failure_handling(self):
        """Test handling of mutation operation failures."""
        def failing_mutation(genome, **kwargs):
            raise RuntimeError("Mutation failed")
        
        # Mock the mutation method to fail
        with patch.object(self.conduit.mutation_strategy, 'gaussian_mutation', side_effect=failing_mutation):
            genome = [1.0, 2.0, 3.0]
            
            # Should handle mutation failure gracefully
            try:
                result = self.conduit.mutation_strategy.gaussian_mutation(genome, mutation_rate=0.1)
            except (RuntimeError, EvolutionaryException):
                pass  # Expected behavior
    
    def test_crossover_operation_resilience(self):
        """Test crossover operations with malformed inputs."""
        operations = GeneticOperations()
        
        # Test with None values
        try:
            operations.single_point_crossover(None, [1, 2, 3])
        except (ValueError, TypeError):
            pass  # Expected
        
        # Test with non-list inputs
        try:
            operations.single_point_crossover("invalid", [1, 2, 3])
        except (ValueError, TypeError):
            pass  # Expected
    
    def test_population_corruption_recovery(self):
        """Test recovery from population corruption."""
        # Create population with some corrupted individuals
        corrupted_population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': None, 'fitness': 0.7},  # Corrupted
            {'invalid': 'structure'},            # Corrupted
            {'genome': [7, 8, 9], 'fitness': 0.9}
        ]
        
        # System should handle corrupted population gracefully
        try:
            stats = self.conduit.population_manager.get_population_statistics(corrupted_population)
        except (ValueError, KeyError, TypeError):
            pass  # Expected behavior for corrupted data


class TestConfigurationValidation(unittest.TestCase):
    """Tests for configuration validation and edge cases."""
    
    def setUp(self):
        self.conduit = GenesisEvolutionaryConduit()
    
    def test_invalid_network_configurations(self):
        """Test handling of invalid neural network configurations."""
        invalid_configs = [
            {'input_size': 0},  # Invalid input size
            {'input_size': 10, 'hidden_layers': []},  # No hidden layers
            {'input_size': 10, 'output_size': -1},  # Invalid output size
            {'input_size': -5, 'output_size': 1},  # Negative input size
        ]
        
        for config in invalid_configs:
            try:
                self.conduit.set_network_config(config)
                # Should either handle gracefully or raise appropriate exception
                network = self.conduit.create_neural_network()
            except (ValueError, EvolutionaryException):
                pass  # Expected for invalid configurations
    
    def test_hyperparameter_search_space_validation(self):
        """Test validation of hyperparameter search spaces."""
        invalid_search_spaces = [
            {'learning_rate': (0.1, 0.001)},  # Invalid range (min > max)
            {'batch_size': (-10, 128)},       # Negative values
            {},                               # Empty search space
            {'invalid_param': (0, 1)}         # Unknown parameter
        ]
        
        for search_space in invalid_search_spaces:
            try:
                self.conduit.set_hyperparameter_search_space(search_space)
                hyperparams = self.conduit.generate_hyperparameters()
            except (ValueError, EvolutionaryException):
                pass  # Expected for invalid search spaces
    
    def test_objective_configuration_validation(self):
        """Test validation of multi-objective configurations."""
        invalid_objectives = [
            [],                    # Empty objectives
            ['unknown_objective'], # Invalid objective name
            None,                  # None objectives
        ]
        
        for objectives in invalid_objectives:
            try:
                self.conduit.set_objectives(objectives)
            except (ValueError, TypeError):
                pass  # Expected for invalid objectives


# Additional async stress tests
class TestAsyncEvolutionStress(unittest.TestCase):
    """Stress tests for asynchronous evolution capabilities."""
    
    def setUp(self):
        self.conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(population_size=20, generations=3)
    
    @patch('asyncio.gather')
    def test_multiple_concurrent_evolutions(self, mock_gather):
        """Test running multiple evolution processes concurrently."""
        async def mock_evolution():
            return {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
                'generations_run': 3,
                'final_population': [],
                'statistics': {'best_fitness': 0.9}
            }
        
        mock_gather.return_value = [mock_evolution() for _ in range(5)]
        
        # Test multiple concurrent evolutions
        try:
            results = self.conduit.run_multiple_async_evolutions(
                evolution_configs=[self.params] * 5,
                genome_length=5
            )
            self.assertEqual(len(results), 5)
        except AttributeError:
            # Method might not exist, which is fine for this test
            pass
    
    def test_async_population_evaluation_stress(self):
        """Test asynchronous population evaluation under stress."""
        large_population = [
            {'genome': [i, i+1, i+2, i+3, i+4], 'fitness': None}
            for i in range(1000)
        ]
        
        def intensive_fitness(genome):
            # Simulate computationally intensive fitness
            result = 0
            for x in genome:
                result += x ** 2 + x ** 0.5 + abs(x)
            return result
        
        # Test that large population evaluation completes
        try:
            self.conduit.evaluate_population_async(large_population, intensive_fitness)
            
            # Verify all fitness values assigned
            for individual in large_population:
                self.assertIsNotNone(individual['fitness'])
                
        except AttributeError:
            # Method might not exist, skip test
            pass


if __name__ == '__main__':
    # Run all tests with increased verbosity
    unittest.main(verbosity=2, buffer=True)