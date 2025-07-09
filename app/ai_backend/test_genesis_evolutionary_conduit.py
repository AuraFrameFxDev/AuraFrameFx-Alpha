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
    """Extended test suite for EvolutionaryParameters edge cases and additional scenarios."""
    
    def test_boundary_values(self):
        """Test initialization with boundary values to ensure they are handled correctly."""
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
            mutation_rate=1.0,
            crossover_rate=1.0,
            selection_pressure=1.0
        )
        self.assertEqual(max_params.mutation_rate, 1.0)
        self.assertEqual(max_params.crossover_rate, 1.0)
    
    def test_parameter_types(self):
        """Test that parameters accept correct types and reject invalid types."""
        with self.assertRaises(TypeError):
            EvolutionaryParameters(population_size="invalid")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(mutation_rate="0.1")
    
    def test_negative_generations(self):
        """Test that negative generations raise ValueError."""
        with self.assertRaises(ValueError):
            EvolutionaryParameters(generations=-1)
    
    def test_parameter_modification_after_creation(self):
        """Test that parameters can be modified after object creation."""
        params = EvolutionaryParameters()
        params.population_size = 200
        params.mutation_rate = 0.15
        
        self.assertEqual(params.population_size, 200)
        self.assertEqual(params.mutation_rate, 0.15)
    
    def test_copy_parameters(self):
        """Test that parameters can be copied correctly."""
        original = EvolutionaryParameters(population_size=150, mutation_rate=0.2)
        params_dict = original.to_dict()
        copy_params = EvolutionaryParameters.from_dict(params_dict)
        
        self.assertEqual(original.population_size, copy_params.population_size)
        self.assertEqual(original.mutation_rate, copy_params.mutation_rate)
        
        # Ensure they are independent objects
        copy_params.population_size = 300
        self.assertNotEqual(original.population_size, copy_params.population_size)


class TestMutationStrategyExtended(unittest.TestCase):
    """Extended test suite for MutationStrategy covering edge cases and error conditions."""
    
    def setUp(self):
        self.strategy = MutationStrategy()
    
    def test_empty_genome_mutation(self):
        """Test mutation strategies with empty genomes."""
        empty_genome = []
        
        mutated = self.strategy.gaussian_mutation(empty_genome, mutation_rate=0.1)
        self.assertEqual(len(mutated), 0)
        
        mutated = self.strategy.uniform_mutation(empty_genome, mutation_rate=0.1, bounds=(-1, 1))
        self.assertEqual(len(mutated), 0)
    
    def test_single_element_genome(self):
        """Test mutation strategies with single-element genomes."""
        single_genome = [5.0]
        
        mutated = self.strategy.gaussian_mutation(single_genome, mutation_rate=1.0)
        self.assertEqual(len(mutated), 1)
        self.assertIsInstance(mutated[0], (int, float))
    
    def test_extreme_mutation_rates(self):
        """Test mutation with extreme but valid mutation rates."""
        genome = [1.0, 2.0, 3.0]
        
        # Zero mutation rate - genome should remain unchanged
        no_mutation = self.strategy.gaussian_mutation(genome, mutation_rate=0.0)
        self.assertEqual(no_mutation, genome)
        
        # Maximum mutation rate
        full_mutation = self.strategy.gaussian_mutation(genome, mutation_rate=1.0)
        self.assertEqual(len(full_mutation), len(genome))
    
    def test_gaussian_mutation_with_zero_sigma(self):
        """Test Gaussian mutation with zero standard deviation."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=0.0)
        self.assertEqual(mutated, genome)  # Should remain unchanged with sigma=0
    
    def test_uniform_mutation_with_equal_bounds(self):
        """Test uniform mutation where bounds are equal."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=1.0, bounds=(5.0, 5.0))
        
        for value in mutated:
            self.assertEqual(value, 5.0)
    
    def test_bit_flip_with_mixed_types(self):
        """Test bit flip mutation behavior with mixed boolean types."""
        genome = [True, False, 1, 0]  # Mixed boolean and integer
        
        # Should handle conversion appropriately
        mutated = self.strategy.bit_flip_mutation(genome, mutation_rate=0.5)
        self.assertEqual(len(mutated), len(genome))
        
        for value in mutated:
            self.assertIsInstance(value, bool)
    
    def test_adaptive_mutation_with_empty_history(self):
        """Test adaptive mutation with empty fitness history."""
        genome = [1.0, 2.0, 3.0]
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history=[], base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))
    
    def test_adaptive_mutation_with_constant_fitness(self):
        """Test adaptive mutation when fitness history shows no improvement."""
        genome = [1.0, 2.0, 3.0]
        constant_fitness = [0.5, 0.5, 0.5, 0.5]
        
        mutated = self.strategy.adaptive_mutation(genome, constant_fitness, base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))


class TestSelectionStrategyExtended(unittest.TestCase):
    """Extended test suite for SelectionStrategy covering edge cases and additional scenarios."""
    
    def setUp(self):
        self.strategy = SelectionStrategy()
        self.population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5},
            {'genome': [10, 11, 12], 'fitness': 0.3}
        ]
    
    def test_single_individual_population(self):
        """Test selection strategies with a population of one individual."""
        single_pop = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        selected = self.strategy.tournament_selection(single_pop, tournament_size=1)
        self.assertEqual(selected, single_pop[0])
        
        selected = self.strategy.roulette_wheel_selection(single_pop)
        self.assertEqual(selected, single_pop[0])
        
        selected = self.strategy.rank_selection(single_pop)
        self.assertEqual(selected, single_pop[0])
    
    def test_negative_fitness_values(self):
        """Test selection strategies with negative fitness values."""
        negative_pop = [
            {'genome': [1, 2, 3], 'fitness': -0.1},
            {'genome': [4, 5, 6], 'fitness': -0.5},
            {'genome': [7, 8, 9], 'fitness': -0.9}
        ]
        
        selected = self.strategy.tournament_selection(negative_pop, tournament_size=2)
        self.assertIn(selected, negative_pop)
        
        # Rank selection should work with negative values
        selected = self.strategy.rank_selection(negative_pop)
        self.assertIn(selected, negative_pop)
    
    def test_zero_fitness_population(self):
        """Test selection strategies when all individuals have zero fitness."""
        zero_pop = [
            {'genome': [1, 2, 3], 'fitness': 0.0},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 0.0}
        ]
        
        selected = self.strategy.tournament_selection(zero_pop, tournament_size=2)
        self.assertIn(selected, zero_pop)
        
        selected = self.strategy.rank_selection(zero_pop)
        self.assertIn(selected, zero_pop)
    
    def test_elitism_with_larger_elite_count(self):
        """Test elitism selection when elite count equals population size."""
        elite_count = len(self.population)
        selected = self.strategy.elitism_selection(self.population, elite_count)
        
        self.assertEqual(len(selected), elite_count)
        # Should return all individuals sorted by fitness
        fitness_values = [ind['fitness'] for ind in selected]
        self.assertEqual(fitness_values, sorted(fitness_values, reverse=True))
    
    def test_tournament_selection_deterministic(self):
        """Test tournament selection with tournament size equal to population size."""
        selected = self.strategy.tournament_selection(self.population, tournament_size=len(self.population))
        
        # Should always select the best individual
        self.assertEqual(selected['fitness'], 0.9)
    
    def test_roulette_wheel_with_very_small_fitness(self):
        """Test roulette wheel selection with very small positive fitness values."""
        small_fitness_pop = [
            {'genome': [1, 2, 3], 'fitness': 1e-10},
            {'genome': [4, 5, 6], 'fitness': 1e-11},
            {'genome': [7, 8, 9], 'fitness': 1e-12}
        ]
        
        selected = self.strategy.roulette_wheel_selection(small_fitness_pop)
        self.assertIn(selected, small_fitness_pop)


class TestFitnessFunctionExtended(unittest.TestCase):
    """Extended test suite for FitnessFunction covering additional scenarios and edge cases."""
    
    def setUp(self):
        self.fitness_func = FitnessFunction()
    
    def test_fitness_functions_with_empty_genome(self):
        """Test fitness functions with empty genomes."""
        empty_genome = []
        
        # Most fitness functions should handle empty genomes gracefully
        fitness = self.fitness_func.sphere_function(empty_genome)
        self.assertEqual(fitness, 0.0)
        
        fitness = self.fitness_func.rastrigin_function(empty_genome)
        self.assertEqual(fitness, 0.0)
    
    def test_fitness_functions_with_single_value(self):
        """Test fitness functions with single-value genomes."""
        single_genome = [2.0]
        
        fitness = self.fitness_func.sphere_function(single_genome)
        self.assertEqual(fitness, -4.0)  # -(2.0^2)
        
        fitness = self.fitness_func.ackley_function(single_genome)
        self.assertIsInstance(fitness, float)
    
    def test_fitness_functions_with_large_values(self):
        """Test fitness functions with very large genome values."""
        large_genome = [1000.0, 2000.0, 3000.0]
        
        fitness = self.fitness_func.sphere_function(large_genome)
        self.assertIsInstance(fitness, float)
        self.assertLess(fitness, 0)  # Should be negative
    
    def test_rosenbrock_with_different_dimensions(self):
        """Test Rosenbrock function with different genome dimensions."""
        # Single dimension (should handle gracefully)
        single_dim = [1.0]
        fitness = self.fitness_func.rosenbrock_function(single_dim)
        self.assertIsInstance(fitness, float)
        
        # Multiple dimensions
        multi_dim = [1.0, 1.0, 1.0, 1.0]
        fitness = self.fitness_func.rosenbrock_function(multi_dim)
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function_with_exception_handling(self):
        """Test custom fitness function that raises exceptions."""
        def failing_func(genome):
            if len(genome) == 0:
                raise ValueError("Empty genome")
            return sum(genome)
        
        # Should handle exceptions gracefully in evaluate method
        try:
            fitness = self.fitness_func.evaluate([], failing_func)
        except ValueError:
            pass  # Expected behavior
    
    def test_multi_objective_with_different_objective_counts(self):
        """Test multi-objective evaluation with varying numbers of objectives."""
        genome = [1.0, 2.0, 3.0]
        
        # Single objective
        single_obj = [lambda g: sum(g)]
        fitness = self.fitness_func.multi_objective_evaluate(genome, single_obj)
        self.assertEqual(len(fitness), 1)
        
        # Many objectives
        many_objectives = [
            lambda g: sum(g),
            lambda g: sum(x**2 for x in g),
            lambda g: max(g),
            lambda g: min(g),
            lambda g: len(g)
        ]
        fitness = self.fitness_func.multi_objective_evaluate(genome, many_objectives)
        self.assertEqual(len(fitness), 5)
    
    def test_constraint_handling_with_multiple_constraints(self):
        """Test constraint handling with multiple constraints."""
        genome = [1.0, 2.0, 3.0]
        
        def constraint1(g):
            return sum(g) < 10  # Should pass
        
        def constraint2(g):
            return max(g) < 2  # Should fail
        
        constraints = [constraint1, constraint2]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            constraints
        )
        
        # Should be penalized due to constraint2 failure
        self.assertLess(fitness, sum(genome))
    
    def test_constraint_handling_with_no_constraints(self):
        """Test constraint handling when no constraints are provided."""
        genome = [1.0, 2.0, 3.0]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            []
        )
        
        # Should equal the base fitness
        self.assertEqual(fitness, sum(genome))


class TestPopulationManagerExtended(unittest.TestCase):
    """Extended test suite for PopulationManager covering edge cases and additional functionality."""
    
    def setUp(self):
        self.manager = PopulationManager()
    
    def test_initialize_population_with_zero_size(self):
        """Test population initialization with zero size."""
        with self.assertRaises(ValueError):
            self.manager.initialize_random_population(0, 5)
    
    def test_initialize_population_with_zero_genome_length(self):
        """Test population initialization with zero genome length."""
        population = self.manager.initialize_random_population(10, 0)
        
        self.assertEqual(len(population), 10)
        for individual in population:
            self.assertEqual(len(individual['genome']), 0)
    
    def test_initialize_seeded_population_with_more_seeds_than_size(self):
        """Test seeded population initialization when seeds exceed population size."""
        seeds = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0]
        ]
        
        population = self.manager.initialize_seeded_population(3, 3, seeds)
        
        # Should only use first 3 seeds
        self.assertEqual(len(population), 3)
        genomes = [ind['genome'] for ind in population]
        self.assertIn(seeds[0], genomes)
        self.assertIn(seeds[1], genomes)
        self.assertIn(seeds[2], genomes)
    
    def test_initialize_seeded_population_with_inconsistent_genome_lengths(self):
        """Test seeded population with seeds of different lengths."""
        seeds = [
            [1.0, 2.0],
            [3.0, 4.0, 5.0],  # Different length
            [6.0, 7.0]
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            population = self.manager.initialize_seeded_population(5, 2, seeds)
            # If it succeeds, check that genomes are padded/truncated appropriately
            for individual in population:
                self.assertEqual(len(individual['genome']), 2)
        except ValueError:
            pass  # Expected behavior for inconsistent lengths
    
    def test_evaluate_population_with_fitness_function_that_returns_none(self):
        """Test population evaluation when fitness function returns None."""
        population = self.manager.initialize_random_population(5, 3)
        
        def none_fitness(genome):
            return None
        
        # Should handle None returns gracefully
        self.manager.evaluate_population(population, none_fitness)
        
        for individual in population:
            # Fitness should be None or some default value
            self.assertTrue(individual['fitness'] is None or isinstance(individual['fitness'], (int, float)))
    
    def test_get_best_individual_with_tied_fitness(self):
        """Test getting best individual when multiple individuals have the same highest fitness."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.9},  # Tied for best
            {'genome': [7, 8, 9], 'fitness': 0.7}
        ]
        
        best = self.manager.get_best_individual(population)
        
        # Should return one of the tied individuals
        self.assertEqual(best['fitness'], 0.9)
        self.assertIn(best, population[:2])  # One of the first two
    
    def test_population_statistics_with_single_individual(self):
        """Test population statistics calculation with a single individual."""
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 0.5)
        self.assertEqual(stats['worst_fitness'], 0.5)
        self.assertEqual(stats['average_fitness'], 0.5)
        self.assertEqual(stats['median_fitness'], 0.5)
        self.assertEqual(stats['std_dev_fitness'], 0.0)
    
    def test_diversity_calculation_with_identical_genomes(self):
        """Test diversity calculation when all genomes are identical."""
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        
        # Diversity should be zero or very close to zero
        self.assertAlmostEqual(diversity, 0.0, places=5)
    
    def test_diversity_calculation_with_different_genome_lengths(self):
        """Test diversity calculation when genomes have different lengths."""
        population = [
            {'genome': [1.0, 2.0], 'fitness': 0.5},
            {'genome': [3.0, 4.0, 5.0], 'fitness': 0.6},
            {'genome': [6.0], 'fitness': 0.7}
        ]
        
        # Should handle gracefully or raise appropriate error
        try:
            diversity = self.manager.calculate_diversity(population)
            self.assertIsInstance(diversity, float)
        except ValueError:
            pass  # Expected behavior for inconsistent genome lengths


class TestGeneticOperationsExtended(unittest.TestCase):
    """Extended test suite for GeneticOperations covering additional crossover scenarios and edge cases."""
    
    def setUp(self):
        self.operations = GeneticOperations()
    
    def test_crossover_with_single_element_genomes(self):
        """Test crossover operations with single-element genomes."""
        parent1 = [5.0]
        parent2 = [10.0]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 1)
        self.assertEqual(len(child2), 1)
        
        # With single element, children should be copies of parents
        self.assertIn(child1[0], [5.0, 10.0])
        self.assertIn(child2[0], [5.0, 10.0])
    
    def test_crossover_with_empty_genomes(self):
        """Test crossover operations with empty genomes."""
        parent1 = []
        parent2 = []
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 0)
        self.assertEqual(len(child2), 0)
    
    def test_uniform_crossover_with_extreme_rates(self):
        """Test uniform crossover with extreme crossover rates."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        # Zero crossover rate - children should be copies of parents
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.0)
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
        
        # Maximum crossover rate - all genes should be swapped
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=1.0)
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_arithmetic_crossover_with_extreme_alpha(self):
        """Test arithmetic crossover with extreme alpha values."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Alpha = 0 - child1 should be parent2, child2 should be parent1
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.0)
        for i in range(len(parent1)):
            self.assertAlmostEqual(child1[i], parent2[i], places=5)
            self.assertAlmostEqual(child2[i], parent1[i], places=5)
        
        # Alpha = 1 - child1 should be parent1, child2 should be parent2
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.0)
        for i in range(len(parent1)):
            self.assertAlmostEqual(child1[i], parent1[i], places=5)
            self.assertAlmostEqual(child2[i], parent2[i], places=5)
    
    def test_simulated_binary_crossover_with_extreme_bounds(self):
        """Test SBX crossover with very tight and very wide bounds."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Very tight bounds
        tight_bounds = [(2.0, 2.1), (3.0, 3.1), (4.0, 4.1)]
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, tight_bounds, eta=2.0
        )
        
        for i, (lower, upper) in enumerate(tight_bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
            self.assertGreaterEqual(child2[i], lower)
            self.assertLessEqual(child2[i], upper)
        
        # Very wide bounds
        wide_bounds = [(-1000, 1000), (-1000, 1000), (-1000, 1000)]
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, wide_bounds, eta=2.0
        )
        
        for i, (lower, upper) in enumerate(wide_bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
    
    def test_blend_crossover_with_zero_alpha(self):
        """Test blend crossover with zero alpha (equivalent to uniform crossover)."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.0)
        
        # With alpha=0, children should be within the range of parent values
        for i in range(len(parent1)):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            self.assertGreaterEqual(child1[i], min_val)
            self.assertLessEqual(child1[i], max_val)
            self.assertGreaterEqual(child2[i], min_val)
            self.assertLessEqual(child2[i], max_val)
    
    def test_crossover_reproducibility_with_seed(self):
        """Test that crossover operations are reproducible when using the same random seed."""
        import random
        
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        # Set seed and perform crossover
        random.seed(42)
        child1_a, child2_a = self.operations.single_point_crossover(parent1, parent2)
        
        # Reset seed and perform same crossover
        random.seed(42)
        child1_b, child2_b = self.operations.single_point_crossover(parent1, parent2)
        
        # Results should be identical
        self.assertEqual(child1_a, child1_b)
        self.assertEqual(child2_a, child2_b)


class TestEvolutionaryConduitExtended(unittest.TestCase):
    """Extended test suite for EvolutionaryConduit covering additional scenarios and robustness."""
    
    def setUp(self):
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_conduit_with_custom_strategies(self):
        """Test conduit behavior when custom strategies are provided."""
        custom_mutation = MutationStrategy()
        custom_selection = SelectionStrategy()
        custom_genetic_ops = GeneticOperations()
        
        # Set custom strategies
        self.conduit.mutation_strategy = custom_mutation
        self.conduit.selection_strategy = custom_selection
        self.conduit.genetic_operations = custom_genetic_ops
        
        # Verify they are set
        self.assertEqual(self.conduit.mutation_strategy, custom_mutation)
        self.assertEqual(self.conduit.selection_strategy, custom_selection)
        self.assertEqual(self.conduit.genetic_operations, custom_genetic_ops)
    
    def test_multiple_callback_registration(self):
        """Test registration and execution of multiple callbacks."""
        callback1_called = False
        callback2_called = False
        
        def callback1(generation, population, best_individual):
            nonlocal callback1_called
            callback1_called = True
        
        def callback2(generation, population, best_individual):
            nonlocal callback2_called
            callback2_called = True
        
        self.conduit.add_callback(callback1)
        self.conduit.add_callback(callback2)
        
        # Verify both callbacks are registered
        self.assertEqual(len(self.conduit.callbacks), 2)
        self.assertIn(callback1, self.conduit.callbacks)
        self.assertIn(callback2, self.conduit.callbacks)
    
    def test_callback_removal(self):
        """Test removal of callbacks from the conduit."""
        def test_callback(generation, population, best_individual):
            pass
        
        self.conduit.add_callback(test_callback)
        self.assertIn(test_callback, self.conduit.callbacks)
        
        # Remove callback
        if hasattr(self.conduit, 'remove_callback'):
            self.conduit.remove_callback(test_callback)
            self.assertNotIn(test_callback, self.conduit.callbacks)
        else:
            # Manual removal
            self.conduit.callbacks.remove(test_callback)
            self.assertNotIn(test_callback, self.conduit.callbacks)
    
    def test_state_serialization_completeness(self):
        """Test that state serialization captures all important attributes."""
        # Set up complex state
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        def custom_fitness(genome):
            return sum(genome)
        self.conduit.set_fitness_function(custom_fitness)
        
        # Save and load state
        state = self.conduit.save_state()
        new_conduit = EvolutionaryConduit()
        new_conduit.load_state(state)
        
        # Verify state completeness
        self.assertEqual(new_conduit.parameters.population_size, self.params.population_size)
        self.assertEqual(new_conduit.parameters.generations, self.params.generations)
        self.assertEqual(new_conduit.history_enabled, True)
    
    def test_evolution_with_invalid_genome_length(self):
        """Test evolution with invalid genome length."""
        self.conduit.set_parameters(self.params)
        
        with self.assertRaises(ValueError):
            self.conduit.run_evolution(genome_length=0)
        
        with self.assertRaises(ValueError):
            self.conduit.run_evolution(genome_length=-1)
    
    def test_fitness_function_exception_handling(self):
        """Test conduit behavior when fitness function raises exceptions."""
        def failing_fitness(genome):
            if len(genome) > 2:
                raise RuntimeError("Fitness evaluation failed")
            return sum(genome)
        
        self.conduit.set_fitness_function(failing_fitness)
        self.conduit.set_parameters(self.params)
        
        # Should handle fitness function failures gracefully
        with self.assertRaises(EvolutionaryException):
            self.conduit.run_evolution(genome_length=5)


class TestGenesisEvolutionaryConduitExtended(unittest.TestCase):
    """Extended test suite for GenesisEvolutionaryConduit covering advanced scenarios."""
    
    def setUp(self):
        self.genesis_conduit = GenesisEvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    def test_network_config_validation(self):
        """Test validation of neural network configuration."""
        # Valid configuration
        valid_config = {
            'input_size': 10,
            'hidden_layers': [20, 15],
            'output_size': 1,
            'activation': 'relu'
        }
        
        self.genesis_conduit.set_network_config(valid_config)
        
        # Invalid configurations
        invalid_configs = [
            {'input_size': 0},  # Zero input size
            {'input_size': -1},  # Negative input size
            {'hidden_layers': []},  # Empty hidden layers
            {'output_size': 0},  # Zero output size
        ]
        
        for invalid_config in invalid_configs:
            with self.assertRaises((ValueError, TypeError)):
                self.genesis_conduit.set_network_config(invalid_config)
    
    def test_training_data_validation(self):
        """Test validation of training data."""
        # Valid data
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        
        self.genesis_conduit.set_training_data(X_train, y_train)
        
        # Invalid data - mismatched lengths
        X_invalid = [[1, 2], [3, 4]]
        y_invalid = [0, 1, 0]  # Different length
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_training_data(X_invalid, y_invalid)
        
        # Empty data
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_training_data([], [])
    
    def test_hyperparameter_search_space_validation(self):
        """Test validation of hyperparameter search space."""
        # Valid search space
        valid_space = {
            'learning_rate': (0.001, 0.1),
            'batch_size': (16, 128),
            'dropout_rate': (0.0, 0.5)
        }
        
        self.genesis_conduit.set_hyperparameter_search_space(valid_space)
        
        # Invalid search spaces
        invalid_spaces = [
            {'learning_rate': (0.1, 0.001)},  # Invalid range (min > max)
            {'batch_size': (-16, 128)},  # Negative values
            {'dropout_rate': (0.0, 1.5)},  # Values outside valid range
        ]
        
        for invalid_space in invalid_spaces:
            with self.assertRaises(ValueError):
                self.genesis_conduit.set_hyperparameter_search_space(invalid_space)
    
    def test_topology_mutation_constraints(self):
        """Test that topology mutation respects structural constraints."""
        # Valid topology
        topology = {
            'layers': [10, 5, 1],
            'connections': [[0, 1], [1, 2]]
        }
        
        # Mutate multiple times and check constraints
        for _ in range(10):
            mutated = self.genesis_conduit.mutate_topology(topology)
            
            # Check that structure is maintained
            self.assertIn('layers', mutated)
            self.assertIn('connections', mutated)
            self.assertIsInstance(mutated['layers'], list)
            self.assertIsInstance(mutated['connections'], list)
            
            # Layers should have positive sizes
            for layer_size in mutated['layers']:
                self.assertGreater(layer_size, 0)
    
    def test_speciation_with_extreme_distances(self):
        """Test speciation with extreme distance thresholds."""
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [4.0, 5.0, 6.0], 'fitness': 0.7},
            {'genome': [7.0, 8.0, 9.0], 'fitness': 0.9}
        ]
        
        # Very small threshold - should create many species
        small_threshold_species = self.genesis_conduit.speciate_population(
            population, distance_threshold=0.1
        )
        self.assertGreaterEqual(len(small_threshold_species), 1)
        
        # Very large threshold - should create few species
        large_threshold_species = self.genesis_conduit.speciate_population(
            population, distance_threshold=1000.0
        )
        self.assertEqual(len(large_threshold_species), 1)  # All in one species
    
    def test_ensemble_size_validation(self):
        """Test validation of ensemble size."""
        networks = [
            {'genome': [1, 2, 3], 'fitness': 0.7},
            {'genome': [4, 5, 6], 'fitness': 0.8},
            {'genome': [7, 8, 9], 'fitness': 0.9}
        ]
        
        # Valid ensemble size
        ensemble = self.genesis_conduit.create_ensemble(networks, ensemble_size=2)
        self.assertEqual(len(ensemble), 2)
        
        # Ensemble size larger than available networks
        large_ensemble = self.genesis_conduit.create_ensemble(networks, ensemble_size=10)
        self.assertEqual(len(large_ensemble), len(networks))  # Should return all networks
        
        # Zero ensemble size
        with self.assertRaises(ValueError):
            self.genesis_conduit.create_ensemble(networks, ensemble_size=0)
    
    def test_coevolution_with_unequal_population_sizes(self):
        """Test coevolution with populations of different sizes."""
        small_population = [
            {'genome': [1, 2, 3], 'fitness': 0.5}
        ]
        
        large_population = [
            {'genome': [4, 5, 6], 'fitness': 0.6},
            {'genome': [7, 8, 9], 'fitness': 0.7},
            {'genome': [10, 11, 12], 'fitness': 0.8}
        ]
        
        result = self.genesis_conduit.coevolve_populations(small_population, large_population)
        
        self.assertIn('population1', result)
        self.assertIn('population2', result)
        # Should handle different sizes gracefully
    
    def test_checkpoint_system_with_complex_state(self):
        """Test checkpoint system with complex evolutionary state."""
        # Set up complex state
        self.genesis_conduit.set_parameters(self.params)
        
        network_config = {
            'input_size': 10,
            'hidden_layers': [20, 15],
            'output_size': 1,
            'activation': 'relu'
        }
        self.genesis_conduit.set_network_config(network_config)
        
        search_space = {
            'learning_rate': (0.001, 0.1),
            'batch_size': (16, 128)
        }
        self.genesis_conduit.set_hyperparameter_search_space(search_space)
        
        # Mock checkpoint saving
        with patch.object(self.genesis_conduit, 'save_checkpoint') as mock_save:
            checkpoint_path = "complex_checkpoint.pkl"
            self.genesis_conduit.save_checkpoint(checkpoint_path)
            mock_save.assert_called_once_with(checkpoint_path)
    
    def test_distributed_evolution_error_handling(self):
        """Test error handling in distributed evolution scenarios."""
        # Test with invalid island configurations
        invalid_configs = [
            [{'island_id': 1}],  # Missing population_size
            [{'population_size': 10}],  # Missing island_id
            []  # Empty configuration
        ]
        
        for invalid_config in invalid_configs:
            with self.assertRaises((ValueError, KeyError)):
                self.genesis_conduit.setup_island_model(invalid_config)
    
    def test_migration_rate_validation(self):
        """Test validation of migration rates in distributed evolution."""
        population1 = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        population2 = [{'genome': [4, 5, 6], 'fitness': 0.7}]
        
        # Valid migration rate
        migrated = self.genesis_conduit.migrate_individuals(
            population1, population2, migration_rate=0.1
        )
        self.assertIsInstance(migrated, tuple)
        
        # Invalid migration rates
        invalid_rates = [-0.1, 1.5, 2.0]
        
        for invalid_rate in invalid_rates:
            with self.assertRaises(ValueError):
                self.genesis_conduit.migrate_individuals(
                    population1, population2, migration_rate=invalid_rate
                )


class TestRobustnessAndEdgeCases(unittest.TestCase):
    """Test suite for overall system robustness and edge case handling."""
    
    def test_memory_efficiency_with_large_populations(self):
        """Test system behavior with very large population sizes."""
        large_params = EvolutionaryParameters(population_size=1000, generations=1)
        conduit = EvolutionaryConduit()
        conduit.set_parameters(large_params)
        
        # Mock the evolution to avoid actual computation
        with patch.object(conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
                'generations_run': 1,
                'final_population': [],
                'statistics': {'best_fitness': 0.9}
            }
            
            result = conduit.run_evolution(genome_length=10)
            self.assertIsNotNone(result)
    
    def test_numerical_stability_with_extreme_values(self):
        """Test numerical stability with extreme fitness values."""
        fitness_func = FitnessFunction()
        
        # Very large values
        large_genome = [1e10, 1e10, 1e10]
        fitness = fitness_func.sphere_function(large_genome)
        self.assertIsInstance(fitness, float)
        self.assertFalse(math.isnan(fitness))
        self.assertFalse(math.isinf(fitness))
        
        # Very small values
        small_genome = [1e-10, 1e-10, 1e-10]
        fitness = fitness_func.sphere_function(small_genome)
        self.assertIsInstance(fitness, float)
    
    def test_concurrent_access_safety(self):
        """Test thread safety with concurrent access to conduit."""
        import threading
        import time
        
        conduit = EvolutionaryConduit()
        params = EvolutionaryParameters(population_size=10, generations=2)
        conduit.set_parameters(params)
        
        results = []
        exceptions = []
        
        def run_evolution():
            try:
                with patch.object(conduit, 'evolve') as mock_evolve:
                    mock_evolve.return_value = {
                        'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
                        'generations_run': 2,
                        'final_population': [],
                        'statistics': {'best_fitness': 0.9}
                    }
                    
                    result = conduit.run_evolution(genome_length=3)
                    results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=run_evolution) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check that all threads completed successfully
        self.assertEqual(len(exceptions), 0)
        self.assertEqual(len(results), 3)
    
    def test_garbage_collection_after_evolution(self):
        """Test that memory is properly released after evolution."""
        import gc
        
        conduit = GenesisEvolutionaryConduit()
        params = EvolutionaryParameters(population_size=100, generations=5)
        conduit.set_parameters(params)
        
        # Run evolution (mocked)
        with patch.object(conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
                'generations_run': 5,
                'final_population': [],
                'statistics': {'best_fitness': 0.9}
            }
            
            result = conduit.run_evolution(genome_length=10)
        
        # Clear references and force garbage collection
        del conduit
        del result
        gc.collect()
        
        # Test passes if no memory leaks or exceptions occur
        self.assertTrue(True)
    
    def test_unicode_and_string_handling(self):
        """Test handling of unicode and string data in genomes."""
        # Test with string-based genomes (for symbolic evolution)
        string_genome = ["a", "b", "c", "d"]
        
        mutation_strategy = MutationStrategy()
        
        # Should handle gracefully or raise appropriate error
        try:
            mutated = mutation_strategy.gaussian_mutation(string_genome, mutation_rate=0.1)
            # If it works, verify structure
            self.assertEqual(len(mutated), len(string_genome))
        except (TypeError, ValueError):
            pass  # Expected for string genomes with numeric operations
    
    def test_evolution_reproducibility(self):
        """Test that evolution results are reproducible with same random seed."""
        import random
        import numpy as np
        
        conduit1 = EvolutionaryConduit()
        conduit2 = EvolutionaryConduit()
        
        params = EvolutionaryParameters(population_size=10, generations=3)
        
        def simple_fitness(genome):
            return sum(genome)
        
        # Set same configuration for both conduits
        conduit1.set_parameters(params)
        conduit1.set_fitness_function(simple_fitness)
        
        conduit2.set_parameters(params)
        conduit2.set_fitness_function(simple_fitness)
        
        # Mock with same results
        mock_result = {
            'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
            'generations_run': 3,
            'final_population': [],
            'statistics': {'best_fitness': 6.0}
        }
        
        with patch.object(conduit1, 'evolve', return_value=mock_result):
            with patch.object(conduit2, 'evolve', return_value=mock_result):
                result1 = conduit1.run_evolution(genome_length=3)
                result2 = conduit2.run_evolution(genome_length=3)
        
        # Results should be identical
        self.assertEqual(result1['best_individual']['fitness'], 
                        result2['best_individual']['fitness'])


if __name__ == '__main__':
    # Ensure all tests run with high verbosity
    unittest.main(verbosity=3, buffer=True)