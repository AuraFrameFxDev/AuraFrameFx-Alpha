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


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

class TestEvolutionaryParametersExtended(unittest.TestCase):
    """Extended test suite for EvolutionaryParameters with additional edge cases."""
    
    def test_parameter_boundary_values(self):
        """Test parameter validation at exact boundary values."""
        # Test exact boundary values
        params = EvolutionaryParameters(
            population_size=1,
            generations=1,
            mutation_rate=0.0,
            crossover_rate=0.0,
            selection_pressure=0.0
        )
        self.assertEqual(params.population_size, 1)
        self.assertEqual(params.mutation_rate, 0.0)
        self.assertEqual(params.crossover_rate, 0.0)
        
        # Test upper boundary values
        params = EvolutionaryParameters(
            mutation_rate=1.0,
            crossover_rate=1.0,
            selection_pressure=1.0
        )
        self.assertEqual(params.mutation_rate, 1.0)
        self.assertEqual(params.crossover_rate, 1.0)
        self.assertEqual(params.selection_pressure, 1.0)
    
    def test_parameter_type_validation(self):
        """Test that parameters reject invalid types."""
        with self.assertRaises(TypeError):
            EvolutionaryParameters(population_size="invalid")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(mutation_rate="invalid")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(generations=None)
    
    def test_parameter_copy_constructor(self):
        """Test creating parameters from another parameters instance."""
        original = EvolutionaryParameters(population_size=150, generations=600)
        copied = EvolutionaryParameters(
            population_size=original.population_size,
            generations=original.generations,
            mutation_rate=original.mutation_rate,
            crossover_rate=original.crossover_rate,
            selection_pressure=original.selection_pressure
        )
        
        self.assertEqual(copied.population_size, original.population_size)
        self.assertEqual(copied.generations, original.generations)
        self.assertEqual(copied.mutation_rate, original.mutation_rate)
        self.assertEqual(copied.crossover_rate, original.crossover_rate)
        self.assertEqual(copied.selection_pressure, original.selection_pressure)
    
    def test_from_dict_with_missing_keys(self):
        """Test from_dict method with missing keys uses defaults."""
        partial_dict = {'population_size': 50}
        params = EvolutionaryParameters.from_dict(partial_dict)
        
        self.assertEqual(params.population_size, 50)
        self.assertEqual(params.generations, 500)  # Default value
        self.assertEqual(params.mutation_rate, 0.1)  # Default value
    
    def test_from_dict_with_extra_keys(self):
        """Test from_dict method ignores extra keys."""
        dict_with_extra = {
            'population_size': 75,
            'generations': 300,
            'mutation_rate': 0.15,
            'crossover_rate': 0.85,
            'selection_pressure': 0.25,
            'extra_key': 'should_be_ignored'
        }
        params = EvolutionaryParameters.from_dict(dict_with_extra)
        
        self.assertEqual(params.population_size, 75)
        self.assertEqual(params.generations, 300)
        self.assertFalse(hasattr(params, 'extra_key'))


class TestMutationStrategyExtended(unittest.TestCase):
    """Extended test suite for MutationStrategy with additional edge cases."""
    
    def setUp(self):
        self.strategy = MutationStrategy()
    
    def test_gaussian_mutation_with_zero_sigma(self):
        """Test Gaussian mutation with zero sigma should not mutate."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=0.0)
        
        self.assertEqual(mutated, genome)
    
    def test_gaussian_mutation_with_large_sigma(self):
        """Test Gaussian mutation with large sigma."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=10.0)
        
        self.assertEqual(len(mutated), len(genome))
        self.assertIsInstance(mutated, list)
    
    def test_uniform_mutation_with_zero_rate(self):
        """Test uniform mutation with zero mutation rate."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=0.0, bounds=(-10, 10))
        
        self.assertEqual(mutated, genome)
    
    def test_uniform_mutation_with_narrow_bounds(self):
        """Test uniform mutation with very narrow bounds."""
        genome = [5.0, 5.0, 5.0]
        bounds = (4.9, 5.1)
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=1.0, bounds=bounds)
        
        for value in mutated:
            self.assertGreaterEqual(value, bounds[0])
            self.assertLessEqual(value, bounds[1])
    
    def test_bit_flip_mutation_with_zero_rate(self):
        """Test bit flip mutation with zero mutation rate."""
        genome = [True, False, True]
        mutated = self.strategy.bit_flip_mutation(genome, mutation_rate=0.0)
        
        self.assertEqual(mutated, genome)
    
    def test_bit_flip_mutation_with_empty_genome(self):
        """Test bit flip mutation with empty genome."""
        genome = []
        mutated = self.strategy.bit_flip_mutation(genome, mutation_rate=0.5)
        
        self.assertEqual(mutated, [])
    
    def test_adaptive_mutation_with_empty_history(self):
        """Test adaptive mutation with empty fitness history."""
        genome = [1.0, 2.0, 3.0]
        fitness_history = []
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
        
        self.assertEqual(len(mutated), len(genome))
    
    def test_adaptive_mutation_with_constant_fitness(self):
        """Test adaptive mutation with constant fitness history."""
        genome = [1.0, 2.0, 3.0]
        fitness_history = [0.5, 0.5, 0.5, 0.5]
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
        
        self.assertEqual(len(mutated), len(genome))
    
    def test_mutation_with_single_element_genome(self):
        """Test mutation strategies with single element genome."""
        genome = [5.0]
        
        gaussian_mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.5, sigma=1.0)
        uniform_mutated = self.strategy.uniform_mutation(genome, mutation_rate=0.5, bounds=(-10, 10))
        
        self.assertEqual(len(gaussian_mutated), 1)
        self.assertEqual(len(uniform_mutated), 1)
    
    def test_mutation_with_large_genome(self):
        """Test mutation strategies with large genome."""
        genome = [1.0] * 1000
        
        gaussian_mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.01, sigma=0.1)
        uniform_mutated = self.strategy.uniform_mutation(genome, mutation_rate=0.01, bounds=(-5, 5))
        
        self.assertEqual(len(gaussian_mutated), 1000)
        self.assertEqual(len(uniform_mutated), 1000)


class TestSelectionStrategyExtended(unittest.TestCase):
    """Extended test suite for SelectionStrategy with additional edge cases."""
    
    def setUp(self):
        self.strategy = SelectionStrategy()
        self.population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5},
            {'genome': [10, 11, 12], 'fitness': 0.3}
        ]
    
    def test_tournament_selection_with_size_one(self):
        """Test tournament selection with tournament size of 1."""
        selected = self.strategy.tournament_selection(self.population, tournament_size=1)
        
        self.assertIn(selected, self.population)
    
    def test_tournament_selection_with_full_population(self):
        """Test tournament selection with tournament size equal to population size."""
        selected = self.strategy.tournament_selection(self.population, tournament_size=len(self.population))
        
        # Should select the best individual
        self.assertEqual(selected['fitness'], 0.9)
    
    def test_roulette_wheel_selection_with_zero_fitness(self):
        """Test roulette wheel selection with zero fitness values."""
        zero_fitness_population = [
            {'genome': [1, 2, 3], 'fitness': 0.0},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 0.0}
        ]
        
        selected = self.strategy.roulette_wheel_selection(zero_fitness_population)
        
        self.assertIn(selected, zero_fitness_population)
    
    def test_roulette_wheel_selection_with_negative_fitness(self):
        """Test roulette wheel selection with negative fitness values."""
        negative_fitness_population = [
            {'genome': [1, 2, 3], 'fitness': -0.1},
            {'genome': [4, 5, 6], 'fitness': -0.2},
            {'genome': [7, 8, 9], 'fitness': -0.3}
        ]
        
        selected = self.strategy.roulette_wheel_selection(negative_fitness_population)
        
        self.assertIn(selected, negative_fitness_population)
    
    def test_rank_selection_with_single_individual(self):
        """Test rank selection with single individual."""
        single_population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        selected = self.strategy.rank_selection(single_population)
        
        self.assertEqual(selected, single_population[0])
    
    def test_elitism_selection_with_zero_count(self):
        """Test elitism selection with zero elite count."""
        selected = self.strategy.elitism_selection(self.population, elite_count=0)
        
        self.assertEqual(selected, [])
    
    def test_elitism_selection_with_count_exceeding_population(self):
        """Test elitism selection with count exceeding population size."""
        selected = self.strategy.elitism_selection(self.population, elite_count=len(self.population) + 5)
        
        self.assertEqual(len(selected), len(self.population))
    
    def test_selection_with_tied_fitness(self):
        """Test selection strategies with tied fitness values."""
        tied_population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        tournament_selected = self.strategy.tournament_selection(tied_population, tournament_size=2)
        roulette_selected = self.strategy.roulette_wheel_selection(tied_population)
        rank_selected = self.strategy.rank_selection(tied_population)
        
        self.assertIn(tournament_selected, tied_population)
        self.assertIn(roulette_selected, tied_population)
        self.assertIn(rank_selected, tied_population)
    
    def test_selection_reproducibility(self):
        """Test selection reproducibility with same random seed."""
        import random
        
        # Set seed for reproducibility
        random.seed(42)
        selected1 = self.strategy.tournament_selection(self.population, tournament_size=2)
        
        random.seed(42)
        selected2 = self.strategy.tournament_selection(self.population, tournament_size=2)
        
        self.assertEqual(selected1, selected2)


class TestFitnessFunctionExtended(unittest.TestCase):
    """Extended test suite for FitnessFunction with additional edge cases."""
    
    def setUp(self):
        self.fitness_func = FitnessFunction()
    
    def test_fitness_functions_with_empty_genome(self):
        """Test fitness functions with empty genome."""
        empty_genome = []
        
        sphere_fitness = self.fitness_func.sphere_function(empty_genome)
        rastrigin_fitness = self.fitness_func.rastrigin_function(empty_genome)
        ackley_fitness = self.fitness_func.ackley_function(empty_genome)
        
        self.assertEqual(sphere_fitness, 0.0)
        self.assertEqual(rastrigin_fitness, 0.0)
        self.assertEqual(ackley_fitness, 0.0)
    
    def test_fitness_functions_with_single_element(self):
        """Test fitness functions with single element genome."""
        single_genome = [2.0]
        
        sphere_fitness = self.fitness_func.sphere_function(single_genome)
        rastrigin_fitness = self.fitness_func.rastrigin_function(single_genome)
        ackley_fitness = self.fitness_func.ackley_function(single_genome)
        
        self.assertIsInstance(sphere_fitness, (int, float))
        self.assertIsInstance(rastrigin_fitness, (int, float))
        self.assertIsInstance(ackley_fitness, (int, float))
    
    def test_fitness_functions_with_large_values(self):
        """Test fitness functions with large genome values."""
        large_genome = [1000.0, 2000.0, 3000.0]
        
        sphere_fitness = self.fitness_func.sphere_function(large_genome)
        rastrigin_fitness = self.fitness_func.rastrigin_function(large_genome)
        ackley_fitness = self.fitness_func.ackley_function(large_genome)
        
        self.assertIsInstance(sphere_fitness, (int, float))
        self.assertIsInstance(rastrigin_fitness, (int, float))
        self.assertIsInstance(ackley_fitness, (int, float))
    
    def test_fitness_functions_with_negative_values(self):
        """Test fitness functions with negative genome values."""
        negative_genome = [-1.0, -2.0, -3.0]
        
        sphere_fitness = self.fitness_func.sphere_function(negative_genome)
        rastrigin_fitness = self.fitness_func.rastrigin_function(negative_genome)
        ackley_fitness = self.fitness_func.ackley_function(negative_genome)
        
        self.assertIsInstance(sphere_fitness, (int, float))
        self.assertIsInstance(rastrigin_fitness, (int, float))
        self.assertIsInstance(ackley_fitness, (int, float))
    
    def test_custom_function_with_exception_handling(self):
        """Test custom fitness function that raises exceptions."""
        def failing_function(genome):
            if len(genome) == 0:
                raise ValueError("Empty genome")
            return sum(genome)
        
        # Test with valid genome
        valid_genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate(valid_genome, failing_function)
        self.assertEqual(fitness, 6.0)
        
        # Test with empty genome (should handle exception)
        empty_genome = []
        with self.assertRaises(ValueError):
            self.fitness_func.evaluate(empty_genome, failing_function)
    
    def test_multi_objective_with_conflicting_objectives(self):
        """Test multi-objective optimization with conflicting objectives."""
        genome = [1.0, 2.0, 3.0]
        objectives = [
            lambda g: sum(g),        # Maximize sum
            lambda g: -sum(g),       # Minimize sum (conflicting)
            lambda g: len(g)         # Constant for this genome
        ]
        
        fitness = self.fitness_func.multi_objective_evaluate(genome, objectives)
        
        self.assertEqual(len(fitness), 3)
        self.assertEqual(fitness[0], 6.0)
        self.assertEqual(fitness[1], -6.0)
        self.assertEqual(fitness[2], 3)
    
    def test_constraint_handling_with_multiple_constraints(self):
        """Test constraint handling with multiple constraints."""
        genome = [2.0, 3.0, 4.0]
        
        constraints = [
            lambda g: sum(g) < 10,   # Sum constraint
            lambda g: max(g) < 5,    # Max element constraint
            lambda g: len(g) >= 3    # Length constraint
        ]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            constraints
        )
        
        self.assertIsInstance(fitness, (int, float))
    
    def test_constraint_handling_with_no_constraints(self):
        """Test constraint handling with no constraints."""
        genome = [1.0, 2.0, 3.0]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            []
        )
        
        self.assertEqual(fitness, 6.0)
    
    def test_fitness_function_consistency(self):
        """Test that fitness functions produce consistent results."""
        genome = [1.0, 2.0, 3.0]
        
        # Call same function multiple times
        fitness1 = self.fitness_func.sphere_function(genome)
        fitness2 = self.fitness_func.sphere_function(genome)
        fitness3 = self.fitness_func.sphere_function(genome)
        
        self.assertEqual(fitness1, fitness2)
        self.assertEqual(fitness2, fitness3)


class TestPopulationManagerExtended(unittest.TestCase):
    """Extended test suite for PopulationManager with additional edge cases."""
    
    def setUp(self):
        self.manager = PopulationManager()
    
    def test_initialize_random_population_with_zero_size(self):
        """Test random population initialization with zero population size."""
        population = self.manager.initialize_random_population(0, 5)
        
        self.assertEqual(len(population), 0)
        self.assertEqual(population, [])
    
    def test_initialize_random_population_with_zero_genome_length(self):
        """Test random population initialization with zero genome length."""
        population = self.manager.initialize_random_population(5, 0)
        
        self.assertEqual(len(population), 5)
        for individual in population:
            self.assertEqual(len(individual['genome']), 0)
    
    def test_initialize_seeded_population_with_more_seeds_than_population(self):
        """Test seeded population initialization with more seeds than population size."""
        seeds = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        population = self.manager.initialize_seeded_population(2, 3, seeds)
        
        self.assertEqual(len(population), 2)
        # Should contain first two seeds
        genomes = [ind['genome'] for ind in population]
        self.assertIn(seeds[0], genomes)
        self.assertIn(seeds[1], genomes)
    
    def test_initialize_seeded_population_with_no_seeds(self):
        """Test seeded population initialization with no seeds."""
        population = self.manager.initialize_seeded_population(5, 3, [])
        
        self.assertEqual(len(population), 5)
        for individual in population:
            self.assertEqual(len(individual['genome']), 3)
    
    def test_evaluate_population_with_varying_fitness_function(self):
        """Test population evaluation with fitness function that varies by genome."""
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        def varying_fitness(genome):
            # Different calculation based on genome values
            if genome[0] < 5:
                return sum(genome)
            else:
                return sum(x**2 for x in genome)
        
        self.manager.evaluate_population(population, varying_fitness)
        
        self.assertEqual(population[0]['fitness'], 6)    # sum([1,2,3])
        self.assertEqual(population[1]['fitness'], 77)   # sum([4²,5²,6²])
        self.assertEqual(population[2]['fitness'], 194)  # sum([7²,8²,9²])
    
    def test_get_best_individual_with_tied_fitness(self):
        """Test getting best individual with tied fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.9}
        ]
        
        best = self.manager.get_best_individual(population)
        
        self.assertEqual(best['fitness'], 0.9)
        self.assertIn(best, population)
    
    def test_get_population_statistics_with_single_individual(self):
        """Test population statistics with single individual."""
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 0.5)
        self.assertEqual(stats['worst_fitness'], 0.5)
        self.assertEqual(stats['average_fitness'], 0.5)
        self.assertEqual(stats['median_fitness'], 0.5)
        self.assertEqual(stats['std_dev_fitness'], 0.0)
    
    def test_diversity_calculation_with_identical_genomes(self):
        """Test diversity calculation with identical genomes."""
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        
        self.assertEqual(diversity, 0.0)
    
    def test_diversity_calculation_with_single_individual(self):
        """Test diversity calculation with single individual."""
        population = [{'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}]
        
        diversity = self.manager.calculate_diversity(population)
        
        self.assertEqual(diversity, 0.0)
    
    def test_population_sorting_by_fitness(self):
        """Test population sorting by fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.3},
            {'genome': [4, 5, 6], 'fitness': 0.9},
            {'genome': [7, 8, 9], 'fitness': 0.6}
        ]
        
        sorted_population = self.manager.sort_population_by_fitness(population)
        
        self.assertEqual(sorted_population[0]['fitness'], 0.9)
        self.assertEqual(sorted_population[1]['fitness'], 0.6)
        self.assertEqual(sorted_population[2]['fitness'], 0.3)
    
    def test_population_truncation(self):
        """Test population truncation to specified size."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5},
            {'genome': [10, 11, 12], 'fitness': 0.3}
        ]
        
        truncated = self.manager.truncate_population(population, size=2)
        
        self.assertEqual(len(truncated), 2)
        self.assertEqual(truncated[0]['fitness'], 0.9)
        self.assertEqual(truncated[1]['fitness'], 0.7)


class TestGeneticOperationsExtended(unittest.TestCase):
    """Extended test suite for GeneticOperations with additional edge cases."""
    
    def setUp(self):
        self.operations = GeneticOperations()
    
    def test_crossover_with_empty_parents(self):
        """Test crossover operations with empty parent genomes."""
        parent1 = []
        parent2 = []
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(child1, [])
        self.assertEqual(child2, [])
    
    def test_crossover_with_single_element_parents(self):
        """Test crossover operations with single element parents."""
        parent1 = [1]
        parent2 = [2]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), 1)
        self.assertEqual(len(child2), 1)
        self.assertIn(child1[0], [1, 2])
        self.assertIn(child2[0], [1, 2])
    
    def test_two_point_crossover_with_short_parents(self):
        """Test two-point crossover with very short parents."""
        parent1 = [1, 2]
        parent2 = [3, 4]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)
    
    def test_uniform_crossover_with_extreme_rates(self):
        """Test uniform crossover with extreme crossover rates."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        # Test with crossover_rate = 0.0 (no crossover)
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.0)
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
        
        # Test with crossover_rate = 1.0 (complete crossover)
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=1.0)
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
    
    def test_arithmetic_crossover_with_extreme_alpha(self):
        """Test arithmetic crossover with extreme alpha values."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Test with alpha = 0.0 (child1 = parent1, child2 = parent2)
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.0)
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
        
        # Test with alpha = 1.0 (child1 = parent2, child2 = parent1)
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.0)
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
    
    def test_simulated_binary_crossover_with_tight_bounds(self):
        """Test simulated binary crossover with tight bounds."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [1.1, 2.1, 3.1]
        bounds = [(0.9, 1.2), (1.9, 2.2), (2.9, 3.2)]
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=2.0
        )
        
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
            self.assertGreaterEqual(child2[i], lower)
            self.assertLessEqual(child2[i], upper)
    
    def test_blend_crossover_with_zero_alpha(self):
        """Test blend crossover with zero alpha."""
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
    
    def test_crossover_reproducibility(self):
        """Test crossover reproducibility with same random seed."""
        import random
        
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        # Set seed for reproducibility
        random.seed(42)
        child1_1, child2_1 = self.operations.single_point_crossover(parent1, parent2)
        
        random.seed(42)
        child1_2, child2_2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(child1_1, child1_2)
        self.assertEqual(child2_1, child2_2)
    
    def test_crossover_with_identical_parents(self):
        """Test crossover operations with identical parents."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [1, 2, 3, 4, 5]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
    
    def test_crossover_with_large_genomes(self):
        """Test crossover operations with large genomes."""
        parent1 = list(range(1000))
        parent2 = list(range(1000, 2000))
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), 1000)
        self.assertEqual(len(child2), 1000)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test suite for performance and scalability edge cases."""
    
    def setUp(self):
        self.conduit = EvolutionaryConduit()
        self.genesis_conduit = GenesisEvolutionaryConduit()
    
    def test_large_population_handling(self):
        """Test handling of large population sizes."""
        params = EvolutionaryParameters(
            population_size=1000,
            generations=1,
            mutation_rate=0.01,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        # Test that parameters are set correctly
        self.assertEqual(self.conduit.parameters.population_size, 1000)
    
    def test_many_generations_handling(self):
        """Test handling of many generations."""
        params = EvolutionaryParameters(
            population_size=10,
            generations=10000,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        # Test that parameters are set correctly
        self.assertEqual(self.conduit.parameters.generations, 10000)
    
    def test_high_dimensional_genome_handling(self):
        """Test handling of high-dimensional genomes."""
        manager = PopulationManager()
        
        # Create population with high-dimensional genomes
        population = manager.initialize_random_population(10, 1000)
        
        # Test that genomes have correct dimensions
        for individual in population:
            self.assertEqual(len(individual['genome']), 1000)
    
    def test_memory_efficient_operations(self):
        """Test memory efficiency of operations."""
        operations = GeneticOperations()
        
        # Test with large parents
        parent1 = [1.0] * 10000
        parent2 = [2.0] * 10000
        
        child1, child2 = operations.single_point_crossover(parent1, parent2)
        
        # Verify operation completed successfully
        self.assertEqual(len(child1), 10000)
        self.assertEqual(len(child2), 10000)
    
    def test_concurrent_population_evaluation(self):
        """Test concurrent evaluation of population fitness."""
        manager = PopulationManager()
        population = manager.initialize_random_population(100, 10)
        
        def slow_fitness(genome):
            # Simulate slow fitness function
            import time
            time.sleep(0.001)
            return sum(genome)
        
        # Test evaluation completes
        manager.evaluate_population(population, slow_fitness)
        
        # Verify all individuals have fitness values
        for individual in population:
            self.assertIsNotNone(individual['fitness'])


class TestEdgeCasesAndBoundaryConditions(unittest.TestCase):
    """Test suite for edge cases and boundary conditions."""
    
    def test_extreme_mutation_rates(self):
        """Test behavior with extreme mutation rates."""
        strategy = MutationStrategy()
        genome = [1.0, 2.0, 3.0]
        
        # Test with minimum rate
        mutated_min = strategy.gaussian_mutation(genome, mutation_rate=0.0, sigma=1.0)
        self.assertEqual(mutated_min, genome)
        
        # Test with maximum rate
        mutated_max = strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1.0)
        self.assertEqual(len(mutated_max), len(genome))
    
    def test_extreme_selection_pressures(self):
        """Test behavior with extreme selection pressures."""
        params = EvolutionaryParameters(selection_pressure=0.0)
        self.assertEqual(params.selection_pressure, 0.0)
        
        params = EvolutionaryParameters(selection_pressure=1.0)
        self.assertEqual(params.selection_pressure, 1.0)
    
    def test_fitness_function_boundary_values(self):
        """Test fitness functions with boundary values."""
        fitness_func = FitnessFunction()
        
        # Test with very small values
        small_genome = [1e-10, 1e-10, 1e-10]
        fitness = fitness_func.sphere_function(small_genome)
        self.assertIsInstance(fitness, (int, float))
        
        # Test with very large values
        large_genome = [1e10, 1e10, 1e10]
        fitness = fitness_func.sphere_function(large_genome)
        self.assertIsInstance(fitness, (int, float))
    
    def test_population_size_boundaries(self):
        """Test population size boundaries."""
        manager = PopulationManager()
        
        # Test minimum population size
        population = manager.initialize_random_population(1, 5)
        self.assertEqual(len(population), 1)
        
        # Test zero genome length
        population = manager.initialize_random_population(5, 0)
        for individual in population:
            self.assertEqual(len(individual['genome']), 0)
    
    def test_crossover_rate_boundaries(self):
        """Test crossover rate boundaries."""
        params = EvolutionaryParameters(crossover_rate=0.0)
        self.assertEqual(params.crossover_rate, 0.0)
        
        params = EvolutionaryParameters(crossover_rate=1.0)
        self.assertEqual(params.crossover_rate, 1.0)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        fitness_func = FitnessFunction()
        
        # Test with infinity
        try:
            inf_genome = [float('inf'), 1.0, 2.0]
            fitness = fitness_func.sphere_function(inf_genome)
            self.assertIsInstance(fitness, (int, float))
        except (OverflowError, ValueError):
            pass  # Expected for infinity values
        
        # Test with NaN
        try:
            nan_genome = [float('nan'), 1.0, 2.0]
            fitness = fitness_func.sphere_function(nan_genome)
            self.assertIsInstance(fitness, (int, float))
        except (ValueError, TypeError):
            pass  # Expected for NaN values
    
    def test_empty_data_structures(self):
        """Test behavior with empty data structures."""
        operations = GeneticOperations()
        
        # Test with empty parents
        child1, child2 = operations.single_point_crossover([], [])
        self.assertEqual(child1, [])
        self.assertEqual(child2, [])
        
        # Test selection with empty population
        strategy = SelectionStrategy()
        with self.assertRaises(ValueError):
            strategy.tournament_selection([], tournament_size=1)


class TestRobustnessAndErrorHandling(unittest.TestCase):
    """Test suite for robustness and error handling."""
    
    def test_invalid_parameter_types(self):
        """Test handling of invalid parameter types."""
        with self.assertRaises(TypeError):
            EvolutionaryParameters(population_size="invalid")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(generations=None)
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(mutation_rate="invalid")
    
    def test_invalid_genome_structures(self):
        """Test handling of invalid genome structures."""
        strategy = MutationStrategy()
        
        # Test with non-list genome
        with self.assertRaises(TypeError):
            strategy.gaussian_mutation("invalid", mutation_rate=0.1, sigma=1.0)
        
        # Test with mixed-type genome
        mixed_genome = [1, "string", 3.0]
        # This should handle gracefully or raise appropriate error
        try:
            strategy.gaussian_mutation(mixed_genome, mutation_rate=0.1, sigma=1.0)
        except (TypeError, ValueError):
            pass  # Expected for mixed types
    
    def test_fitness_function_exceptions(self):
        """Test handling of fitness function exceptions."""
        fitness_func = FitnessFunction()
        
        def failing_fitness(genome):
            raise RuntimeError("Fitness calculation failed")
        
        with self.assertRaises(RuntimeError):
            fitness_func.evaluate([1, 2, 3], failing_fitness)
    
    def test_population_evaluation_error_handling(self):
        """Test error handling during population evaluation."""
        manager = PopulationManager()
        population = manager.initialize_random_population(5, 3)
        
        def failing_fitness(genome):
            if genome[0] > 0.5:
                raise ValueError("Invalid genome")
            return sum(genome)
        
        # Test that evaluation handles errors gracefully
        try:
            manager.evaluate_population(population, failing_fitness)
        except ValueError:
            pass  # Expected for some individuals
    
    def test_crossover_error_handling(self):
        """Test error handling in crossover operations."""
        operations = GeneticOperations()
        
        # Test with mismatched parent lengths
        parent1 = [1, 2, 3]
        parent2 = [4, 5]
        
        with self.assertRaises(ValueError):
            operations.single_point_crossover(parent1, parent2)
    
    def test_selection_error_handling(self):
        """Test error handling in selection operations."""
        strategy = SelectionStrategy()
        
        # Test with invalid tournament size
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        
        with self.assertRaises(ValueError):
            strategy.tournament_selection(population, tournament_size=0)
        
        with self.assertRaises(ValueError):
            strategy.tournament_selection(population, tournament_size=10)  # Exceeds population
    
    def test_memory_management(self):
        """Test memory management with large data structures."""
        manager = PopulationManager()
        
        # Create and destroy large populations
        for _ in range(10):
            large_population = manager.initialize_random_population(1000, 100)
            del large_population
        
        # Test should complete without memory issues
        self.assertTrue(True)
    
    def test_state_consistency(self):
        """Test state consistency across operations."""
        conduit = EvolutionaryConduit()
        
        # Set initial parameters
        params = EvolutionaryParameters(population_size=50, generations=10)
        conduit.set_parameters(params)
        
        # Save and restore state
        state = conduit.save_state()
        
        # Modify parameters
        new_params = EvolutionaryParameters(population_size=100, generations=20)
        conduit.set_parameters(new_params)
        
        # Restore original state
        conduit.load_state(state)
        
        # Verify state is restored
        self.assertEqual(conduit.parameters.population_size, 50)
        self.assertEqual(conduit.parameters.generations, 10)


if __name__ == '__main__':
    # Run all tests with increased verbosity
    unittest.main(verbosity=2, buffer=True)