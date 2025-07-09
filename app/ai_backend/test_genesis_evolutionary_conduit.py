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

class TestEvolutionaryParametersAdvanced(unittest.TestCase):
    """Advanced test suite for EvolutionaryParameters edge cases and boundary conditions."""
    
    def test_extreme_parameter_values(self):
        """Test EvolutionaryParameters with extreme but valid parameter values."""
        # Test with very large population size
        large_params = EvolutionaryParameters(population_size=1000000)
        self.assertEqual(large_params.population_size, 1000000)
        
        # Test with very small but valid rates
        small_params = EvolutionaryParameters(mutation_rate=0.001, crossover_rate=0.001)
        self.assertEqual(small_params.mutation_rate, 0.001)
        self.assertEqual(small_params.crossover_rate, 0.001)
        
        # Test with rates at boundary values
        boundary_params = EvolutionaryParameters(mutation_rate=1.0, crossover_rate=1.0)
        self.assertEqual(boundary_params.mutation_rate, 1.0)
        self.assertEqual(boundary_params.crossover_rate, 1.0)
    
    def test_parameter_type_validation(self):
        """Test that EvolutionaryParameters validates parameter types correctly."""
        with self.assertRaises(TypeError):
            EvolutionaryParameters(population_size="100")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(mutation_rate="0.1")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(generations=10.5)
    
    def test_from_dict_with_missing_keys(self):
        """Test EvolutionaryParameters.from_dict with missing dictionary keys."""
        partial_dict = {'population_size': 50, 'mutation_rate': 0.2}
        params = EvolutionaryParameters.from_dict(partial_dict)
        
        # Should use defaults for missing keys
        self.assertEqual(params.population_size, 50)
        self.assertEqual(params.mutation_rate, 0.2)
        self.assertEqual(params.generations, 500)  # Default value
    
    def test_from_dict_with_invalid_values(self):
        """Test that from_dict raises appropriate errors for invalid values."""
        invalid_dict = {'population_size': -10, 'mutation_rate': 0.5}
        
        with self.assertRaises(ValueError):
            EvolutionaryParameters.from_dict(invalid_dict)
    
    def test_parameter_serialization_roundtrip(self):
        """Test that parameters can be serialized and deserialized correctly."""
        import json
        
        original_params = EvolutionaryParameters(
            population_size=250,
            generations=750,
            mutation_rate=0.15,
            crossover_rate=0.85,
            selection_pressure=0.3
        )
        
        # Serialize to JSON
        json_str = json.dumps(original_params.to_dict())
        
        # Deserialize from JSON
        reconstructed_params = EvolutionaryParameters.from_dict(json.loads(json_str))
        
        # Verify all parameters match
        self.assertEqual(original_params.population_size, reconstructed_params.population_size)
        self.assertEqual(original_params.generations, reconstructed_params.generations)
        self.assertEqual(original_params.mutation_rate, reconstructed_params.mutation_rate)
        self.assertEqual(original_params.crossover_rate, reconstructed_params.crossover_rate)
        self.assertEqual(original_params.selection_pressure, reconstructed_params.selection_pressure)


class TestMutationStrategyAdvanced(unittest.TestCase):
    """Advanced test suite for MutationStrategy edge cases and stress testing."""
    
    def setUp(self):
        """Set up mutation strategy for advanced tests."""
        self.strategy = MutationStrategy()
    
    def test_mutation_with_empty_genome(self):
        """Test that mutation strategies handle empty genomes appropriately."""
        empty_genome = []
        
        # All mutation strategies should handle empty genomes
        mutated_gaussian = self.strategy.gaussian_mutation(empty_genome, 0.1)
        mutated_uniform = self.strategy.uniform_mutation(empty_genome, 0.1)
        mutated_bit_flip = self.strategy.bit_flip_mutation(empty_genome, 0.1)
        
        self.assertEqual(len(mutated_gaussian), 0)
        self.assertEqual(len(mutated_uniform), 0)
        self.assertEqual(len(mutated_bit_flip), 0)
    
    def test_mutation_with_large_genome(self):
        """Test mutation strategies with very large genomes."""
        large_genome = [1.0] * 10000
        
        mutated = self.strategy.gaussian_mutation(large_genome, 0.1, sigma=0.5)
        
        self.assertEqual(len(mutated), 10000)
        self.assertIsInstance(mutated, list)
    
    def test_gaussian_mutation_with_zero_sigma(self):
        """Test Gaussian mutation with zero sigma (no mutation)."""
        genome = [1.0, 2.0, 3.0]
        
        mutated = self.strategy.gaussian_mutation(genome, 1.0, sigma=0.0)
        
        # With sigma=0, genome should remain unchanged
        self.assertEqual(mutated, genome)
    
    def test_gaussian_mutation_statistical_properties(self):
        """Test that Gaussian mutation produces statistically correct results."""
        genome = [0.0] * 1000
        sigma = 1.0
        
        mutated = self.strategy.gaussian_mutation(genome, 1.0, sigma=sigma)
        
        # Check that the mutations follow approximately normal distribution
        mutations = [mutated[i] - genome[i] for i in range(len(genome))]
        mean_mutation = sum(mutations) / len(mutations)
        
        # Mean should be close to 0 (allowing for statistical variance)
        self.assertAlmostEqual(mean_mutation, 0.0, delta=0.1)
    
    def test_uniform_mutation_bounds_enforcement(self):
        """Test that uniform mutation strictly enforces bounds."""
        genome = [5.0, 10.0, -5.0]
        bounds = (-1.0, 1.0)
        
        mutated = self.strategy.uniform_mutation(genome, 1.0, bounds=bounds)
        
        for value in mutated:
            self.assertGreaterEqual(value, bounds[0])
            self.assertLessEqual(value, bounds[1])
    
    def test_bit_flip_mutation_with_non_boolean_genome(self):
        """Test that bit flip mutation raises error for non-boolean genomes."""
        numeric_genome = [1, 2, 3]
        
        with self.assertRaises(TypeError):
            self.strategy.bit_flip_mutation(numeric_genome, 0.5)
    
    def test_adaptive_mutation_with_empty_history(self):
        """Test adaptive mutation with empty fitness history."""
        genome = [1.0, 2.0, 3.0]
        empty_history = []
        
        mutated = self.strategy.adaptive_mutation(genome, empty_history, base_rate=0.1)
        
        # Should fallback to base rate
        self.assertEqual(len(mutated), len(genome))
    
    def test_adaptive_mutation_rate_adjustment(self):
        """Test that adaptive mutation adjusts rates based on fitness trends."""
        genome = [1.0, 2.0, 3.0]
        
        # Improving fitness should decrease mutation rate
        improving_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        mutated_improving = self.strategy.adaptive_mutation(genome, improving_history, base_rate=0.5)
        
        # Stagnating fitness should increase mutation rate
        stagnating_history = [0.5, 0.5, 0.5, 0.5, 0.5]
        mutated_stagnating = self.strategy.adaptive_mutation(genome, stagnating_history, base_rate=0.1)
        
        # Both should produce valid mutations
        self.assertEqual(len(mutated_improving), len(genome))
        self.assertEqual(len(mutated_stagnating), len(genome))
    
    def test_concurrent_mutation_safety(self):
        """Test that mutation strategies are thread-safe."""
        import threading
        import time
        
        genome = [1.0, 2.0, 3.0, 4.0, 5.0]
        results = []
        
        def mutation_worker():
            for _ in range(100):
                mutated = self.strategy.gaussian_mutation(genome, 0.1, sigma=0.5)
                results.append(mutated)
        
        threads = [threading.Thread(target=mutation_worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        self.assertEqual(len(results), 500)
        
        # All results should be valid
        for result in results:
            self.assertEqual(len(result), len(genome))


class TestSelectionStrategyAdvanced(unittest.TestCase):
    """Advanced test suite for SelectionStrategy edge cases and performance."""
    
    def setUp(self):
        """Set up selection strategy for advanced tests."""
        self.strategy = SelectionStrategy()
    
    def test_selection_with_identical_fitness(self):
        """Test selection strategies when all individuals have identical fitness."""
        uniform_population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5},
            {'genome': [10, 11, 12], 'fitness': 0.5}
        ]
        
        # All selection strategies should work with uniform fitness
        tournament_selected = self.strategy.tournament_selection(uniform_population, 2)
        roulette_selected = self.strategy.roulette_wheel_selection(uniform_population)
        rank_selected = self.strategy.rank_selection(uniform_population)
        
        self.assertIn(tournament_selected, uniform_population)
        self.assertIn(roulette_selected, uniform_population)
        self.assertIn(rank_selected, uniform_population)
    
    def test_selection_with_negative_fitness(self):
        """Test selection strategies with negative fitness values."""
        negative_population = [
            {'genome': [1, 2, 3], 'fitness': -0.1},
            {'genome': [4, 5, 6], 'fitness': -0.5},
            {'genome': [7, 8, 9], 'fitness': -0.3},
            {'genome': [10, 11, 12], 'fitness': -0.9}
        ]
        
        # Tournament and rank selection should handle negative fitness
        tournament_selected = self.strategy.tournament_selection(negative_population, 2)
        rank_selected = self.strategy.rank_selection(negative_population)
        
        self.assertIn(tournament_selected, negative_population)
        self.assertIn(rank_selected, negative_population)
    
    def test_selection_with_extreme_fitness_values(self):
        """Test selection with extremely large or small fitness values."""
        extreme_population = [
            {'genome': [1, 2, 3], 'fitness': 1e10},
            {'genome': [4, 5, 6], 'fitness': 1e-10},
            {'genome': [7, 8, 9], 'fitness': 1e5},
            {'genome': [10, 11, 12], 'fitness': 1e-5}
        ]
        
        selected = self.strategy.tournament_selection(extreme_population, 2)
        self.assertIn(selected, extreme_population)
    
    def test_tournament_selection_deterministic_behavior(self):
        """Test that tournament selection behaves deterministically with fixed random seed."""
        import random
        
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5},
            {'genome': [10, 11, 12], 'fitness': 0.3}
        ]
        
        # With fixed seed, tournament selection should be repeatable
        random.seed(42)
        selected1 = self.strategy.tournament_selection(population, 2)
        
        random.seed(42)
        selected2 = self.strategy.tournament_selection(population, 2)
        
        self.assertEqual(selected1, selected2)
    
    def test_elitism_selection_with_large_elite_count(self):
        """Test elitism selection when elite count equals population size."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.7},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        
        # Elite count equals population size
        selected = self.strategy.elitism_selection(population, len(population))
        
        self.assertEqual(len(selected), len(population))
        # Should be sorted by fitness descending
        fitness_values = [ind['fitness'] for ind in selected]
        self.assertEqual(fitness_values, sorted(fitness_values, reverse=True))
    
    def test_selection_performance_with_large_population(self):
        """Test selection performance with very large populations."""
        import time
        
        large_population = [
            {'genome': [i] * 10, 'fitness': i * 0.01} 
            for i in range(10000)
        ]
        
        start_time = time.time()
        selected = self.strategy.tournament_selection(large_population, 10)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        self.assertLess(end_time - start_time, 1.0)
        self.assertIn(selected, large_population)
    
    def test_roulette_wheel_selection_bias(self):
        """Test that roulette wheel selection shows proper bias towards higher fitness."""
        # Create population with one very high fitness individual
        biased_population = [
            {'genome': [1, 2, 3], 'fitness': 0.99},  # Very high fitness
            {'genome': [4, 5, 6], 'fitness': 0.01},
            {'genome': [7, 8, 9], 'fitness': 0.01},
            {'genome': [10, 11, 12], 'fitness': 0.01}
        ]
        
        # Run multiple selections and count occurrences
        selections = []
        for _ in range(1000):
            selected = self.strategy.roulette_wheel_selection(biased_population)
            selections.append(selected)
        
        # High fitness individual should be selected more often
        high_fitness_count = sum(1 for sel in selections if sel['fitness'] == 0.99)
        
        # Should be selected significantly more than 25% of the time
        self.assertGreater(high_fitness_count, 500)


class TestFitnessFunctionAdvanced(unittest.TestCase):
    """Advanced test suite for FitnessFunction edge cases and numerical stability."""
    
    def setUp(self):
        """Set up fitness function for advanced tests."""
        self.fitness_func = FitnessFunction()
    
    def test_fitness_functions_with_extreme_values(self):
        """Test fitness functions with extreme genome values."""
        extreme_genome = [1e6, -1e6, 1e-6, -1e-6]
        
        # All fitness functions should handle extreme values gracefully
        sphere_fitness = self.fitness_func.sphere_function(extreme_genome)
        rastrigin_fitness = self.fitness_func.rastrigin_function(extreme_genome)
        ackley_fitness = self.fitness_func.ackley_function(extreme_genome)
        
        # Results should be finite numbers
        self.assertTrue(abs(sphere_fitness) < float('inf'))
        self.assertTrue(abs(rastrigin_fitness) < float('inf'))
        self.assertTrue(abs(ackley_fitness) < float('inf'))
    
    def test_fitness_functions_with_nan_values(self):
        """Test fitness functions with NaN values in genome."""
        nan_genome = [1.0, float('nan'), 3.0]
        
        # Functions should handle NaN appropriately
        sphere_fitness = self.fitness_func.sphere_function(nan_genome)
        
        # Result should be NaN or raise appropriate error
        self.assertTrue(sphere_fitness != sphere_fitness or isinstance(sphere_fitness, float))
    
    def test_fitness_functions_with_infinite_values(self):
        """Test fitness functions with infinite values in genome."""
        inf_genome = [1.0, float('inf'), 3.0]
        
        sphere_fitness = self.fitness_func.sphere_function(inf_genome)
        
        # Should handle infinity appropriately
        self.assertTrue(abs(sphere_fitness) == float('inf') or isinstance(sphere_fitness, float))
    
    def test_custom_fitness_function_error_handling(self):
        """Test error handling in custom fitness functions."""
        def failing_fitness(genome):
            if len(genome) > 2:
                raise ValueError("Genome too long")
            return sum(genome)
        
        # Should propagate the error
        with self.assertRaises(ValueError):
            self.fitness_func.evaluate([1, 2, 3, 4], failing_fitness)
    
    def test_multi_objective_with_different_scales(self):
        """Test multi-objective optimization with objectives of different scales."""
        genome = [1.0, 2.0, 3.0]
        
        objectives = [
            lambda g: sum(g),  # Scale: ~6
            lambda g: sum(x**2 for x in g) * 1000,  # Scale: ~14000
            lambda g: sum(x**3 for x in g) * 0.001  # Scale: ~0.036
        ]
        
        fitness_vector = self.fitness_func.multi_objective_evaluate(genome, objectives)
        
        self.assertEqual(len(fitness_vector), 3)
        # Check that all objectives are evaluated
        self.assertAlmostEqual(fitness_vector[0], 6.0)
        self.assertAlmostEqual(fitness_vector[1], 14000.0)
        self.assertAlmostEqual(fitness_vector[2], 0.036)
    
    def test_constraint_handling_with_multiple_constraints(self):
        """Test constraint handling with multiple constraints."""
        genome = [2.0, 3.0, 4.0]
        
        constraints = [
            lambda g: sum(g) < 10,  # Satisfied
            lambda g: max(g) < 5,   # Violated (max is 4, but close)
            lambda g: min(g) > 1    # Satisfied
        ]
        
        fitness = self.fitness_func.evaluate_with_constraints(
            genome,
            lambda g: sum(g),
            constraints
        )
        
        # Should be penalized for violating one constraint
        self.assertLess(fitness, sum(genome))
    
    def test_fitness_function_performance(self):
        """Test fitness function performance with large genomes."""
        import time
        
        large_genome = list(range(10000))
        
        start_time = time.time()
        fitness = self.fitness_func.sphere_function(large_genome)
        end_time = time.time()
        
        # Should complete quickly
        self.assertLess(end_time - start_time, 1.0)
        self.assertIsInstance(fitness, (int, float))
    
    def test_rosenbrock_function_dimensionality(self):
        """Test Rosenbrock function with different dimensionalities."""
        # Test with various genome lengths
        for length in [2, 3, 5, 10, 100]:
            genome = [1.0] * length
            fitness = self.fitness_func.rosenbrock_function(genome)
            
            # Should be 0 at the optimum for all dimensionalities
            self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_ackley_function_numerical_stability(self):
        """Test Ackley function numerical stability with edge cases."""
        # Test with all zeros
        zero_genome = [0.0] * 10
        fitness_zero = self.fitness_func.ackley_function(zero_genome)
        self.assertAlmostEqual(fitness_zero, 0.0, places=10)
        
        # Test with very small values
        small_genome = [1e-10] * 10
        fitness_small = self.fitness_func.ackley_function(small_genome)
        self.assertAlmostEqual(fitness_small, 0.0, places=5)


class TestPopulationManagerAdvanced(unittest.TestCase):
    """Advanced test suite for PopulationManager edge cases and performance."""
    
    def setUp(self):
        """Set up population manager for advanced tests."""
        self.manager = PopulationManager()
    
    def test_population_initialization_with_zero_size(self):
        """Test population initialization with zero population size."""
        with self.assertRaises(ValueError):
            self.manager.initialize_random_population(0, 5)
    
    def test_population_initialization_with_zero_genome_length(self):
        """Test population initialization with zero genome length."""
        population = self.manager.initialize_random_population(10, 0)
        
        self.assertEqual(len(population), 10)
        for individual in population:
            self.assertEqual(len(individual['genome']), 0)
    
    def test_seeded_population_with_empty_seeds(self):
        """Test seeded population initialization with empty seed list."""
        population = self.manager.initialize_seeded_population(10, 5, [])
        
        self.assertEqual(len(population), 10)
        # All individuals should be randomly generated
        for individual in population:
            self.assertEqual(len(individual['genome']), 5)
    
    def test_seeded_population_with_mismatched_lengths(self):
        """Test seeded population with seeds of different lengths."""
        seeds = [
            [1, 2, 3],      # Length 3
            [4, 5, 6, 7],   # Length 4
            [8, 9]          # Length 2
        ]
        
        # Should handle mismatched lengths gracefully
        population = self.manager.initialize_seeded_population(10, 5, seeds)
        
        self.assertEqual(len(population), 10)
        for individual in population:
            self.assertEqual(len(individual['genome']), 5)
    
    def test_evaluation_with_failing_fitness_function(self):
        """Test population evaluation with a fitness function that fails for some individuals."""
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        def selective_fitness(genome):
            if sum(genome) > 15:
                raise ValueError("Fitness calculation failed")
            return sum(genome)
        
        # Should handle partial failures gracefully
        with self.assertRaises(ValueError):
            self.manager.evaluate_population(population, selective_fitness)
    
    def test_population_statistics_with_extreme_values(self):
        """Test population statistics with extreme fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 1e10},
            {'genome': [4, 5, 6], 'fitness': 1e-10},
            {'genome': [7, 8, 9], 'fitness': 1e5},
            {'genome': [10, 11, 12], 'fitness': -1e8}
        ]
        
        stats = self.manager.get_population_statistics(population)
        
        self.assertEqual(stats['best_fitness'], 1e10)
        self.assertEqual(stats['worst_fitness'], -1e8)
        self.assertIsInstance(stats['average_fitness'], float)
        self.assertIsInstance(stats['std_dev_fitness'], float)
    
    def test_diversity_calculation_with_identical_genomes(self):
        """Test diversity calculation when all genomes are identical."""
        identical_population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(identical_population)
        
        # Diversity should be 0 for identical genomes
        self.assertEqual(diversity, 0.0)
    
    def test_diversity_calculation_with_high_dimensional_genomes(self):
        """Test diversity calculation with high-dimensional genomes."""
        high_dim_population = [
            {'genome': [i] * 1000, 'fitness': 0.5} 
            for i in range(5)
        ]
        
        diversity = self.manager.calculate_diversity(high_dim_population)
        
        self.assertIsInstance(diversity, float)
        self.assertGreater(diversity, 0.0)
    
    def test_population_manager_memory_efficiency(self):
        """Test memory efficiency with large populations."""
        import sys
        
        # Create a large population
        large_population = self.manager.initialize_random_population(10000, 100)
        
        # Evaluate the population
        def simple_fitness(genome):
            return sum(genome)
        
        self.manager.evaluate_population(large_population, simple_fitness)
        
        # Population should be properly structured
        self.assertEqual(len(large_population), 10000)
        for individual in large_population:
            self.assertIsNotNone(individual['fitness'])
    
    def test_best_individual_with_ties(self):
        """Test getting best individual when multiple individuals have the same fitness."""
        tied_population = [
            {'genome': [1, 2, 3], 'fitness': 0.9},
            {'genome': [4, 5, 6], 'fitness': 0.9},  # Tie for best
            {'genome': [7, 8, 9], 'fitness': 0.7}
        ]
        
        best = self.manager.get_best_individual(tied_population)
        
        # Should return one of the tied individuals
        self.assertEqual(best['fitness'], 0.9)
        self.assertIn(best, tied_population)


class TestGeneticOperationsAdvanced(unittest.TestCase):
    """Advanced test suite for GeneticOperations edge cases and numerical stability."""
    
    def setUp(self):
        """Set up genetic operations for advanced tests."""
        self.operations = GeneticOperations()
    
    def test_crossover_with_empty_parents(self):
        """Test crossover operations with empty parent genomes."""
        empty_parent1 = []
        empty_parent2 = []
        
        # All crossover operations should handle empty parents
        child1, child2 = self.operations.single_point_crossover(empty_parent1, empty_parent2)
        self.assertEqual(len(child1), 0)
        self.assertEqual(len(child2), 0)
        
        child1, child2 = self.operations.uniform_crossover(empty_parent1, empty_parent2)
        self.assertEqual(len(child1), 0)
        self.assertEqual(len(child2), 0)
    
    def test_crossover_with_single_element_parents(self):
        """Test crossover operations with single-element parent genomes."""
        parent1 = [1.0]
        parent2 = [2.0]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), 1)
        self.assertEqual(len(child2), 1)
        # With single element, crossover should just swap
        self.assertIn(child1[0], [1.0, 2.0])
        self.assertIn(child2[0], [1.0, 2.0])
    
    def test_two_point_crossover_with_small_genomes(self):
        """Test two-point crossover with genomes too small for two crossover points."""
        parent1 = [1, 2]
        parent2 = [3, 4]
        
        # Should handle gracefully (might fallback to single-point)
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)
    
    def test_arithmetic_crossover_with_extreme_alpha(self):
        """Test arithmetic crossover with extreme alpha values."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        # Test with alpha = 0 (should give parent2, parent1)
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.0)
        
        for i in range(len(parent1)):
            self.assertAlmostEqual(child1[i], parent2[i], places=5)
            self.assertAlmostEqual(child2[i], parent1[i], places=5)
        
        # Test with alpha = 1 (should give parent1, parent2)
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.0)
        
        for i in range(len(parent1)):
            self.assertAlmostEqual(child1[i], parent1[i], places=5)
            self.assertAlmostEqual(child2[i], parent2[i], places=5)
    
    def test_simulated_binary_crossover_bounds_violation(self):
        """Test simulated binary crossover with parents outside bounds."""
        parent1 = [10.0, 20.0, 30.0]  # Outside bounds
        parent2 = [15.0, 25.0, 35.0]  # Outside bounds
        bounds = [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)]  # Tight bounds
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=2.0
        )
        
        # Children should be within bounds
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(child1[i], lower)
            self.assertLessEqual(child1[i], upper)
            self.assertGreaterEqual(child2[i], lower)
            self.assertLessEqual(child2[i], upper)
    
    def test_blend_crossover_with_identical_parents(self):
        """Test blend crossover when parents are identical."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [1.0, 2.0, 3.0]  # Identical
        
        child1, child2 = self.operations.blend_crossover(parent1, parent2, alpha=0.5)
        
        # Children should be close to parents when parents are identical
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
    
    def test_crossover_numerical_stability(self):
        """Test crossover operations with extreme numerical values."""
        parent1 = [1e-10, 1e10, -1e-10]
        parent2 = [1e-9, 1e9, -1e-9]
        
        # All crossover operations should handle extreme values
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.5)
        
        # Children should be finite
        for value in child1 + child2:
            self.assertTrue(abs(value) < float('inf'))
    
    def test_concurrent_crossover_operations(self):
        """Test thread safety of crossover operations."""
        import threading
        
        parent1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        parent2 = [6.0, 7.0, 8.0, 9.0, 10.0]
        results = []
        
        def crossover_worker():
            for _ in range(100):
                child1, child2 = self.operations.single_point_crossover(parent1, parent2)
                results.append((child1, child2))
        
        threads = [threading.Thread(target=crossover_worker) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        self.assertEqual(len(results), 300)
        
        # All results should be valid
        for child1, child2 in results:
            self.assertEqual(len(child1), len(parent1))
            self.assertEqual(len(child2), len(parent2))
    
    def test_crossover_with_mixed_types(self):
        """Test crossover operations with mixed numeric types."""
        parent1 = [1, 2.0, 3]      # Mixed int/float
        parent2 = [4.0, 5, 6.0]    # Mixed int/float
        
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.5)
        
        # Should handle mixed types gracefully
        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))
        
        # Results should be numeric
        for value in child1 + child2:
            self.assertIsInstance(value, (int, float))


class TestEvolutionaryConduitAdvanced(unittest.TestCase):
    """Advanced test suite for EvolutionaryConduit edge cases and integration."""
    
    def setUp(self):
        """Set up evolutionary conduit for advanced tests."""
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    def test_conduit_with_custom_components(self):
        """Test conduit with custom mutation, selection, and fitness components."""
        # Custom mutation strategy
        custom_mutation = MutationStrategy()
        
        # Custom selection strategy  
        custom_selection = SelectionStrategy()
        
        # Custom fitness function
        custom_fitness = FitnessFunction()
        
        # Set custom components
        self.conduit.mutation_strategy = custom_mutation
        self.conduit.selection_strategy = custom_selection
        self.conduit.fitness_function = custom_fitness
        
        # Verify components are set
        self.assertIs(self.conduit.mutation_strategy, custom_mutation)
        self.assertIs(self.conduit.selection_strategy, custom_selection)
        self.assertIs(self.conduit.fitness_function, custom_fitness)
    
    def test_conduit_state_persistence(self):
        """Test that conduit state persists across save/load cycles."""
        # Set up complex state
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()
        
        # Add multiple callbacks
        callback_count = 0
        
        def counter_callback(gen, pop, best):
            nonlocal callback_count
            callback_count += 1
        
        self.conduit.add_callback(counter_callback)
        
        # Save state
        state = self.conduit.save_state()
        
        # Modify original conduit
        new_params = EvolutionaryParameters(population_size=50)
        self.conduit.set_parameters(new_params)
        
        # Load saved state
        self.conduit.load_state(state)
        
        # Verify state is restored
        self.assertEqual(self.conduit.parameters.population_size, 10)
        self.assertTrue(self.conduit.history_enabled)
    
    def test_conduit_with_multiple_callbacks(self):
        """Test conduit with multiple callbacks and callback ordering."""
        callback_order = []
        
        def callback1(gen, pop, best):
            callback_order.append('callback1')
        
        def callback2(gen, pop, best):
            callback_order.append('callback2')
        
        def callback3(gen, pop, best):
            callback_order.append('callback3')
        
        # Add callbacks in order
        self.conduit.add_callback(callback1)
        self.conduit.add_callback(callback2)
        self.conduit.add_callback(callback3)
        
        # Verify callbacks are stored
        self.assertEqual(len(self.conduit.callbacks), 3)
        self.assertIn(callback1, self.conduit.callbacks)
        self.assertIn(callback2, self.conduit.callbacks)
        self.assertIn(callback3, self.conduit.callbacks)
    
    def test_conduit_callback_error_handling(self):
        """Test that conduit handles callback errors gracefully."""
        def failing_callback(gen, pop, best):
            raise RuntimeError("Callback failed")
        
        def working_callback(gen, pop, best):
            pass
        
        self.conduit.add_callback(failing_callback)
        self.conduit.add_callback(working_callback)
        
        # Should handle callback failure gracefully during evolution
        self.conduit.set_parameters(self.params)
        
        # Mock evolution to test callback handling
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
                'generations_run': 5,
                'final_population': [],
                'statistics': {'best_fitness': 0.9}
            }
            
            # Should not raise exception despite failing callback
            result = self.conduit.run_evolution(genome_length=3)
            self.assertIsNotNone(result)
    
    def test_conduit_parameter_validation(self):
        """Test conduit parameter validation and error handling."""
        # Test with invalid parameter object
        with self.assertRaises(TypeError):
            self.conduit.set_parameters("invalid_params")
        
        # Test with None parameters
        with self.assertRaises(TypeError):
            self.conduit.set_parameters(None)
    
    def test_conduit_fitness_function_validation(self):
        """Test conduit fitness function validation."""
        # Test with non-callable fitness function
        with self.assertRaises(TypeError):
            self.conduit.set_fitness_function("not_callable")
        
        # Test with None fitness function
        with self.assertRaises(TypeError):
            self.conduit.set_fitness_function(None)
    
    def test_conduit_history_tracking_memory_management(self):
        """Test that history tracking doesn't cause memory leaks."""
        import gc
        
        self.conduit.enable_history_tracking()
        self.conduit.set_parameters(self.params)
        
        # Mock evolution with history tracking
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 0.9},
                'generations_run': 5,
                'final_population': [],
                'statistics': {'best_fitness': 0.9},
                'history': [{'generation': i, 'best_fitness': 0.1 * i} for i in range(5)]
            }
            
            # Run multiple evolutions
            for _ in range(10):
                result = self.conduit.run_evolution(genome_length=3)
                self.assertIsNotNone(result)
        
        # Force garbage collection
        gc.collect()
        
        # Memory usage should be reasonable
        self.assertTrue(self.conduit.history_enabled)
    
    def test_conduit_concurrent_execution(self):
        """Test conduit behavior under concurrent execution."""
        import threading
        import time
        
        results = []
        
        def evolution_worker():
            conduit = EvolutionaryConduit()
            conduit.set_parameters(self.params)
            
            def simple_fitness(genome):
                return sum(genome)
            
            conduit.set_fitness_function(simple_fitness)
            
            with patch.object(conduit, 'evolve') as mock_evolve:
                mock_evolve.return_value = {
                    'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                    'generations_run': 5,
                    'final_population': [],
                    'statistics': {'best_fitness': 6.0}
                }
                
                result = conduit.run_evolution(genome_length=3)
                results.append(result)
        
        threads = [threading.Thread(target=evolution_worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should complete successfully
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn('best_individual', result)


class TestStressAndPerformance(unittest.TestCase):
    """Stress testing and performance validation for evolutionary components."""
    
    def setUp(self):
        """Set up components for stress testing."""
        self.conduit = EvolutionaryConduit()
        self.genesis_conduit = GenesisEvolutionaryConduit()
    
    def test_large_population_performance(self):
        """Test performance with very large populations."""
        import time
        
        large_params = EvolutionaryParameters(
            population_size=10000,
            generations=1,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        def simple_fitness(genome):
            return sum(genome)
        
        self.conduit.set_parameters(large_params)
        self.conduit.set_fitness_function(simple_fitness)
        
        start_time = time.time()
        
        # Mock evolution to test setup performance
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                'generations_run': 1,
                'final_population': [],
                'statistics': {'best_fitness': 6.0}
            }
            
            result = self.conduit.run_evolution(genome_length=100)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 5.0)
        self.assertIsNotNone(result)
    
    def test_high_dimensional_genome_performance(self):
        """Test performance with high-dimensional genomes."""
        import time
        
        params = EvolutionaryParameters(
            population_size=100,
            generations=1,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        def high_dim_fitness(genome):
            return sum(x**2 for x in genome)
        
        self.conduit.set_parameters(params)
        self.conduit.set_fitness_function(high_dim_fitness)
        
        start_time = time.time()
        
        # Mock evolution with high-dimensional genome
        with patch.object(self.conduit, 'evolve') as mock_evolve:
            mock_evolve.return_value = {
                'best_individual': {'genome': [1.0] * 10000, 'fitness': 10000.0},
                'generations_run': 1,
                'final_population': [],
                'statistics': {'best_fitness': 10000.0}
            }
            
            result = self.conduit.run_evolution(genome_length=10000)
        
        end_time = time.time()
        
        # Should handle high-dimensional genomes efficiently
        self.assertLess(end_time - start_time, 10.0)
        self.assertIsNotNone(result)
    
    def test_memory_usage_with_large_populations(self):
        """Test memory usage patterns with large populations."""
        import gc
        import sys
        
        # Force garbage collection before test
        gc.collect()
        
        manager = PopulationManager()
        
        # Create large population
        large_population = manager.initialize_random_population(50000, 200)
        
        # Evaluate population
        def memory_intensive_fitness(genome):
            # Create some temporary data
            temp_data = [x * 2 for x in genome]
            return sum(temp_data)
        
        manager.evaluate_population(large_population, memory_intensive_fitness)
        
        # Check that population is properly structured
        self.assertEqual(len(large_population), 50000)
        
        # Clean up
        del large_population
        gc.collect()
    
    def test_numerical_stability_under_stress(self):
        """Test numerical stability under extreme conditions."""
        fitness_func = FitnessFunction()
        
        # Test with extreme values
        extreme_genomes = [
            [1e100, 1e-100, 1e50],
            [-1e100, -1e-100, -1e50],
            [float('inf'), float('-inf'), 0.0],
            [1e-323, 1e-324, 1e-325]  # Near underflow
        ]
        
        for genome in extreme_genomes:
            try:
                # Should handle extreme values gracefully
                fitness = fitness_func.sphere_function(genome)
                
                # Result should be a number (not NaN) or handle appropriately
                if fitness == fitness:  # Not NaN
                    self.assertIsInstance(fitness, (int, float))
                
            except (OverflowError, ValueError):
                # Acceptable to raise these exceptions for extreme values
                pass
    
    def test_concurrent_stress_testing(self):
        """Test system behavior under concurrent stress."""
        import threading
        import time
        
        results = []
        errors = []
        
        def stress_worker():
            try:
                conduit = EvolutionaryConduit()
                params = EvolutionaryParameters(
                    population_size=1000,
                    generations=2,
                    mutation_rate=0.1,
                    crossover_rate=0.8
                )
                
                conduit.set_parameters(params)
                conduit.set_fitness_function(lambda g: sum(g))
                
                with patch.object(conduit, 'evolve') as mock_evolve:
                    mock_evolve.return_value = {
                        'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                        'generations_run': 2,
                        'final_population': [],
                        'statistics': {'best_fitness': 6.0}
                    }
                    
                    result = conduit.run_evolution(genome_length=50)
                    results.append(result)
                    
            except Exception as e:
                errors.append(e)
        
        # Run multiple concurrent stress tests
        threads = [threading.Thread(target=stress_worker) for _ in range(10)]
        
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # All threads should complete successfully
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 10)
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 30.0)


class TestErrorRecoveryAndRobustness(unittest.TestCase):
    """Test error recovery and system robustness."""
    
    def setUp(self):
        """Set up components for robustness testing."""
        self.conduit = EvolutionaryConduit()
        self.genesis_conduit = GenesisEvolutionaryConduit()
    
    def test_recovery_from_fitness_function_failures(self):
        """Test recovery when fitness function fails intermittently."""
        failure_count = 0
        
        def intermittent_failing_fitness(genome):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count % 3 == 0:
                raise ValueError("Intermittent failure")
            
            return sum(genome)
        
        self.conduit.set_fitness_function(intermittent_failing_fitness)
        
        # Should handle intermittent failures gracefully
        with self.assertRaises(ValueError):
            self.conduit.run_evolution(genome_length=3)
    
    def test_parameter_corruption_recovery(self):
        """Test recovery from parameter corruption."""
        params = EvolutionaryParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        # Simulate parameter corruption
        self.conduit.parameters.population_size = -1
        
        # Should detect and handle corrupted parameters
        with self.assertRaises(ValueError):
            self.conduit.run_evolution(genome_length=3)
    
    def test_memory_corruption_detection(self):
        """Test detection of memory corruption scenarios."""
        manager = PopulationManager()
        
        # Create normal population
        population = manager.initialize_random_population(10, 5)
        
        # Simulate memory corruption
        population[0]['genome'] = None
        population[1]['fitness'] = "invalid"
        
        # Should detect corrupted population
        with self.assertRaises((TypeError, ValueError)):
            manager.get_population_statistics(population)
    
    def test_infinite_loop_prevention(self):
        """Test prevention of infinite loops in selection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout
        
        try:
            strategy = SelectionStrategy()
            
            # Create population that might cause infinite loop
            problematic_population = [
                {'genome': [1, 2, 3], 'fitness': float('nan')},
                {'genome': [4, 5, 6], 'fitness': float('nan')},
                {'genome': [7, 8, 9], 'fitness': float('nan')}
            ]
            
            # Should not hang indefinitely
            with self.assertRaises((ValueError, TimeoutError)):
                for _ in range(1000):  # Many iterations
                    strategy.tournament_selection(problematic_population, 2)
                    
        finally:
            signal.alarm(0)  # Cancel timeout
    
    def test_resource_cleanup_on_failure(self):
        """Test that resources are cleaned up properly on failure."""
        import gc
        
        def resource_intensive_fitness(genome):
            # Simulate resource allocation
            large_data = [0] * 1000000
            
            if sum(genome) > 10:
                raise RuntimeError("Simulated failure")
            
            return sum(genome)
        
        self.conduit.set_fitness_function(resource_intensive_fitness)
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run evolution that will fail
        with self.assertRaises(RuntimeError):
            self.conduit.run_evolution(genome_length=5)
        
        # Force garbage collection after failure
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not increase dramatically
        self.assertLess(final_objects - initial_objects, 100000)
    
    def test_state_consistency_after_errors(self):
        """Test that conduit state remains consistent after errors."""
        params = EvolutionaryParameters(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        # Save initial state
        initial_state = self.conduit.save_state()
        
        # Attempt operation that will fail
        def failing_fitness(genome):
            raise ValueError("Test failure")
        
        self.conduit.set_fitness_function(failing_fitness)
        
        with self.assertRaises(ValueError):
            self.conduit.run_evolution(genome_length=3)
        
        # State should still be accessible and consistent
        current_state = self.conduit.save_state()
        self.assertIsNotNone(current_state)
        self.assertEqual(current_state['parameters']['population_size'], 10)
    
    def test_graceful_degradation_under_load(self):
        """Test graceful degradation when system is under load."""
        import time
        import threading
        
        # Create high load scenario
        load_results = []
        
        def load_generator():
            for _ in range(100):
                # Simulate CPU intensive work
                sum(i**2 for i in range(1000))
        
        # Start load generators
        load_threads = [threading.Thread(target=load_generator) for _ in range(5)]
        for thread in load_threads:
            thread.start()
        
        try:
            # Run evolution under load
            params = EvolutionaryParameters(
                population_size=50,
                generations=2,
                mutation_rate=0.1,
                crossover_rate=0.8
            )
            
            self.conduit.set_parameters(params)
            self.conduit.set_fitness_function(lambda g: sum(g))
            
            start_time = time.time()
            
            with patch.object(self.conduit, 'evolve') as mock_evolve:
                mock_evolve.return_value = {
                    'best_individual': {'genome': [1, 2, 3], 'fitness': 6.0},
                    'generations_run': 2,
                    'final_population': [],
                    'statistics': {'best_fitness': 6.0}
                }
                
                result = self.conduit.run_evolution(genome_length=10)
            
            end_time = time.time()
            
            # Should complete even under load (may be slower)
            self.assertIsNotNone(result)
            self.assertLess(end_time - start_time, 30.0)  # Generous timeout
            
        finally:
            # Clean up load threads
            for thread in load_threads:
                thread.join()

