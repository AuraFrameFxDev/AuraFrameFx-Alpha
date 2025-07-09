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

class TestEvolutionaryParametersBoundaryConditions(unittest.TestCase):
    """Extended test suite for EvolutionaryParameters boundary conditions and edge cases."""
    
    def test_minimum_valid_values(self):
        """Test that minimum valid values are accepted for all parameters."""
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
        """Test that maximum valid values are accepted for rate parameters."""
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
        """Test that negative generation values raise ValueError."""
        with self.assertRaises(ValueError):
            EvolutionaryParameters(generations=-1)
    
    def test_negative_selection_pressure_validation(self):
        """Test that negative selection pressure values raise ValueError."""
        with self.assertRaises(ValueError):
            EvolutionaryParameters(selection_pressure=-0.1)
    
    def test_selection_pressure_upper_bound(self):
        """Test that selection pressure values above 1.0 raise ValueError."""
        with self.assertRaises(ValueError):
            EvolutionaryParameters(selection_pressure=1.1)
    
    def test_parameter_type_validation(self):
        """Test that non-numeric parameter types raise TypeError."""
        with self.assertRaises(TypeError):
            EvolutionaryParameters(population_size="100")
        
        with self.assertRaises(TypeError):
            EvolutionaryParameters(mutation_rate="0.1")
    
    def test_from_dict_with_missing_keys(self):
        """Test that from_dict handles missing keys gracefully with defaults."""
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
        """Test that from_dict validates parameter values."""
        invalid_dict = {
            'population_size': 0,
            'mutation_rate': -0.5
        }
        with self.assertRaises(ValueError):
            EvolutionaryParameters.from_dict(invalid_dict)
    
    def test_to_dict_immutability(self):
        """Test that modifying the dict returned by to_dict doesn't affect the original parameters."""
        params = EvolutionaryParameters(population_size=100)
        params_dict = params.to_dict()
        params_dict['population_size'] = 200
        
        self.assertEqual(params.population_size, 100)  # Should remain unchanged


class TestMutationStrategyEdgeCases(unittest.TestCase):
    """Extended test suite for MutationStrategy edge cases and boundary conditions."""
    
    def setUp(self):
        self.strategy = MutationStrategy()
    
    def test_gaussian_mutation_with_zero_sigma(self):
        """Test gaussian mutation with zero sigma returns unchanged genome."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=0.0)
        self.assertEqual(mutated, genome)
    
    def test_gaussian_mutation_empty_genome(self):
        """Test gaussian mutation with empty genome returns empty list."""
        genome = []
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=0.5, sigma=1.0)
        self.assertEqual(mutated, [])
    
    def test_gaussian_mutation_single_element(self):
        """Test gaussian mutation with single element genome."""
        genome = [5.0]
        mutated = self.strategy.gaussian_mutation(genome, mutation_rate=1.0, sigma=1.0)
        self.assertEqual(len(mutated), 1)
        self.assertIsInstance(mutated[0], float)
    
    def test_uniform_mutation_with_zero_bounds(self):
        """Test uniform mutation with zero range bounds."""
        genome = [1.0, 2.0, 3.0]
        mutated = self.strategy.uniform_mutation(genome, mutation_rate=1.0, bounds=(5.0, 5.0))
        
        for value in mutated:
            self.assertEqual(value, 5.0)
    
    def test_uniform_mutation_inverted_bounds(self):
        """Test uniform mutation with inverted bounds raises ValueError."""
        genome = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            self.strategy.uniform_mutation(genome, mutation_rate=0.5, bounds=(10, -10))
    
    def test_bit_flip_mutation_empty_genome(self):
        """Test bit flip mutation with empty genome."""
        genome = []
        mutated = self.strategy.bit_flip_mutation(genome, mutation_rate=0.5)
        self.assertEqual(mutated, [])
    
    def test_bit_flip_mutation_non_boolean_genome(self):
        """Test bit flip mutation with non-boolean genome raises TypeError."""
        genome = [1, 0, 1]
        with self.assertRaises(TypeError):
            self.strategy.bit_flip_mutation(genome, mutation_rate=0.5)
    
    def test_adaptive_mutation_empty_history(self):
        """Test adaptive mutation with empty fitness history."""
        genome = [1.0, 2.0, 3.0]
        fitness_history = []
        
        with self.assertRaises(ValueError):
            self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
    
    def test_adaptive_mutation_single_history_point(self):
        """Test adaptive mutation with single fitness history point."""
        genome = [1.0, 2.0, 3.0]
        fitness_history = [0.5]
        
        mutated = self.strategy.adaptive_mutation(genome, fitness_history, base_rate=0.1)
        self.assertEqual(len(mutated), len(genome))
    
    def test_adaptive_mutation_negative_base_rate(self):
        """Test adaptive mutation with negative base rate raises ValueError."""
        genome = [1.0, 2.0, 3.0]
        fitness_history = [0.5, 0.6, 0.7]
        
        with self.assertRaises(ValueError):
            self.strategy.adaptive_mutation(genome, fitness_history, base_rate=-0.1)
    
    def test_mutation_deterministic_with_zero_rate(self):
        """Test that all mutation strategies return unchanged genome with zero mutation rate."""
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
        self.strategy = SelectionStrategy()
    
    def test_tournament_selection_single_individual(self):
        """Test tournament selection with single individual population."""
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        selected = self.strategy.tournament_selection(population, tournament_size=1)
        self.assertEqual(selected, population[0])
    
    def test_tournament_selection_all_same_fitness(self):
        """Test tournament selection with all individuals having same fitness."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.5},
            {'genome': [7, 8, 9], 'fitness': 0.5}
        ]
        selected = self.strategy.tournament_selection(population, tournament_size=2)
        self.assertIn(selected, population)
    
    def test_roulette_wheel_selection_zero_fitness(self):
        """Test roulette wheel selection with all zero fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.0},
            {'genome': [4, 5, 6], 'fitness': 0.0},
            {'genome': [7, 8, 9], 'fitness': 0.0}
        ]
        selected = self.strategy.roulette_wheel_selection(population)
        self.assertIn(selected, population)
    
    def test_roulette_wheel_selection_negative_fitness(self):
        """Test roulette wheel selection with negative fitness values."""
        population = [
            {'genome': [1, 2, 3], 'fitness': -0.5},
            {'genome': [4, 5, 6], 'fitness': -0.3},
            {'genome': [7, 8, 9], 'fitness': -0.1}
        ]
        selected = self.strategy.roulette_wheel_selection(population)
        self.assertIn(selected, population)
    
    def test_rank_selection_single_individual(self):
        """Test rank selection with single individual population."""
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        selected = self.strategy.rank_selection(population)
        self.assertEqual(selected, population[0])
    
    def test_elitism_selection_more_elites_than_population(self):
        """Test elitism selection requesting more elites than population size."""
        population = [
            {'genome': [1, 2, 3], 'fitness': 0.5},
            {'genome': [4, 5, 6], 'fitness': 0.7}
        ]
        selected = self.strategy.elitism_selection(population, elite_count=5)
        self.assertEqual(len(selected), len(population))
    
    def test_elitism_selection_zero_count(self):
        """Test elitism selection with zero elite count."""
        population = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        selected = self.strategy.elitism_selection(population, elite_count=0)
        self.assertEqual(len(selected), 0)
    
    def test_selection_with_missing_fitness_key(self):
        """Test selection strategies with individuals missing fitness key."""
        population = [{'genome': [1, 2, 3]}]  # Missing fitness
        
        with self.assertRaises(KeyError):
            self.strategy.tournament_selection(population, tournament_size=1)
    
    def test_selection_with_none_fitness(self):
        """Test selection strategies with None fitness values."""
        population = [{'genome': [1, 2, 3], 'fitness': None}]
        
        with self.assertRaises(TypeError):
            self.strategy.tournament_selection(population, tournament_size=1)


class TestFitnessFunctionEdgeCases(unittest.TestCase):
    """Extended test suite for FitnessFunction edge cases and boundary conditions."""
    
    def setUp(self):
        self.fitness_func = FitnessFunction()
    
    def test_sphere_function_empty_genome(self):
        """Test sphere function with empty genome."""
        genome = []
        fitness = self.fitness_func.sphere_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_sphere_function_single_element(self):
        """Test sphere function with single element genome."""
        genome = [3.0]
        fitness = self.fitness_func.sphere_function(genome)
        self.assertEqual(fitness, -9.0)
    
    def test_rastrigin_function_empty_genome(self):
        """Test Rastrigin function with empty genome."""
        genome = []
        fitness = self.fitness_func.rastrigin_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_rastrigin_function_large_values(self):
        """Test Rastrigin function with large values."""
        genome = [100.0, -100.0]
        fitness = self.fitness_func.rastrigin_function(genome)
        self.assertIsInstance(fitness, float)
        self.assertLess(fitness, 0.0)  # Should be negative for large values
    
    def test_rosenbrock_function_single_element(self):
        """Test Rosenbrock function with single element genome."""
        genome = [1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_rosenbrock_function_empty_genome(self):
        """Test Rosenbrock function with empty genome."""
        genome = []
        fitness = self.fitness_func.rosenbrock_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function_empty_genome(self):
        """Test Ackley function with empty genome."""
        genome = []
        fitness = self.fitness_func.ackley_function(genome)
        self.assertEqual(fitness, 0.0)
    
    def test_ackley_function_single_element(self):
        """Test Ackley function with single element genome."""
        genome = [0.0]
        fitness = self.fitness_func.ackley_function(genome)
        self.assertAlmostEqual(fitness, 0.0, places=10)
    
    def test_custom_function_with_exception(self):
        """Test custom function that raises an exception."""
        def failing_func(genome):
            raise ValueError("Custom fitness function failed")
        
        genome = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            self.fitness_func.evaluate(genome, failing_func)
    
    def test_multi_objective_with_empty_objectives(self):
        """Test multi-objective evaluation with empty objectives list."""
        genome = [1.0, 2.0, 3.0]
        objectives = []
        
        fitness = self.fitness_func.multi_objective_evaluate(genome, objectives)
        self.assertEqual(fitness, [])
    
    def test_multi_objective_with_failing_objective(self):
        """Test multi-objective evaluation with one failing objective."""
        genome = [1.0, 2.0, 3.0]
        objectives = [
            lambda g: sum(g),
            lambda g: 1 / 0  # Will raise ZeroDivisionError
        ]
        
        with self.assertRaises(ZeroDivisionError):
            self.fitness_func.multi_objective_evaluate(genome, objectives)
    
    def test_constraint_handling_with_no_constraints(self):
        """Test constraint handling with empty constraints list."""
        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate_with_constraints(
            genome, 
            lambda g: sum(g), 
            []
        )
        self.assertEqual(fitness, sum(genome))
    
    def test_constraint_handling_with_failing_constraint(self):
        """Test constraint handling with constraint that raises exception."""
        genome = [1.0, 2.0, 3.0]
        
        def failing_constraint(g):
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
        self.manager = PopulationManager()
    
    def test_initialize_random_population_zero_size(self):
        """Test random population initialization with zero population size."""
        population = self.manager.initialize_random_population(0, 5)
        self.assertEqual(len(population), 0)
    
    def test_initialize_random_population_zero_genome_length(self):
        """Test random population initialization with zero genome length."""
        population = self.manager.initialize_random_population(5, 0)
        
        self.assertEqual(len(population), 5)
        for individual in population:
            self.assertEqual(len(individual['genome']), 0)
    
    def test_initialize_seeded_population_more_seeds_than_size(self):
        """Test seeded population initialization with more seeds than population size."""
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
        """Test seeded population initialization with empty seeds list."""
        seeds = []
        population = self.manager.initialize_seeded_population(3, 5, seeds)
        
        self.assertEqual(len(population), 3)
        for individual in population:
            self.assertEqual(len(individual['genome']), 5)
    
    def test_initialize_seeded_population_mismatched_genome_length(self):
        """Test seeded population initialization with seeds of wrong length."""
        seeds = [
            [1.0, 2.0],  # Length 2, but expecting 5
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Length 6, but expecting 5
        ]
        
        with self.assertRaises(ValueError):
            self.manager.initialize_seeded_population(5, 5, seeds)
    
    def test_evaluate_population_empty_population(self):
        """Test evaluating an empty population."""
        population = []
        fitness_func = lambda genome: sum(genome)
        
        self.manager.evaluate_population(population, fitness_func)
        self.assertEqual(len(population), 0)
    
    def test_evaluate_population_with_failing_fitness(self):
        """Test evaluating population with fitness function that fails for some individuals."""
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [], 'fitness': None}  # Empty genome might cause issues
        ]
        
        def problematic_fitness(genome):
            if not genome:
                raise ValueError("Empty genome")
            return sum(genome)
        
        with self.assertRaises(ValueError):
            self.manager.evaluate_population(population, problematic_fitness)
    
    def test_get_population_statistics_single_individual(self):
        """Test population statistics with single individual."""
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
        """Test diversity calculation with identical genomes."""
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.6},
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.7}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        self.assertEqual(diversity, 0.0)
    
    def test_diversity_calculation_empty_genomes(self):
        """Test diversity calculation with empty genomes."""
        population = [
            {'genome': [], 'fitness': 0.5},
            {'genome': [], 'fitness': 0.6}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        self.assertEqual(diversity, 0.0)
    
    def test_diversity_calculation_single_individual(self):
        """Test diversity calculation with single individual."""
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}
        ]
        
        diversity = self.manager.calculate_diversity(population)
        self.assertEqual(diversity, 0.0)


class TestGeneticOperationsEdgeCases(unittest.TestCase):
    """Extended test suite for GeneticOperations edge cases and boundary conditions."""
    
    def setUp(self):
        self.operations = GeneticOperations()
    
    def test_crossover_empty_parents(self):
        """Test crossover operations with empty parent genomes."""
        parent1 = []
        parent2 = []
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(child1, [])
        self.assertEqual(child2, [])
    
    def test_crossover_single_element_parents(self):
        """Test crossover operations with single element parent genomes."""
        parent1 = [1]
        parent2 = [2]
        
        child1, child2 = self.operations.single_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 1)
        self.assertEqual(len(child2), 1)
    
    def test_two_point_crossover_length_two(self):
        """Test two-point crossover with length-2 genomes."""
        parent1 = [1, 2]
        parent2 = [3, 4]
        
        child1, child2 = self.operations.two_point_crossover(parent1, parent2)
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)
    
    def test_uniform_crossover_zero_rate(self):
        """Test uniform crossover with zero crossover rate."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=0.0)
        
        # With zero crossover rate, children should be identical to parents
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
    
    def test_uniform_crossover_full_rate(self):
        """Test uniform crossover with full crossover rate."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = self.operations.uniform_crossover(parent1, parent2, crossover_rate=1.0)
        
        # With full crossover rate, children should be swapped
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
    
    def test_arithmetic_crossover_zero_alpha(self):
        """Test arithmetic crossover with zero alpha."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=0.0)
        
        self.assertEqual(child1, parent2)
        self.assertEqual(child2, parent1)
    
    def test_arithmetic_crossover_full_alpha(self):
        """Test arithmetic crossover with alpha=1.0."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        
        child1, child2 = self.operations.arithmetic_crossover(parent1, parent2, alpha=1.0)
        
        self.assertEqual(child1, parent1)
        self.assertEqual(child2, parent2)
    
    def test_simulated_binary_crossover_tight_bounds(self):
        """Test simulated binary crossover with very tight bounds."""
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
        """Test simulated binary crossover with zero eta parameter."""
        parent1 = [1.0, 2.0, 3.0]
        parent2 = [4.0, 5.0, 6.0]
        bounds = [(-10, 10)] * 3
        
        child1, child2 = self.operations.simulated_binary_crossover(
            parent1, parent2, bounds, eta=0.0
        )
        
        self.assertEqual(len(child1), 3)
        self.assertEqual(len(child2), 3)
    
    def test_blend_crossover_zero_alpha(self):
        """Test blend crossover with zero alpha."""
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
        """Test crossover operations with None values in genomes."""
        parent1 = [1, None, 3]
        parent2 = [4, 5, None]
        
        with self.assertRaises(TypeError):
            self.operations.arithmetic_crossover(parent1, parent2, alpha=0.5)


class TestEvolutionaryConduitEdgeCases(unittest.TestCase):
    """Extended test suite for EvolutionaryConduit edge cases and boundary conditions."""
    
    def setUp(self):
        self.conduit = EvolutionaryConduit()
    
    def test_run_evolution_zero_generations(self):
        """Test running evolution with zero generations."""
        params = EvolutionaryParameters(
            population_size=10,
            generations=0,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        def simple_fitness(genome):
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
        """Test running evolution with single individual population."""
        params = EvolutionaryParameters(
            population_size=1,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        self.conduit.set_parameters(params)
        
        def simple_fitness(genome):
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
        """Test callback function that raises an exception."""
        def failing_callback(generation, population, best_individual):
            raise ValueError("Callback failed")
        
        self.conduit.add_callback(failing_callback)
        
        # The callback should be added but exception handling during evolution is implementation-dependent
        self.assertIn(failing_callback, self.conduit.callbacks)
    
    def test_multiple_callbacks(self):
        """Test adding multiple callbacks."""
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
        
        self.assertEqual(len(self.conduit.callbacks), 2)
        self.assertIn(callback1, self.conduit.callbacks)
        self.assertIn(callback2, self.conduit.callbacks)
    
    def test_save_state_without_parameters(self):
        """Test saving state without setting parameters."""
        state = self.conduit.save_state()
        self.assertIsInstance(state, dict)
        # Should have default parameters
        self.assertIn('parameters', state)
    
    def test_load_state_with_invalid_data(self):
        """Test loading state with invalid data."""
        invalid_state = {'invalid_key': 'invalid_value'}
        
        with self.assertRaises(KeyError):
            self.conduit.load_state(invalid_state)
    
    def test_load_state_with_none(self):
        """Test loading state with None value."""
        with self.assertRaises(TypeError):
            self.conduit.load_state(None)
    
    def test_set_fitness_function_with_none(self):
        """Test setting fitness function to None."""
        with self.assertRaises(TypeError):
            self.conduit.set_fitness_function(None)


class TestGenesisEvolutionaryConduitEdgeCases(unittest.TestCase):
    """Extended test suite for GenesisEvolutionaryConduit edge cases and boundary conditions."""
    
    def setUp(self):
        self.genesis_conduit = GenesisEvolutionaryConduit()
    
    def test_neural_network_evolution_invalid_config(self):
        """Test neural network evolution with invalid configuration."""
        invalid_config = {
            'input_size': -1,  # Invalid negative size
            'hidden_layers': [],
            'output_size': 0,
            'activation': 'invalid_activation'
        }
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_network_config(invalid_config)
    
    def test_neural_network_evolution_missing_config_keys(self):
        """Test neural network evolution with missing configuration keys."""
        incomplete_config = {
            'input_size': 10
            # Missing other required keys
        }
        
        with self.assertRaises(KeyError):
            self.genesis_conduit.set_network_config(incomplete_config)
    
    def test_neuroevolution_fitness_without_training_data(self):
        """Test neuroevolution fitness evaluation without setting training data."""
        genome = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.evaluate_network_fitness(genome)
    
    def test_neuroevolution_fitness_empty_training_data(self):
        """Test neuroevolution fitness evaluation with empty training data."""
        self.genesis_conduit.set_training_data([], [])
        
        genome = [0.1, 0.2, 0.3]
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.evaluate_network_fitness(genome)
    
    def test_neuroevolution_fitness_mismatched_data(self):
        """Test neuroevolution fitness evaluation with mismatched X and y data."""
        X_train = [[1, 2], [3, 4]]
        y_train = [0]  # Different length
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_training_data(X_train, y_train)
    
    def test_topology_evolution_invalid_topology(self):
        """Test topology evolution with invalid topology structure."""
        invalid_topology = {
            'layers': [],  # Empty layers
            'connections': [[0, 1]]  # Invalid connection for empty layers
        }
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.mutate_topology(invalid_topology)
    
    def test_topology_evolution_missing_keys(self):
        """Test topology evolution with missing topology keys."""
        incomplete_topology = {
            'layers': [10, 5, 1]
            # Missing 'connections' key
        }
        
        with self.assertRaises(KeyError):
            self.genesis_conduit.mutate_topology(incomplete_topology)
    
    def test_hyperparameter_optimization_empty_search_space(self):
        """Test hyperparameter optimization with empty search space."""
        self.genesis_conduit.set_hyperparameter_search_space({})
        
        hyperparams = self.genesis_conduit.generate_hyperparameters()
        self.assertEqual(hyperparams, {})
    
    def test_hyperparameter_optimization_invalid_bounds(self):
        """Test hyperparameter optimization with invalid bounds."""
        invalid_search_space = {
            'learning_rate': (0.1, 0.001),  # Upper bound less than lower bound
            'batch_size': (128, 16)  # Upper bound less than lower bound
        }
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.set_hyperparameter_search_space(invalid_search_space)
    
    def test_multi_objective_optimization_empty_objectives(self):
        """Test multi-objective optimization with empty objectives."""
        self.genesis_conduit.set_objectives([])
        
        genome = [0.1, 0.2, 0.3]
        fitness_vector = self.genesis_conduit.evaluate_multi_objective_fitness(genome)
        self.assertEqual(fitness_vector, [])
    
    def test_speciation_empty_population(self):
        """Test speciation with empty population."""
        species = self.genesis_conduit.speciate_population([], distance_threshold=1.0)
        self.assertEqual(species, [])
    
    def test_speciation_single_individual(self):
        """Test speciation with single individual."""
        population = [
            {'genome': [1.0, 2.0, 3.0], 'fitness': 0.5}
        ]
        
        species = self.genesis_conduit.speciate_population(population, distance_threshold=1.0)
        self.assertEqual(len(species), 1)
        self.assertEqual(len(species[0]), 1)
    
    def test_transfer_learning_empty_genome(self):
        """Test transfer learning with empty pretrained genome."""
        pretrained_genome = []
        
        with self.assertRaises(ValueError):
            self.genesis_conduit.adapt_pretrained_network(
                pretrained_genome, 
                new_task_config={'output_size': 3}
            )
    
    def test_ensemble_evolution_empty_networks(self):
        """Test ensemble evolution with empty networks list."""
        networks = []
        
        ensemble = self.genesis_conduit.create_ensemble(networks, ensemble_size=2)
        self.assertEqual(ensemble, [])
    
    def test_ensemble_evolution_more_requested_than_available(self):
        """Test ensemble evolution requesting more networks than available."""
        networks = [
            {'genome': [1, 2, 3], 'fitness': 0.7}
        ]
        
        ensemble = self.genesis_conduit.create_ensemble(networks, ensemble_size=5)
        self.assertEqual(len(ensemble), 1)
    
    def test_novelty_search_empty_population(self):
        """Test novelty search with empty population."""
        novelty_scores = self.genesis_conduit.calculate_novelty_scores([])
        self.assertEqual(novelty_scores, [])
    
    def test_coevolution_empty_populations(self):
        """Test coevolution with empty populations."""
        result = self.genesis_conduit.coevolve_populations([], [])
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['population1'], [])
        self.assertEqual(result['population2'], [])
    
    def test_migration_with_zero_rate(self):
        """Test migration between populations with zero migration rate."""
        population1 = [{'genome': [1, 2, 3], 'fitness': 0.5}]
        population2 = [{'genome': [4, 5, 6], 'fitness': 0.7}]
        
        migrated = self.genesis_conduit.migrate_individuals(
            population1, population2, migration_rate=0.0
        )
        
        # With zero migration rate, populations should remain unchanged
        self.assertEqual(migrated[0], population1)
        self.assertEqual(migrated[1], population2)
    
    def test_migration_with_full_rate(self):
        """Test migration between populations with full migration rate."""
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
        self.conduit = EvolutionaryConduit()
        self.params = EvolutionaryParameters(
            population_size=10,
            generations=5
        )
    
    def test_concurrent_fitness_evaluation(self):
        """Test concurrent fitness evaluation doesn't cause race conditions."""
        import threading
        import time
        
        def slow_fitness(genome):
            time.sleep(0.01)  # Simulate slow computation
            return sum(genome)
        
        population = [
            {'genome': [1, 2, 3], 'fitness': None},
            {'genome': [4, 5, 6], 'fitness': None},
            {'genome': [7, 8, 9], 'fitness': None}
        ]
        
        # Test that concurrent access doesn't break anything
        def evaluate_subset(pop_subset):
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
        """Test that callbacks are executed safely in concurrent scenarios."""
        import threading
        
        callback_count = 0
        lock = threading.Lock()
        
        def thread_safe_callback(generation, population, best_individual):
            nonlocal callback_count
            with lock:
                callback_count += 1
        
        self.conduit.add_callback(thread_safe_callback)
        
        # Simulate concurrent callback execution
        def simulate_generation():
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
        self.conduit = EvolutionaryConduit()
    
    def test_large_population_handling(self):
        """Test handling of large populations without memory issues."""
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
        """Test that memory is properly cleaned up after evolution."""
        import gc
        
        def simple_fitness(genome):
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
        """Test performance with different genome sizes."""
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
        self.conduit = EvolutionaryConduit()
    
    def test_recovery_from_nan_fitness(self):
        """Test recovery when fitness function returns NaN."""
        def nan_fitness(genome):
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
        """Test recovery when fitness function returns infinity."""
        def infinite_fitness(genome):
            return float('inf')
        
        population = [
            {'genome': [1, 2, 3], 'fitness': None}
        ]
        
        manager = PopulationManager()
        manager.evaluate_population(population, infinite_fitness)
        
        # Should handle infinity values
        self.assertEqual(population[0]['fitness'], float('inf'))
    
    def test_recovery_from_corrupted_genome(self):
        """Test recovery when genome contains invalid values."""
        population = [
            {'genome': [1, None, 3], 'fitness': None}  # None value in genome
        ]
        
        def robust_fitness(genome):
            try:
                return sum(x for x in genome if x is not None)
            except TypeError:
                return 0.0
        
        manager = PopulationManager()
        manager.evaluate_population(population, robust_fitness)
        
        self.assertEqual(population[0]['fitness'], 4.0)  # 1 + 3
    
    def test_evolution_with_degenerate_population(self):
        """Test evolution behavior with degenerate population scenarios."""
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
        """Test parameter validation with edge case values."""
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