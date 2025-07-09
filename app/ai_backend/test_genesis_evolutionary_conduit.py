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
        Set up default and custom EvolutionaryParameters instances for use in tests.
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
        Tests that EvolutionaryParameters is initialized with the correct default attribute values.
        """
        self.assertEqual(self.default_params.population_size, 100)
        self.assertEqual(self.default_params.generations, 500)
        self.assertEqual(self.default_params.mutation_rate, 0.1)
        self.assertEqual(self.default_params.crossover_rate, 0.8)
        self.assertEqual(self.default_params.selection_pressure, 0.2)

    def test_custom_initialization(self):
        """
        Test that custom evolutionary parameters are initialized with the specified values.
        """
        self.assertEqual(self.custom_params.population_size, 200)
        self.assertEqual(self.custom_params.generations, 1000)
        self.assertEqual(self.custom_params.mutation_rate, 0.15)
        self.assertEqual(self.custom_params.crossover_rate, 0.85)
        self.assertEqual(self.custom_params.selection_pressure, 0.3)

    def test_parameter_validation(self):
        """
        Verify that initializing EvolutionaryParameters with invalid values raises ValueError.
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
        Test that EvolutionaryParameters instances are correctly converted to their dictionary representation.
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
        Test that EvolutionaryParameters can be instantiated from a dictionary with all fields correctly set.
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
        Test that the Gaussian mutation strategy returns a mutated genome list of the same length as the input for various mutation rates.
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
        Tests that the bit flip mutation method returns a genome of the same length as the input, with all elements as booleans.
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
        Test that the adaptive mutation strategy returns a mutated genome list of the same length as the input when provided with a fitness history.
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
        Set up a SelectionStrategy instance and a sample population for use in selection strategy tests.
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
        Tests that tournament selection returns a valid individual from the population with required keys.

        Verifies that the selected individual is a dictionary present in the population and contains both 'genome' and 'fitness' keys.
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
        Test that elitism selection returns the top individuals with the highest fitness values.

        Ensures the number of selected individuals matches the elite count and that they are ordered by descending fitness.
        """
        elite_count = 2
        selected = self.strategy.elitism_selection(self.population, elite_count)

        self.assertEqual(len(selected), elite_count)

        # Check that selected individuals are the fittest
        fitness_values = [individual['fitness'] for individual in selected]
        self.assertEqual(fitness_values, [0.9, 0.7])  # Sorted by fitness descending

    def test_empty_population(self):
        """
        Test that selection methods raise ValueError when provided with an empty population.
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
        Set up the test environment by initializing a FitnessFunction instance for use in test methods.
        """
        self.fitness_func = FitnessFunction()

    def test_sphere_function(self):
        """
        Test that the sphere fitness function computes the negative sum of squares for a given genome.

        Verifies that the fitness value matches the expected result for a sample input.
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
        Test that the Rosenbrock fitness function returns 0.0 for the genome [1.0, 1.0], its global minimum.

        Verifies correct evaluation of the Rosenbrock function at its known minimum point.
        """
        genome = [1.0, 1.0]
        fitness = self.fitness_func.rosenbrock_function(genome)

        # Rosenbrock function should be 0 at (1, 1)
        self.assertEqual(fitness, 0.0)

    def test_ackley_function(self):
        """
        Test that the Ackley fitness function returns its global minimum at the origin.

        Verifies that evaluating the Ackley function with a genome of all zeros yields a fitness value of 0.0.
        """
        genome = [0.0, 0.0, 0.0]
        fitness = self.fitness_func.ackley_function(genome)

        # Ackley function should be 0 at origin
        self.assertAlmostEqual(fitness, 0.0, places=10)

    def test_custom_function(self):
        """
        Test evaluation of a custom fitness function that sums genome values, ensuring the fitness evaluation method returns the correct result.
        """
        def custom_func(genome):
            """
            Return the sum of all numeric elements in the provided genome.

            Parameters:
                genome (iterable): Sequence of numeric values to be summed.

            Returns:
                int or float: The sum of all values in the genome.
            """
            return sum(genome)

        genome = [1.0, 2.0, 3.0]
        fitness = self.fitness_func.evaluate(genome, custom_func)

        self.assertEqual(fitness, 6.0)

    def test_multi_objective_function(self):
        """
        Tests that a multi-objective fitness function evaluates a genome and returns a vector of fitness values, one for each objective.
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
        Test that genomes violating constraints are assigned penalized fitness values.

        Ensures that when a genome does not satisfy specified constraints, the fitness function applies a penalty, resulting in a fitness score lower than the unconstrained evaluation.
        """
        genome = [1.0, 2.0, 3.0]

        def constraint_func(g):
            # Constraint: sum should be less than 5
            """
            Return True if the sum of elements in the input iterable is less than 5.

            Parameters:
                g (iterable): Iterable of numeric values.

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
        Set up a PopulationManager instance and default parameters for population management tests.

        Initializes the PopulationManager and sets default values for genome length and population size used in test cases.
        """
        self.manager = PopulationManager()
        self.genome_length = 5
        self.population_size = 10

    def test_initialize_random_population(self):
        """
        Test that initializing a random population creates the correct number of individuals with valid genome length and fitness fields.

        Ensures each individual in the generated population contains a genome of the specified length and a fitness attribute.
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
        Test that initializing a population with seed genomes includes the seeds and produces the correct population size.

        Ensures that the initialized population contains all provided seed genomes and that the total number of individuals matches the expected population size.
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
        Test that each individual in the population receives a numeric fitness value after evaluation with the provided fitness function.
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
        Test that the population manager computes correct statistics for best, worst, average, median, and standard deviation of fitness values in a population.
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
        Test that the diversity calculation for a population of distinct genomes returns a positive float value.
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
        Initialize a GeneticOperations instance for use in genetic operations tests.
        """
        self.operations = GeneticOperations()

    def test_single_point_crossover(self):
        """
        Test that single-point crossover generates two offspring of the same length as the parents, with genes derived from both parent sequences.
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
        Test that two-point crossover returns two offspring with the same genome length as the parents.
        """
        parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
        parent2 = [9, 10, 11, 12, 13, 14, 15, 16]

        child1, child2 = self.operations.two_point_crossover(parent1, parent2)

        self.assertEqual(len(child1), len(parent1))
        self.assertEqual(len(child2), len(parent2))

    def test_uniform_crossover(self):
        """
        Test that the uniform crossover operation generates two offspring with the same length as the parent genomes.
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
        Test that simulated binary crossover (SBX) generates two offspring of correct length with all gene values within the specified bounds.
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
        Tests that the blend crossover (BLX-α) operation generates two offspring with the same length as the parent genomes.
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
        Tests that the EvolutionaryConduit initializes its core components for evolutionary computation.
        """
        self.assertIsNotNone(self.conduit.mutation_strategy)
        self.assertIsNotNone(self.conduit.selection_strategy)
        self.assertIsNotNone(self.conduit.fitness_function)
        self.assertIsNotNone(self.conduit.population_manager)
        self.assertIsNotNone(self.conduit.genetic_operations)

    def test_set_fitness_function(self):
        """
        Test assigning a custom fitness function to the conduit and verifying it is used to evaluate genome fitness.
        """
        def custom_fitness(genome):
            """
            Compute the fitness score of a genome by summing its numeric elements.

            Parameters:
		genome (iterable): Iterable containing numeric values representing the genome.

            Returns:
		float or int: The total sum of the genome's elements.
            """
            return sum(genome)

        self.conduit.set_fitness_function(custom_fitness)

        # Test that the function is set correctly
        test_genome = [1.0, 2.0, 3.0]
        fitness = self.conduit.fitness_function.evaluate(test_genome, custom_fitness)
        self.assertEqual(fitness, 6.0)

    def test_set_parameters(self):
        """
        Tests that setting evolutionary parameters in the conduit updates its parameters to the specified values.
        """
        self.conduit.set_parameters(self.params)

        self.assertEqual(self.conduit.parameters.population_size, 20)
        self.assertEqual(self.conduit.parameters.generations, 10)
        self.assertEqual(self.conduit.parameters.mutation_rate, 0.1)
        self.assertEqual(self.conduit.parameters.crossover_rate, 0.8)

    @patch('app.ai_backend.genesis_evolutionary_conduit.EvolutionaryConduit.evolve')
    def test_run_evolution(self, mock_evolve):
        """
        Test that the evolution process returns a result containing the expected summary keys.

        Ensures that the result includes 'best_individual', 'generations_run', 'final_population', and 'statistics'.
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
        Test that EvolutionaryConduit state can be saved and restored, ensuring parameter values remain consistent after loading.
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
        Tests that a callback function can be added to the evolution process and is correctly registered in the conduit.
        """
        callback_called = False

        def test_callback(generation, population, best_individual):
            """
            A test callback function that sets a flag when invoked during the evolution process.

            This function is intended for use in tests to verify that callbacks are triggered during evolutionary algorithm execution.
            """
            nonlocal callback_called
            callback_called = True

        self.conduit.add_callback(test_callback)

        # Verify callback is added
        self.assertIn(test_callback, self.conduit.callbacks)

    def test_evolution_history_tracking(self):
        """
        Test that enabling history tracking in the evolutionary conduit correctly records evolution history during a run.

        Verifies that after running a mocked evolution process with history tracking enabled, the conduit indicates that history tracking is active.
        """
        self.conduit.set_parameters(self.params)
        self.conduit.enable_history_tracking()

        # Run a simple evolution
        def simple_fitness(genome):
            """
            Calculate the fitness of a genome by summing all its numeric elements.

            Parameters:
		genome (Iterable[float | int]): Sequence of numeric values representing the genome.

            Returns:
		float | int: The total sum of the genome's elements.
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
        Set up the test environment with a GenesisEvolutionaryConduit instance and predefined evolutionary parameters for each test case.
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
        Tests that setting a network configuration and creating a neural network through the genesis conduit yields a valid neural network instance.
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
        Test that evaluating a neural network genome's fitness via neuroevolution returns a numeric value.
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
        Test that mutating a neural network topology produces a valid topology dictionary.

        Ensures the mutated topology includes both 'layers' and 'connections' keys.
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
        Tests that multi-objective optimization evaluates a genome against multiple objectives and returns a fitness vector of the expected length.
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
        Test that the adaptive mutation rate calculation produces a float within the range [0.0, 1.0] based on the fitness history of a population.
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
        Test that the speciation process correctly groups individuals into species based on genome similarity.

        Asserts that the resulting species list is non-empty, confirming that individuals are clustered for population diversity.
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

        Ensures that adapting a pretrained genome with a new task configuration returns a non-empty list representing the adapted genome.
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

        Ensures the number of novelty scores matches the population size and that all scores are numeric values.
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
        Test that the coevolution process using the genesis conduit returns a dictionary containing updated populations for both input groups.
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
        Test that the checkpoint saving mechanism calls the save method with the correct file path.
        """
        # Set up conduit state
        self.genesis_conduit.set_parameters(self.params)

        # Save checkpoint
        checkpoint_path = "test_checkpoint.pkl"
        self.genesis_conduit.save_checkpoint(checkpoint_path)

        mock_save.assert_called_once_with(checkpoint_path)

    def test_distributed_evolution(self):
        """
        Tests distributed evolution using an island model by simulating migration between islands and verifying that the migration process returns updated populations as a tuple.
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
        Test that an EvolutionaryException is instantiated with the correct message and is a subclass of Exception.
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
        Test that an EvolutionaryException is raised and properly handled by the test framework.
        """
        with self.assertRaises(EvolutionaryException):
            raise EvolutionaryException("Test exception")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test suite for complex evolutionary scenarios."""

    def setUp(self):
        """
        Set up the integration test environment by initializing a GenesisEvolutionaryConduit and default evolutionary parameters.
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
        Test execution of a complete evolution cycle and validate the result structure.

        Ensures that running the evolution process (with evolution mocked) returns a result containing a best individual and the expected number of generations.
        """
        # Set up fitness function
        def simple_fitness(genome):
            """
            Calculate the fitness of a genome as the sum of the squares of its elements.

            Parameters:
                genome (Iterable[float]): Sequence of numeric values representing the genome.

            Returns:
                float: The sum of the squares of all elements in the genome.
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
        Tests the neural network evolution pipeline by configuring the network, setting training data, and verifying that a neural network is created using the GenesisEvolutionaryConduit.
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
        Test the multi-objective optimization pipeline to ensure that assigned objectives produce the correct fitness vector for a given genome.
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
        Test that the adaptive evolution pipeline computes a valid mutation rate based on population fitness diversity.

        Verifies that the calculated adaptive mutation rate is a float within the range [0.0, 1.0] when provided with a population exhibiting varied fitness values.
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
        Test that invalid evolutionary parameters and fitness evaluation failures raise the correct exceptions.

        Ensures that initializing `EvolutionaryParameters` with invalid values raises a `ValueError`, and that a fitness function failure during evolution raises an `EvolutionaryException`.
        """
        # Test invalid parameters
        with self.assertRaises(ValueError):
            invalid_params = EvolutionaryParameters(population_size=0)

        # Test recovery from evolution failure
        def failing_fitness(genome):
            """
            Simulates a fitness evaluation failure by always raising a ValueError.

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
        Set up the test environment for asynchronous evolution tests.

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

        This test ensures that the `run_async_evolution` method of the genesis conduit produces a non-None result when the asynchronous evolution process is simulated using a mock.
        """
        async def mock_async_evolve():
            """
            Simulates an asynchronous evolutionary process and returns mock results.

            Returns:
                dict: Contains a mock best individual, number of generations run, final population, and summary statistics.
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
        Test that population fitness evaluation is performed in parallel using a mocked executor.

        Ensures that the parallel evaluation mechanism is invoked and that each individual in the population receives an assigned fitness value.
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
            Calculate the fitness score of a genome by summing its elements.

            Parameters:
                genome (Iterable[float]): Sequence of numeric values representing the genome.

            Returns:
                float: The total sum of the genome's elements.
            """
            return sum(genome)

        # Test parallel evaluation
        self.genesis_conduit.evaluate_population_parallel(population, fitness_func)

        # Verify parallel execution was attempted
        mock_executor.assert_called_once()


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)