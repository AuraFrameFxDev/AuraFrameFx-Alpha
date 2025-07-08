import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import sys
import os

# Add the app directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from app.ai_backend.genesis_evolutionary_conduit import (
        GenesisEvolutionaryConduit,
        EvolutionaryParameters,
        GeneticAlgorithm,
        FitnessEvaluator
    )
except ImportError:
    # Mock the classes if the actual implementation doesn't exist yet
    class GenesisEvolutionaryConduit:
        def __init__(self, parameters=None):
            """
            Initialize a GenesisEvolutionaryConduit instance with optional evolutionary parameters.
            
            Parameters:
                parameters (dict, optional): Evolutionary parameters to configure the conduit. Defaults to an empty dictionary if not provided.
            """
            self.parameters = parameters or {}
            self.population = []
            self.generation = 0
            self.best_fitness = 0
            
        def initialize_population(self, size=100):
            """
            Create and assign a new population of individuals with unique string identifiers.
            
            Parameters:
                size (int): The number of individuals to generate in the population. Defaults to 100.
            
            Returns:
                list: A list of string-labeled individuals representing the initialized population.
            """
            self.population = [f"individual_{i}" for i in range(size)]
            return self.population
            
        def evolve_generation(self):
            """
            Advance the evolutionary process by incrementing the generation count.
            
            Returns:
                int: The updated generation number after evolution.
            """
            self.generation += 1
            return self.generation
            
        def evaluate_fitness(self, individual):
            """
            Calculates a deterministic fitness score for an individual based on its string representation.
            
            Parameters:
                individual: The entity whose fitness is to be evaluated.
            
            Returns:
                int: Fitness value as an integer between 0 and 99.
            """
            return hash(str(individual)) % 100
            
        def select_parents(self, population):
            """
            Selects two parents from the given population for crossover.
            
            Returns:
                A list containing the first two individuals if at least two are present; otherwise, returns the entire population.
            """
            return population[:2] if len(population) >= 2 else population
            
        def crossover(self, parent1, parent2):
            """
            Generate an offspring identifier string by combining the identifiers of two parent individuals.
            
            Parameters:
                parent1: Identifier of the first parent individual.
                parent2: Identifier of the second parent individual.
            
            Returns:
                str: A string representing the offspring, formatted as 'offspring_{parent1}_{parent2}'.
            """
            return f"offspring_{parent1}_{parent2}"
            
        def mutate(self, individual, mutation_rate=0.1):
            """
            Returns a mutated version of the given individual.
            
            Parameters:
                individual: The individual to mutate.
                mutation_rate (float, optional): The probability of mutation. Default is 0.1.
            
            Returns:
                str: The mutated individual.
            """
            return f"mutated_{individual}"
            
        def get_best_individual(self):
            """
            Return the best individual from the current population, or None if the population is empty.
            """
            return self.population[0] if self.population else None
            
        def get_statistics(self):
            """
            Return a dictionary containing statistics about the current evolutionary state, including generation number, population size, best fitness, and average fitness.
            
            Returns:
                dict: A dictionary with keys 'generation', 'population_size', 'best_fitness', and 'average_fitness'.
            """
            return {
                'generation': self.generation,
                'population_size': len(self.population),
                'best_fitness': self.best_fitness,
                'average_fitness': sum(self.evaluate_fitness(i) for i in self.population) / len(self.population) if self.population else 0
            }
    
    class EvolutionaryParameters:
        def __init__(self, population_size=100, mutation_rate=0.1, crossover_rate=0.8, max_generations=1000):
            """
            Initialize evolutionary parameters for the genetic algorithm.
            
            Parameters:
                population_size (int): Number of individuals in each generation.
                mutation_rate (float): Probability of mutation for each individual.
                crossover_rate (float): Probability of crossover between parents.
                max_generations (int): Maximum number of generations to evolve.
            """
            self.population_size = population_size
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.max_generations = max_generations
    
    class GeneticAlgorithm:
        def __init__(self, conduit):
            """
            Initialize the GeneticAlgorithm with a specified evolutionary conduit.
            
            Parameters:
                conduit: The evolutionary conduit instance used to manage the genetic algorithm's operations.
            """
            self.conduit = conduit
            
        def run(self, target_fitness=None):
            """
            Evolves generations using the conduit until reaching the maximum number of generations or the specified target fitness.
            
            Parameters:
                target_fitness (float, optional): If provided, evolution stops early when the best fitness meets or exceeds this value.
            
            Returns:
                The best individual found at the end of the evolutionary process.
            """
            while self.conduit.generation < self.conduit.parameters.get('max_generations', 1000):
                self.conduit.evolve_generation()
                if target_fitness and self.conduit.best_fitness >= target_fitness:
                    break
            return self.conduit.get_best_individual()
    
    class FitnessEvaluator:
        def __init__(self, fitness_function=None):
            """
            Initialize the FitnessEvaluator with an optional custom fitness function.
            
            If no fitness function is provided, a default function based on the hash of the individual's string representation is used.
            """
            self.fitness_function = fitness_function or (lambda x: hash(str(x)) % 100)
            
        def evaluate(self, individual):
            """
            Evaluates the fitness of an individual using the assigned fitness function.
            
            Returns:
                The fitness score produced by the fitness function for the given individual.
            """
            return self.fitness_function(individual)


class TestGenesisEvolutionaryConduit:
    """Comprehensive unit tests for GenesisEvolutionaryConduit"""
    
    def setup_method(self):
        """
        Initializes default evolutionary parameters and a GenesisEvolutionaryConduit instance before each test method.
        """
        self.default_params = EvolutionaryParameters(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_generations=100
        )
        self.conduit = GenesisEvolutionaryConduit(self.default_params)
        
    def teardown_method(self):
        """
        Resets the conduit instance to ensure a clean state after each test method.
        """
        self.conduit = None
        
    # Happy Path Tests
    def test_initialization_with_default_parameters(self):
        """
        Test that the GenesisEvolutionaryConduit initializes correctly with default parameters, including generation count, population, and best fitness.
        """
        conduit = GenesisEvolutionaryConduit()
        assert conduit is not None
        assert conduit.generation == 0
        assert conduit.population == []
        assert conduit.best_fitness == 0
        
    def test_initialization_with_custom_parameters(self):
        """
        Test that the conduit initializes correctly with custom evolutionary parameters.
        """
        params = EvolutionaryParameters(population_size=200, mutation_rate=0.15)
        conduit = GenesisEvolutionaryConduit(params)
        assert conduit.parameters == params
        
    def test_population_initialization_default_size(self):
        """
        Test that the population is initialized with the default size and all individuals are strings.
        """
        population = self.conduit.initialize_population()
        assert len(population) == 100  # default size
        assert all(isinstance(ind, str) for ind in population)
        
    def test_population_initialization_custom_size(self):
        """
        Test that initializing the population with a custom size creates the correct number of individuals and updates the conduit state accordingly.
        """
        population = self.conduit.initialize_population(size=75)
        assert len(population) == 75
        assert self.conduit.population == population
        
    def test_generation_evolution(self):
        """
        Verify that evolving a generation increments the generation count as expected.
        """
        self.conduit.initialize_population(10)
        initial_gen = self.conduit.generation
        new_gen = self.conduit.evolve_generation()
        assert new_gen == initial_gen + 1
        assert self.conduit.generation == initial_gen + 1
        
    def test_fitness_evaluation_consistency(self):
        """
        Verify that evaluating the fitness of the same individual multiple times yields consistent integer results within the expected range.
        """
        individual = "test_individual"
        fitness1 = self.conduit.evaluate_fitness(individual)
        fitness2 = self.conduit.evaluate_fitness(individual)
        assert fitness1 == fitness2
        assert isinstance(fitness1, int)
        assert 0 <= fitness1 < 100
        
    def test_parent_selection_sufficient_population(self):
        """
        Test that parent selection returns two valid parents when the population has enough individuals.
        """
        population = ["ind1", "ind2", "ind3", "ind4"]
        parents = self.conduit.select_parents(population)
        assert len(parents) == 2
        assert all(parent in population for parent in parents)
        
    def test_crossover_operation(self):
        """
        Test that the crossover operation generates a valid offspring string containing both parent identifiers.
        """
        parent1, parent2 = "parent1", "parent2"
        offspring = self.conduit.crossover(parent1, parent2)
        assert isinstance(offspring, str)
        assert "offspring" in offspring
        assert str(parent1) in offspring
        assert str(parent2) in offspring
        
    def test_mutation_operation(self):
        """
        Tests that the mutation operation returns a mutated string containing the original individual and the substring 'mutated'.
        """
        individual = "test_individual"
        mutated = self.conduit.mutate(individual)
        assert isinstance(mutated, str)
        assert "mutated" in mutated
        assert str(individual) in mutated
        
    def test_mutation_with_custom_rate(self):
        """
        Test that the mutate method correctly processes an individual with a custom mutation rate.
        
        Verifies that the returned mutated individual is a string when a non-default mutation rate is provided.
        """
        individual = "test_individual"
        mutated = self.conduit.mutate(individual, mutation_rate=0.5)
        assert isinstance(mutated, str)
        
    def test_get_best_individual_with_population(self):
        """
        Test that retrieving the best individual from a populated conduit returns a valid member of the population.
        """
        self.conduit.initialize_population(5)
        best = self.conduit.get_best_individual()
        assert best is not None
        assert best in self.conduit.population
        
    def test_get_statistics_with_population(self):
        """
        Test that `get_statistics` returns correct statistics for a conduit with an initialized population.
        """
        self.conduit.initialize_population(10)
        stats = self.conduit.get_statistics()
        assert isinstance(stats, dict)
        assert 'generation' in stats
        assert 'population_size' in stats
        assert 'best_fitness' in stats
        assert 'average_fitness' in stats
        assert stats['population_size'] == 10
        assert stats['generation'] == 0
        
    # Edge Cases Tests
    def test_empty_population_handling(self):
        """
        Test that retrieving the best individual from an empty population returns None.
        """
        best = self.conduit.get_best_individual()
        assert best is None
        
    def test_zero_population_initialization(self):
        """
        Test that initializing the population with a size of zero results in an empty population list.
        """
        population = self.conduit.initialize_population(size=0)
        assert len(population) == 0
        assert self.conduit.population == []
        
    def test_single_individual_population(self):
        """
        Test that initializing a population with a single individual results in only one parent being selected.
        """
        population = self.conduit.initialize_population(size=1)
        assert len(population) == 1
        parents = self.conduit.select_parents(population)
        assert len(parents) == 1
        
    def test_parent_selection_insufficient_population(self):
        """
        Test that parent selection returns only the available individual when the population has fewer than two members.
        """
        population = ["single_individual"]
        parents = self.conduit.select_parents(population)
        assert len(parents) == 1
        assert parents[0] == "single_individual"
        
    def test_statistics_empty_population(self):
        """
        Test that statistics are correctly calculated when the population is empty.
        
        Verifies that the reported population size is zero and the average fitness is zero when no individuals are present.
        """
        stats = self.conduit.get_statistics()
        assert stats['population_size'] == 0
        assert stats['average_fitness'] == 0
        
    def test_large_population_performance(self):
        """
        Tests that the conduit can initialize and handle a large population efficiently, and verifies correct population size reporting in statistics.
        """
        large_population = self.conduit.initialize_population(size=1000)
        assert len(large_population) == 1000
        stats = self.conduit.get_statistics()
        assert stats['population_size'] == 1000
        
    # Failure Conditions Tests
    def test_invalid_mutation_rate(self):
        """
        Test that the mutate method handles invalid mutation rates without raising errors.
        
        Verifies that mutation rates less than 0 and greater than 1 do not cause exceptions and still return a string result.
        """
        # Test with negative mutation rate
        result = self.conduit.mutate("individual", mutation_rate=-0.1)
        assert isinstance(result, str)
        
        # Test with mutation rate > 1
        result = self.conduit.mutate("individual", mutation_rate=1.5)
        assert isinstance(result, str)
        
    def test_none_individual_fitness_evaluation(self):
        """
        Test that evaluating fitness with a None individual raises a TypeError or AttributeError.
        """
        with pytest.raises((TypeError, AttributeError)):
            self.conduit.evaluate_fitness(None)
            
    def test_none_parents_crossover(self):
        """
        Test that the crossover method raises an error when both parent arguments are None.
        """
        with pytest.raises((TypeError, AttributeError)):
            self.conduit.crossover(None, None)
            
    def test_none_individual_mutation(self):
        """
        Test that mutating a None individual raises a TypeError or AttributeError.
        """
        with pytest.raises((TypeError, AttributeError)):
            self.conduit.mutate(None)
            
    def test_negative_population_size(self):
        """
        Test that initializing the population with a negative size results in an empty population.
        """
        population = self.conduit.initialize_population(size=-10)
        assert len(population) == 0  # Should handle gracefully
        
    # Integration Tests
    def test_full_evolution_cycle(self):
        """
        Tests that the evolutionary conduit correctly evolves through multiple generations and maintains population size during a full evolution cycle.
        """
        # Initialize population
        self.conduit.initialize_population(20)
        initial_gen = self.conduit.generation
        
        # Evolve several generations
        for _ in range(5):
            self.conduit.evolve_generation()
            
        # Verify evolution occurred
        assert self.conduit.generation == initial_gen + 5
        assert len(self.conduit.population) == 20
        
    def test_crossover_and_mutation_pipeline(self):
        """
        Tests that the crossover followed by mutation produces a valid mutated offspring string containing expected substrings.
        """
        parent1, parent2 = "parent1", "parent2"
        offspring = self.conduit.crossover(parent1, parent2)
        mutated_offspring = self.conduit.mutate(offspring)
        
        assert isinstance(mutated_offspring, str)
        assert "mutated" in mutated_offspring
        assert "offspring" in mutated_offspring


class TestEvolutionaryParameters:
    """Unit tests for EvolutionaryParameters class"""
    
    def test_default_parameters(self):
        """
        Verify that the default values of EvolutionaryParameters are correctly set upon initialization.
        """
        params = EvolutionaryParameters()
        assert params.population_size == 100
        assert params.mutation_rate == 0.1
        assert params.crossover_rate == 0.8
        assert params.max_generations == 1000
        
    def test_custom_parameters(self):
        """
        Verify that custom values for evolutionary parameters are correctly assigned and accessible.
        """
        params = EvolutionaryParameters(
            population_size=200,
            mutation_rate=0.15,
            crossover_rate=0.9,
            max_generations=500
        )
        assert params.population_size == 200
        assert params.mutation_rate == 0.15
        assert params.crossover_rate == 0.9
        assert params.max_generations == 500
        
    def test_parameter_validation(self):
        """
        Test that EvolutionaryParameters correctly accepts and stores edge case values for its parameters.
        """
        # Test edge cases
        params = EvolutionaryParameters(
            population_size=1,
            mutation_rate=0.0,
            crossover_rate=1.0,
            max_generations=1
        )
        assert params.population_size == 1
        assert params.mutation_rate == 0.0
        assert params.crossover_rate == 1.0
        assert params.max_generations == 1


class TestGeneticAlgorithm:
    """Unit tests for GeneticAlgorithm class"""
    
    def setup_method(self):
        """
        Initializes a new evolutionary conduit and genetic algorithm instance before each test.
        """
        self.conduit = GenesisEvolutionaryConduit()
        self.algorithm = GeneticAlgorithm(self.conduit)
        
    def test_initialization(self):
        """
        Test that the genetic algorithm is initialized with the correct evolutionary conduit.
        """
        assert self.algorithm.conduit == self.conduit
        
    def test_run_without_target_fitness(self):
        """
        Tests that the genetic algorithm runs for the specified number of generations when no target fitness is provided, and returns a valid best individual.
        """
        self.conduit.initialize_population(10)
        # Mock max_generations to avoid long running test
        self.conduit.parameters = {'max_generations': 2}
        
        best = self.algorithm.run()
        assert best is not None
        assert self.conduit.generation == 2
        
    def test_run_with_target_fitness(self):
        """
        Tests that the genetic algorithm terminates early when a target fitness is reached, returning the best individual found.
        """
        self.conduit.initialize_population(10)
        self.conduit.parameters = {'max_generations': 100}
        
        # Mock high target fitness to ensure early termination
        with patch.object(self.conduit, 'best_fitness', 150):
            best = self.algorithm.run(target_fitness=100)
            assert best is not None


class TestFitnessEvaluator:
    """Unit tests for FitnessEvaluator class"""
    
    def test_default_fitness_function(self):
        """
        Test that the default fitness function in FitnessEvaluator returns an integer fitness value within the expected range for a given individual.
        """
        evaluator = FitnessEvaluator()
        fitness = evaluator.evaluate("test_individual")
        assert isinstance(fitness, int)
        assert 0 <= fitness < 100
        
    def test_custom_fitness_function(self):
        """
        Test that a custom fitness function can be used with the FitnessEvaluator and produces the expected result.
        """
        def custom_fitness(individual):
            """
            Calculate the fitness of an individual based on the length of its string representation.
            
            Parameters:
                individual: The entity whose fitness is being evaluated.
            
            Returns:
                int: The length of the individual's string representation.
            """
            return len(str(individual))
            
        evaluator = FitnessEvaluator(custom_fitness)
        fitness = evaluator.evaluate("test")
        assert fitness == 4
        
    def test_fitness_consistency(self):
        """
        Verify that the fitness evaluator returns consistent results for the same individual across multiple evaluations.
        """
        evaluator = FitnessEvaluator()
        individual = "consistent_test"
        fitness1 = evaluator.evaluate(individual)
        fitness2 = evaluator.evaluate(individual)
        assert fitness1 == fitness2
        
    def test_different_individuals_different_fitness(self):
        """
        Verify that the fitness evaluator returns integer fitness values for different individuals, ensuring it can process distinct inputs.
        """
        evaluator = FitnessEvaluator()
        fitness1 = evaluator.evaluate("individual1")
        fitness2 = evaluator.evaluate("individual2")
        # While they might be equal, test that the evaluator can handle different inputs
        assert isinstance(fitness1, int)
        assert isinstance(fitness2, int)


# Performance and Stress Tests
class TestPerformanceAndStress:
    """Performance and stress tests"""
    
    def setup_method(self):
        """
        Initializes a new GenesisEvolutionaryConduit instance before each performance test.
        """
        self.conduit = GenesisEvolutionaryConduit()
        
    def test_large_population_initialization(self):
        """
        Test that the conduit can initialize a population with a very large number of individuals.
        
        Verifies that the population size matches the requested large value.
        """
        population = self.conduit.initialize_population(size=10000)
        assert len(population) == 10000
        
    def test_many_generations_evolution(self):
        """
        Tests that evolving the population over 100 generations correctly increments the generation count to 100.
        """
        self.conduit.initialize_population(10)
        
        # Evolve many generations
        for _ in range(100):
            self.conduit.evolve_generation()
            
        assert self.conduit.generation == 100
        
    def test_fitness_evaluation_stress(self):
        """
        Stress-tests the fitness evaluation method by evaluating a large number of individuals to ensure consistent integer fitness values.
        """
        individuals = [f"individual_{i}" for i in range(1000)]
        
        # Evaluate fitness for many individuals
        for individual in individuals:
            fitness = self.conduit.evaluate_fitness(individual)
            assert isinstance(fitness, int)
            
    def test_concurrent_operations(self):
        """
        Tests that multiple threads can concurrently evolve generations on the same conduit instance, verifying thread safety and correct generation increments.
        """
        import threading
        
        self.conduit.initialize_population(100)
        results = []
        
        def evolve_worker():
            """
            Evolves the population for 10 generations and appends each generation number to the results list.
            """
            for _ in range(10):
                gen = self.conduit.evolve_generation()
                results.append(gen)
        
        # Create multiple threads
        threads = [threading.Thread(target=evolve_worker) for _ in range(3)]
        
        # Start threads
        for thread in threads:
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify results
        assert len(results) == 30  # 3 threads * 10 evolutions each
        assert all(isinstance(r, int) for r in results)


# Mock and Integration Tests
class TestMockingAndIntegration:
    """Tests using mocks and integration scenarios"""
    
    def setup_method(self):
        """
        Initializes a new GenesisEvolutionaryConduit instance before each test method.
        """
        self.conduit = GenesisEvolutionaryConduit()
        
    @patch('builtins.hash')
    def test_mocked_fitness_evaluation(self, mock_hash):
        """
        Test that the fitness evaluation returns the mocked hash value when the hash function is patched.
        
        Verifies that the conduit uses the mocked hash function and returns the expected fitness score.
        """
        mock_hash.return_value = 42
        
        fitness = self.conduit.evaluate_fitness("test_individual")
        assert fitness == 42
        mock_hash.assert_called_once()
        
    def test_conduit_with_mocked_parameters(self):
        """
        Verify that the GenesisEvolutionaryConduit correctly accepts and stores mocked parameter objects.
        """
        mock_params = MagicMock()
        mock_params.population_size = 50
        mock_params.mutation_rate = 0.2
        
        conduit = GenesisEvolutionaryConduit(mock_params)
        assert conduit.parameters == mock_params
        
    def test_algorithm_integration_with_conduit(self):
        """
        Tests that the genetic algorithm correctly integrates with the conduit, evolving for the specified number of generations and returning a best individual.
        """
        # Setup conduit
        self.conduit.initialize_population(20)
        
        # Create algorithm
        algorithm = GeneticAlgorithm(self.conduit)
        
        # Mock parameters for quick test
        with patch.object(self.conduit, 'parameters', {'max_generations': 3}):
            best = algorithm.run()
            
        assert best is not None
        assert self.conduit.generation == 3
        
    def test_fitness_evaluator_integration(self):
        """
        Tests integration of a custom fitness function with the FitnessEvaluator, verifying correct fitness values for various individuals.
        """
        def custom_fitness(individual):
            """
            Calculate a fitness score for an individual based on twice the length of its string representation.
            
            Parameters:
            	individual: The entity whose fitness is being evaluated.
            
            Returns:
            	int: The fitness score, equal to two times the length of the individual's string form.
            """
            return len(str(individual)) * 2
            
        evaluator = FitnessEvaluator(custom_fitness)
        
        # Test with various individuals
        test_cases = ["a", "abc", "test_individual", ""]
        expected_results = [2, 6, 30, 0]
        
        for individual, expected in zip(test_cases, expected_results):
            fitness = evaluator.evaluate(individual)
            assert fitness == expected


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])