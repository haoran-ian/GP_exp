import numpy as np

class ImprovedStochasticRanking:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.99
        self.temperature_threshold = 0.1
        self.min_population_size = 5
        self.momentum = 0.9  # New momentum term

    def adaptive_cooling(self, current_fitness, best_fitness, fitness_variance):
        # Dynamic cooling rate based on fitness variance
        dynamic_cooling = (best_fitness - current_fitness) / max(best_fitness, 1e-10) + fitness_variance
        return max(self.cooling_rate, dynamic_cooling)

    def stochastic_ranking(self, population, fitness):
        perm = np.random.permutation(len(population))
        ranked_population = population[perm]
        ranked_fitness = fitness[perm]
        for i in range(len(population) - 1):
            if np.random.rand() < 0.45 or ranked_fitness[i] < ranked_fitness[i+1]:
                ranked_population[i], ranked_population[i+1] = ranked_population[i+1], ranked_population[i]
                ranked_fitness[i], ranked_fitness[i+1] = ranked_fitness[i+1], ranked_fitness[i]
        return ranked_population, ranked_fitness

    def adjust_population_size(self, current_population_size, evals):
        remaining_budget = self.budget - evals
        adjusted_size = max(self.min_population_size, current_population_size * remaining_budget // self.budget)
        return adjusted_size

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evals = self.population_size
        temperature = self.initial_temperature

        while evals < self.budget:
            current_population_size = len(population)
            fitness_variance = np.var(fitness)

            for i in range(current_population_size):
                # Differential Evolution mutation and crossover with adaptive scaling
                idxs = [idx for idx in range(current_population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = 0.5 + self.momentum * np.random.rand() * (1 - fitness_variance)  # Adaptive mutation scaling factor influenced by variance
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < 0.9
                offspring = np.where(cross_points, mutant, population[i])
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                evals += 1
                
                # Replacement strategy
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness

                    # Simulated annealing acceptance criterion
                    if offspring_fitness < best_fitness or np.exp((best_fitness - offspring_fitness) / temperature) > np.random.rand():
                        best_solution = offspring
                        best_fitness = offspring_fitness
                
                # Update temperature with adaptive cooling
                temperature *= self.adaptive_cooling(offspring_fitness, best_fitness, fitness_variance)
                if temperature < self.temperature_threshold:
                    temperature = self.temperature_threshold

                if evals >= self.budget:
                    break

            # Apply stochastic ranking to enhance exploration and diversity
            population, fitness = self.stochastic_ranking(population, fitness)

            # Dynamically adjust population size
            adjusted_population_size = self.adjust_population_size(current_population_size, evals)
            if adjusted_population_size < current_population_size:
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices[:adjusted_population_size]]
                fitness = fitness[sorted_indices[:adjusted_population_size]]

        return best_solution