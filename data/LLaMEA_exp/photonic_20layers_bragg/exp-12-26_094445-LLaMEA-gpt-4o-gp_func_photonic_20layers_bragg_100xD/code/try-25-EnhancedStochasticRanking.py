import numpy as np

class EnhancedStochasticRanking:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.99
        self.temperature_threshold = 0.1
        self.p_swap = 0.45

    def adaptive_cooling(self, current_fitness, best_fitness):
        return max(self.cooling_rate, (best_fitness - current_fitness) / max(best_fitness, 1e-10))

    def adaptive_stochastic_ranking(self, population, fitness):
        perm = np.random.permutation(self.population_size)
        ranked_population = population[perm]
        ranked_fitness = fitness[perm]
        for i in range(self.population_size - 1):
            swap_prob = self.p_swap * (1 - (ranked_fitness[i+1] - ranked_fitness[i]) / max(ranked_fitness[i+1], 1e-10))
            if np.random.rand() < swap_prob or ranked_fitness[i] < ranked_fitness[i+1]:
                ranked_population[i], ranked_population[i+1] = ranked_population[i+1], ranked_population[i]
                ranked_fitness[i], ranked_fitness[i+1] = ranked_fitness[i+1], ranked_fitness[i]
        return ranked_population, ranked_fitness

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
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover with adaptive scaling
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = 0.5 + 0.4 * np.random.rand()  # Adaptive mutation scaling factor
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
                temperature *= self.adaptive_cooling(offspring_fitness, best_fitness)
                if temperature < self.temperature_threshold:
                    temperature = self.temperature_threshold

                if evals >= self.budget:
                    break

            # Apply adaptive stochastic ranking to enhance exploration and diversity
            population, fitness = self.adaptive_stochastic_ranking(population, fitness)

        return best_solution