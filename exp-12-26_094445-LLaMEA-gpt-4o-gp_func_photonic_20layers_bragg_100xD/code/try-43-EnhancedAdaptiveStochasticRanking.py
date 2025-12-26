import numpy as np

class EnhancedAdaptiveStochasticRanking:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.min_population_size = 5
        self.mutation_rate = 0.9
        self.learning_rate = 0.1
    
    def chaotic_map(self, x):
        return 4.0 * x * (1.0 - x)

    def stochastic_ranking(self, population, fitness):
        perm = np.random.permutation(len(population))
        ranked_population = population[perm]
        ranked_fitness = fitness[perm]
        for i in range(len(population) - 1):
            if np.random.rand() < 0.45 or ranked_fitness[i] < ranked_fitness[i + 1]:
                ranked_population[i], ranked_population[i + 1] = ranked_population[i + 1], ranked_population[i]
                ranked_fitness[i], ranked_fitness[i + 1] = ranked_fitness[i + 1], ranked_fitness[i]
        return ranked_population, ranked_fitness

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evals = self.population_size
        chaos_value = np.random.rand()

        while evals < self.budget:
            current_population_size = len(population)
            for i in range(current_population_size):
                idxs = [idx for idx in range(current_population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = 0.5 + chaos_value * self.learning_rate  # Adaptive F
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < self.mutation_rate
                offspring = np.where(cross_points, mutant, population[i])

                offspring_fitness = func(offspring)
                evals += 1
                
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    
                    if offspring_fitness < best_fitness:
                        best_solution = offspring
                        best_fitness = offspring_fitness
                        self.learning_rate = 0.1 * (best_fitness / max(fitness[i], 1e-10))  # Update learning rate

                if evals >= self.budget:
                    break

            population, fitness = self.stochastic_ranking(population, fitness)
            adjusted_population_size = max(self.min_population_size, len(population) * (self.budget - evals) // self.budget)
            if adjusted_population_size < current_population_size:
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices[:adjusted_population_size]]
                fitness = fitness[sorted_indices[:adjusted_population_size]]

            chaos_value = self.chaotic_map(chaos_value)
            perturbation = chaos_value * (upper_bound - lower_bound) * 0.1  # Self-adaptive perturbation
            population += np.random.uniform(-1, 1, population.shape) * perturbation
            population = np.clip(population, lower_bound, upper_bound)

        return best_solution