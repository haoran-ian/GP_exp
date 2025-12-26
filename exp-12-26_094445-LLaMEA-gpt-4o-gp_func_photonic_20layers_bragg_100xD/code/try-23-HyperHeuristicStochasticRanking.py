import numpy as np

class HyperHeuristicStochasticRanking:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.99
        self.temperature_threshold = 0.1
        self.mutation_strategies = [self.differential_mutation, self.gaussian_mutation]
        self.strategy_weights = np.ones(len(self.mutation_strategies))

    def adaptive_cooling(self, current_fitness, best_fitness):
        return max(self.cooling_rate, (best_fitness - current_fitness) / max(best_fitness, 1e-10))

    def stochastic_ranking(self, population, fitness):
        perm = np.random.permutation(self.population_size)
        ranked_population = population[perm]
        ranked_fitness = fitness[perm]
        for i in range(self.population_size - 1):
            if np.random.rand() < 0.45 or ranked_fitness[i] < ranked_fitness[i+1]:
                ranked_population[i], ranked_population[i+1] = ranked_population[i+1], ranked_population[i]
                ranked_fitness[i], ranked_fitness[i+1] = ranked_fitness[i+1], ranked_fitness[i]
        return ranked_population, ranked_fitness

    def differential_mutation(self, a, b, c, lower_bound, upper_bound):
        F = 0.5 + 0.4 * np.random.rand()  # Adaptive mutation scaling factor
        return np.clip(a + F * (b - c), lower_bound, upper_bound)

    def gaussian_mutation(self, individual, lower_bound, upper_bound):
        mutation_strength = 0.1 * (upper_bound - lower_bound)
        return np.clip(individual + np.random.normal(0, mutation_strength, self.dim), lower_bound, upper_bound)

    def select_mutation_strategy(self):
        probabilities = self.strategy_weights / np.sum(self.strategy_weights)
        return np.random.choice(len(self.mutation_strategies), p=probabilities)

    def update_strategy_weights(self, applied_strategy, success):
        if success:
            self.strategy_weights[applied_strategy] *= 1.1
        else:
            self.strategy_weights[applied_strategy] *= 0.9

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
                # Select and apply a mutation strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                strategy_idx = self.select_mutation_strategy()
                if strategy_idx == 0:
                    mutant = self.differential_mutation(a, b, c, lower_bound, upper_bound)
                else:
                    mutant = self.gaussian_mutation(population[i], lower_bound, upper_bound)

                # Crossover operation
                cross_points = np.random.rand(self.dim) < 0.9
                offspring = np.where(cross_points, mutant, population[i])
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                evals += 1
                
                # Replacement strategy
                success = False
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    success = True
                    
                    # Simulated annealing acceptance criterion
                    if offspring_fitness < best_fitness or np.exp((best_fitness - offspring_fitness) / temperature) > np.random.rand():
                        best_solution = offspring
                        best_fitness = offspring_fitness
                
                # Update strategy weights
                self.update_strategy_weights(strategy_idx, success)

                # Update temperature with adaptive cooling
                temperature *= self.adaptive_cooling(offspring_fitness, best_fitness)
                if temperature < self.temperature_threshold:
                    temperature = self.temperature_threshold

                if evals >= self.budget:
                    break

            # Apply stochastic ranking to enhance exploration and diversity
            population, fitness = self.stochastic_ranking(population, fitness)

        return best_solution