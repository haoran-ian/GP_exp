import numpy as np

class AdaptiveBBO_DynamicIsland:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elite_rate = 0.2
        self.island_size = int(self.population_size * self.elite_rate)
        self.num_islands = self.population_size // self.island_size
        self.dynamic_crossover_decay = 0.99
        self.dynamic_migration_probability = 0.5

    def mutation(self, target_idx, population, best_solution):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        self.bounds = func.bounds
        budget_spent = 0

        # Initialize population
        population = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_spent += self.population_size

        while budget_spent < self.budget:
            next_population = np.copy(population)
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            for i in range(self.population_size):
                mutant = self.mutation(i, population, best_solution)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    next_population[i] = trial
                    fitness[i] = trial_fitness

                if budget_spent >= self.budget:
                    break

            # Adaptive Biogeography-Based Optimization (BBO)
            if budget_spent < self.budget:
                sorted_indices = np.argsort(fitness)
                elites = sorted_indices[:self.island_size]
                for island in range(self.num_islands):
                    island_indices = sorted_indices[island*self.island_size:(island+1)*self.island_size]
                    best_island = population[elites[island]]
                    for idx in island_indices:
                        if np.random.rand() < self.dynamic_migration_probability:
                            population[idx] = best_island + np.random.normal(0, 0.01, self.dim)
                            population[idx] = np.clip(population[idx], self.bounds.lb, self.bounds.ub)
                            fitness[idx] = func(population[idx])
                            budget_spent += 1

                            if budget_spent >= self.budget:
                                break

            # Update crossover rate and migration probability dynamically
            self.crossover_rate *= self.dynamic_crossover_decay
            self.dynamic_migration_probability *= self.dynamic_crossover_decay
            
            # Incrementally decrease the mutation factor
            self.mutation_factor *= 0.995

            population = next_population

        return population[np.argmin(fitness)]