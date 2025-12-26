import numpy as np

class EnhancedAdaptiveBBO_DEM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_mutation_factor = 0.8
        self.initial_crossover_rate = 0.7
        self.elite_rate = 0.2
        self.island_size = int(self.population_size * self.elite_rate)
        self.num_islands = self.population_size // self.island_size
        self.global_best = None
        self.global_best_fitness = float('inf')

    def mutation(self, target_idx, population):
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

            if fitness[best_idx] < self.global_best_fitness:
                self.global_best = best_solution
                self.global_best_fitness = fitness[best_idx]

            for i in range(self.population_size):
                self.crossover_rate = self.initial_crossover_rate * (1 - i / self.population_size)
                self.mutation_factor = self.initial_mutation_factor * (1 + i / self.population_size)

                mutant = self.mutation(i, population)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                budget_spent += 1

                if trial_fitness < fitness[i]:
                    next_population[i] = trial
                    fitness[i] = trial_fitness

                if budget_spent >= self.budget:
                    break

            # Rank-based Biogeography-Based Optimization (BBO)
            if budget_spent < self.budget:
                sorted_indices = np.argsort(fitness)
                for idx in sorted_indices:
                    if np.random.rand() < 0.5:
                        rank = np.where(sorted_indices == idx)[0][0]
                        migration_strength = np.random.normal(0, 0.01 * (1 - rank / self.population_size))
                        emigrant = population[sorted_indices[0]]
                        population[idx] = emigrant + migration_strength
                        population[idx] = np.clip(population[idx], self.bounds.lb, self.bounds.ub)
                        fitness[idx] = func(population[idx])
                        budget_spent += 1

                        if budget_spent >= self.budget:
                            break

            population = next_population

        return population[np.argmin(fitness)]