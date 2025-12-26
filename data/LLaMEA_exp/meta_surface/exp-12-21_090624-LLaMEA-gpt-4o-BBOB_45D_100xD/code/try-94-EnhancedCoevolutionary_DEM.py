import numpy as np

class EnhancedCoevolutionary_DEM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elite_rate = 0.2
        self.island_size = int(self.population_size * self.elite_rate)
        self.num_islands = self.population_size // self.island_size
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.stagnation_threshold = 10
        self.stagnation_counter = 0

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

    def cooperative_coevolution(self, population, fitness):
        # Dynamically adjust subcomponent sizes based on stagnation
        subcomponent_size = max(1, self.dim // max(1, self.num_islands - self.stagnation_counter))
        new_population = np.copy(population)
        for idx in range(self.population_size):
            if idx % subcomponent_size == 0:
                subcomponent = population[idx:idx+subcomponent_size]
                local_best_idx = np.argmin(fitness[idx:idx+subcomponent_size])
                best_solution = subcomponent[local_best_idx]
                for i in range(subcomponent_size):
                    mutant = self.mutation(idx + i, population)
                    trial = self.crossover(population[idx + i], mutant)
                    trial_fitness = func(trial)
                    if trial_fitness < fitness[idx + i]:
                        new_population[idx + i] = trial
                        fitness[idx + i] = trial_fitness
                        if trial_fitness < self.global_best_fitness:
                            self.global_best = trial
                            self.global_best_fitness = trial_fitness
        return new_population, fitness

    def __call__(self, func):
        self.bounds = func.bounds
        budget_spent = 0

        population = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_spent += self.population_size

        while budget_spent < self.budget:
            prev_best_fitness = self.global_best_fitness
            population, fitness = self.cooperative_coevolution(population, fitness)
            
            if self.global_best_fitness >= prev_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter > self.stagnation_threshold:
                self.mutation_factor *= 1.1  # Increase diversity
                self.stagnation_counter = 0

        return population[np.argmin(fitness)]