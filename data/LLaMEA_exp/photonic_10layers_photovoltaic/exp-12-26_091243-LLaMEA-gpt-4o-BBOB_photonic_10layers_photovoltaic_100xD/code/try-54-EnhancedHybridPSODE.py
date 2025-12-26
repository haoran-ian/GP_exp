import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.F_base = 0.5  # Base mutation factor for DE
        self.CR_base = 0.9  # Base crossover probability for DE
        self.archive_size = self.population_size // 2  # Size of the elitist archive
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        pbest = population.copy()
        pbest_fitness = fitness.copy()
        gbest = population[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)
        evals = self.population_size
        archive = []

        while evals < self.budget:
            # Enhance inertia weight calculation
            self.w = self.w_max - ((self.w_max - self.w_min) * (np.var(fitness) / np.mean(fitness)))
            
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population = np.clip(population + velocities, lb, ub)
            fitness = np.array([func(x) for x in population])
            evals += self.population_size
            
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]
            
            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
            
            # Dynamic DE parameters
            F = self.F_base + (0.4 * np.random.randn(self.population_size))
            CR = self.CR_base + (0.2 * np.random.randn(self.population_size))
            F = np.clip(F, 0, 2)
            CR = np.clip(CR, 0, 1)
            
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + F[i] * (x1 - x2)
                cross_points = np.random.rand(self.dim) < CR[i]
                trial = np.where(cross_points, mutant, population[i])
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
            
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            # Update the archive with non-dominated solutions
            combined_pop = np.vstack((population, archive))
            combined_fit = np.hstack((fitness, [func(x) for x in archive]))
            non_dominated_indices = self._non_dominated_sorting(combined_pop, combined_fit)
            archive = combined_pop[non_dominated_indices]
            if len(archive) > self.archive_size:
                archive = self._crowding_distance_sort(archive, combined_fit[non_dominated_indices])[:self.archive_size]

        return gbest
    
    def _non_dominated_sorting(self, population, fitness):
        dominated = set()
        non_dominated = set(range(len(population)))
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if fitness[i] < fitness[j]:
                    dominated.add(j)
                elif fitness[j] < fitness[i]:
                    dominated.add(i)
        non_dominated -= dominated
        return list(non_dominated)
    
    def _crowding_distance_sort(self, archive, fitness):
        if len(archive) < 2:
            return np.arange(len(archive))
        distances = np.zeros(len(archive))
        sorted_indices = np.argsort(fitness)
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
        for i in range(1, len(archive) - 1):
            distances[sorted_indices[i]] = (fitness[sorted_indices[i + 1]] - fitness[sorted_indices[i - 1]]) / (np.max(fitness) - np.min(fitness))
        return np.argsort(-distances)