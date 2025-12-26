import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(5, 2 * dim)
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.F_base = 0.5  # Base mutation factor for DE
        self.CR_base = 0.9  # Base crossover probability for DE
        self.local_search_prob = 0.1  # Probability threshold for local search

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        pbest = population.copy()
        pbest_fitness = fitness.copy()
        gbest = population[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)
        evals = population_size
        generation = 0

        while evals < self.budget:
            # Adaptive inertia weight
            self.w = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            generation += 1
            
            # Particle Swarm Optimization Step
            r1, r2 = np.random.rand(population_size, self.dim), np.random.rand(population_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population = np.clip(population + velocities, lb, ub)
            fitness = np.array([func(x) for x in population])
            evals += population_size
            
            # Update personal and global bests
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]
            
            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
            
            # Self-adaptive differential evolution parameters
            F = self.F_base + (0.2 * np.random.randn(population_size))
            CR = self.CR_base + (0.1 * np.random.randn(population_size))
            F = np.clip(F, 0, 2)
            CR = np.clip(CR, 0, 1)

            # Differential Evolution Step
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + F[i] * (x1 - x2)
                cross_points = np.random.rand(self.dim) < CR[i]
                trial = np.where(cross_points, mutant, population[i])
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness

            # Update personal and global bests again after DE step
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
            
            # Local Search Enhancement
            if np.random.rand() < self.local_search_prob:
                local_candidate = gbest + np.random.randn(self.dim) * 0.1 * (ub - lb)
                local_candidate = np.clip(local_candidate, lb, ub)
                local_fitness = func(local_candidate)
                evals += 1
                if local_fitness < gbest_fitness:
                    gbest, gbest_fitness = local_candidate, local_fitness

            # Dynamic Population Resizing
            population_size = max(5, int(self.initial_population_size * (1 - evals / self.budget)))

        return gbest