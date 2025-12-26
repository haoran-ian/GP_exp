import numpy as np

class AdaptiveHybridPSODE:
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
        
        # Dynamic neighborhood topology: ring topology
        neighbors = np.roll(np.arange(self.population_size), 1)

        while evals < self.budget:
            # Adaptive inertia weight
            self.w = self.w_min + (self.w_max - self.w_min) * (1 - (gbest_fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-10))
            
            # PSO velocity and position update with neighborhood topology
            for i in range(self.population_size):
                local_gbest = population[neighbors[i]]
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (pbest[i] - population[i])
                                 + self.c2 * r2 * (local_gbest - population[i]))
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

            # Evaluate new solutions
            fitness = np.array([func(x) for x in population])
            evals += self.population_size

            # Update personal and global bests
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            # Self-adaptive DE parameters
            F = np.radians(self.F_base + 0.2 * (1 - (gbest_fitness - pbest_fitness) / (np.max(pbest_fitness) - np.min(pbest_fitness) + 1e-10)))
            CR = np.tanh(self.CR_base + 0.1 * np.random.randn(self.population_size))

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

            # Update personal and global bests again
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

        return gbest