import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.F_base = 0.5
        self.CR_base = 0.9

    def levy_flight(self, lambda_=1.5):
        sigma = (np.math.gamma(1 + lambda_) * np.sin(np.pi * lambda_ / 2) / 
                 (np.math.gamma((1 + lambda_) / 2) * lambda_ * 2**((lambda_ - 1) / 2)))** (1 / lambda_)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v)**(1 / lambda_)
        return step

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
        generation = 0

        diversity_threshold = 0.1 * np.linalg.norm(ub - lb)

        while evals < self.budget:
            self.w = self.w_max - ((self.w_max - self.w_min) * (gbest_fitness / np.mean(fitness)))
            generation += 1

            # Dynamic strategy based on diversity
            diversity = np.mean(np.linalg.norm(population - np.mean(population, axis=0), axis=1))
            if diversity < diversity_threshold:
                self.F_base = 0.5 + 0.5 * np.random.rand()
                self.CR_base = 0.1 + 0.8 * np.random.rand()

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

            F = np.random.uniform(0.5, 1.0, self.population_size)
            CR = np.random.uniform(0.1, 1.0, self.population_size)

            for i in range(self.population_size):
                if np.random.rand() < 0.2:
                    trial = population[i] + self.levy_flight()
                else:
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

        return gbest