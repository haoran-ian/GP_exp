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
        
        while evals < self.budget:
            self.w = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            generation += 1
            
            # Dynamic population adjustment
            if generation % 10 == 0:
                self.population_size = min(max(5, self.population_size + np.random.randint(-2, 3)), 10 * self.dim)
                if self.population_size != len(population):
                    extras = np.random.uniform(lb, ub, (self.population_size - len(population), self.dim))
                    population = np.vstack((population, extras))
                    velocities = np.vstack((velocities, np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size - len(velocities), self.dim))))
                    fitness = np.array([func(x) for x in population])
                    pbest = population.copy()
                    pbest_fitness = fitness.copy()
                    gbest = population[np.argmin(fitness)]
                    gbest_fitness = np.min(fitness)
                    evals += self.population_size
            
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population = np.clip(population + velocities, lb, ub)
            fitness = np.array([func(x) for x in population])
            evals += self.population_size
            
            # Elite Learning Strategy: Keep a portion of the best solutions intact
            elite_size = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            non_elite_indices = np.argsort(fitness)[elite_size:]
            
            # Update personal best
            better_indices = fitness[non_elite_indices] < pbest_fitness[non_elite_indices]
            pbest[non_elite_indices[better_indices]] = population[non_elite_indices[better_indices]]
            pbest_fitness[non_elite_indices[better_indices]] = fitness[non_elite_indices[better_indices]]
            
            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
            
            # Differential Evolution with elitism
            F = self.F_base + (0.2 * np.random.randn(self.population_size))
            CR = self.CR_base + (0.1 * np.random.randn(self.population_size))
            F = np.clip(F, 0, 2)
            CR = np.clip(CR, 0, 1)
            
            for i in non_elite_indices:
                indices = np.random.choice(non_elite_indices, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + F[i] * (x1 - x2)
                cross_points = np.random.rand(self.dim) < CR[i]
                trial = np.where(cross_points, mutant, population[i])
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
            
            # Update personal best
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

        return gbest