import numpy as np

class ImprovedEnhancedHybridPSODE:
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
            self.w = self.w_max - ((self.w_max - self.w_min) * (np.var(fitness) / np.mean(fitness)))
            generation += 1
            
            # Elitist selection for dynamic population adjustment
            if generation % 10 == 0:
                elite_size = max(1, self.population_size // 5)
                elite_indices = np.argsort(fitness)[:elite_size]
                population = population[elite_indices]
                velocities = velocities[elite_indices]
                fitness = fitness[elite_indices]
                pbest = pbest[elite_indices]
                pbest_fitness = pbest_fitness[elite_indices]
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
                
                self.population_size = min(max(5, self.population_size + np.random.randint(-2, 3)), 10 * dim)
                new_pop = np.random.uniform(lb, ub, (self.population_size - elite_size, self.dim))
                population = np.vstack((population, new_pop))
                velocities = np.vstack((velocities, np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size - elite_size, self.dim))))
                fitness = np.array([func(x) for x in population])
                pbest = population.copy()
                pbest_fitness = fitness.copy()
                evals += self.population_size
            
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            velocities = np.clip(velocities, -abs(ub - lb), abs(ub - lb))
            population = np.clip(population + velocities, lb, ub)
            fitness = np.array([func(x) for x in population])
            evals += self.population_size
            
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]
            
            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
            
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

        return gbest