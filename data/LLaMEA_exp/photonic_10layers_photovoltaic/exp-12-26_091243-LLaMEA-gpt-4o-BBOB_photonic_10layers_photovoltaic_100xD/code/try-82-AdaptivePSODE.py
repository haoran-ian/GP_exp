import numpy as np

class AdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.c1_init, self.c2_init = 1.5, 1.5
        self.c1, self.c2 = self.c1_init, self.c2_init
        self.w_max, self.w_min = 0.9, 0.4
        self.F_base, self.CR_base = 0.5, 0.9
        
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
        prev_gbest_fitness = gbest_fitness

        while evals < self.budget:
            generation += 1
            
            # Adaptive inertia weight
            self.w = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            
            # Adaptive cognitive and social parameters
            if gbest_fitness < prev_gbest_fitness:
                self.c1 *= 0.9
                self.c2 *= 1.1
            else:
                self.c1 *= 1.1
                self.c2 *= 0.9
                
            prev_gbest_fitness = gbest_fitness
            self.c1, self.c2 = np.clip(self.c1, 0.5, 2.5), np.clip(self.c2, 0.5, 2.5)
            
            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population = np.clip(population + velocities, lb, ub)
            fitness = np.array([func(x) for x in population])
            evals += self.population_size
            
            # Update personal and global best
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]
            
            # Update global best
            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)
            
            # Dynamic DE parameters
            F = self.F_base + (0.4 * np.random.randn(self.population_size))
            CR = self.CR_base + (0.2 * np.random.randn(self.population_size))
            F, CR = np.clip(F, 0, 2), np.clip(CR, 0, 1)
            
            # Differential Evolution Mutation and Crossover
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
                # Selection
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
            
            # Update personal and global best again after DE
            better_indices = fitness < pbest_fitness
            pbest[better_indices] = population[better_indices]
            pbest_fitness[better_indices] = fitness[better_indices]
            
            if np.min(fitness) < gbest_fitness:
                gbest = population[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

        return gbest