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
        self.num_swarms = 3  # Number of swarms
        
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
            # Adaptive inertia weight calculation
            self.w = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            generation += 1
            
            # Create swarms and assign particles
            swarm_size = self.population_size // self.num_swarms
            swarms = [population[i*swarm_size:(i+1)*swarm_size] for i in range(self.num_swarms)]
            swarm_velocities = [velocities[i*swarm_size:(i+1)*swarm_size] for i in range(self.num_swarms)]
            swarm_pbest = [pbest[i*swarm_size:(i+1)*swarm_size] for i in range(self.num_swarms)]
            swarm_pbest_fitness = [pbest_fitness[i*swarm_size:(i+1)*swarm_size] for i in range(self.num_swarms)]
            
            # Update each swarm
            for s in range(self.num_swarms):
                r1, r2 = np.random.rand(swarm_size, self.dim), np.random.rand(swarm_size, self.dim)
                swarm_velocities[s] = self.w * swarm_velocities[s] + self.c1 * r1 * (swarm_pbest[s] - swarms[s]) + self.c2 * r2 * (gbest - swarms[s])
                swarms[s] = np.clip(swarms[s] + swarm_velocities[s], lb, ub)
                swarm_fitness = np.array([func(x) for x in swarms[s]])
                evals += swarm_size
                
                better_indices = swarm_fitness < swarm_pbest_fitness[s]
                swarm_pbest[s][better_indices] = swarms[s][better_indices]
                swarm_pbest_fitness[s][better_indices] = swarm_fitness[better_indices]
                
                if np.min(swarm_fitness) < gbest_fitness:
                    gbest = swarms[s][np.argmin(swarm_fitness)]
                    gbest_fitness = np.min(swarm_fitness)
            
            # Recombination across swarms using DE
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