import numpy as np

class HybridDEPSOImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_min, self.F_max = 0.4, 0.9  # Adaptively varying DE scaling factor
        self.CR_min, self.CR_max = 0.5, 0.9  # Adaptively varying crossover rate
        self.w_min, self.w_max = 0.4, 0.9  # Adaptively varying inertia weight for PSO
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        pbest = pop.copy()
        pbest_fitness = fitness.copy()
        gbest = pop[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)

        evaluations = self.population_size
        while evaluations < self.budget:
            # Adaptive Differential Evolution Phase
            for i in range(self.population_size):
                F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
                CR = self.CR_min + (self.CR_max - self.CR_min) * np.random.rand()
                
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < pbest_fitness[i]:
                        pbest[i] = trial
                        pbest_fitness[i] = trial_fitness
                        if trial_fitness < gbest_fitness:
                            gbest = trial
                            gbest_fitness = trial_fitness
            
            if evaluations >= self.budget:
                break

            # Adaptive Particle Swarm Optimization Phase
            for i in range(self.population_size):
                w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
                
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (pbest[i] - pop[i]) +
                                 self.c2 * r2 * (gbest - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], lb, ub)
                particle_fitness = func(pop[i])
                evaluations += 1
                if particle_fitness < fitness[i]:
                    fitness[i] = particle_fitness
                    if particle_fitness < pbest_fitness[i]:
                        pbest[i] = pop[i]
                        pbest_fitness[i] = particle_fitness
                        if particle_fitness < gbest_fitness:
                            gbest = pop[i]
                            gbest_fitness = particle_fitness

        return gbest