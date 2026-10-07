import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.9
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.elitism_rate = 0.1

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
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_adaptive = np.random.normal(loc=0.5, scale=0.3, size=None)
                F_adaptive = np.clip(F_adaptive, 0, 1)
                mutant = np.clip(a + F_adaptive * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
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
            
            # Elitism: Preserve the best individuals
            elite_count = max(1, int(self.elitism_rate * self.population_size))
            elite_idxs = np.argsort(fitness)[:elite_count]
            elite_pop = pop[elite_idxs]
            elite_fitness = fitness[elite_idxs]
            
            if evaluations >= self.budget:
                break

            # Adaptive Particle Swarm Optimization Phase
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                w_adaptive = np.random.uniform(0.4, 0.9)
                velocities[i] = (w_adaptive * velocities[i] +
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

            # Reinsert elites into the population
            for idx, elite_idx in enumerate(elite_idxs):
                pop[elite_idx] = elite_pop[idx]
                fitness[elite_idx] = elite_fitness[idx]

        return gbest