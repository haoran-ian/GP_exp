import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Differential evolution scaling factor
        self.CR = 0.9  # Crossover rate
        self.w_init = 0.9  # Initial inertia weight for PSO
        self.w_end = 0.4  # Final inertia weight for PSO
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) * 0.1
        fitness = np.array([func(ind) for ind in pop])
        pbest = pop.copy()
        pbest_fitness = fitness.copy()
        gbest = pop[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)

        evaluations = self.population_size
        iteration = 0
        while evaluations < self.budget:
            # Update inertia weight dynamically
            w = self.w_end + (self.w_init - self.w_end) * (self.budget - evaluations) / self.budget

            # Adaptive Differential Evolution Phase
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_adaptive = self.F * (1 + np.random.uniform(-0.1, 0.1))
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

            if evaluations >= self.budget:
                break

            # Dynamic Particle Swarm Optimization Phase
            for i in range(self.population_size):
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

            iteration += 1

        return gbest