import numpy as np

class AdaptiveHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_min, self.F_max = 0.2, 0.8  # Adaptive DE scaling factor range
        self.CR_min, self.CR_max = 0.6, 0.9  # Adaptive crossover rate range
        self.w_min, self.w_max = 0.3, 0.7  # Adaptive inertia weight range
        self.c1_min, self.c1_max = 1.0, 2.0  # Adaptive cognitive component range
        self.c2_min, self.c2_max = 1.0, 2.0  # Adaptive social component range

    def adaptive_parameters(self, progress):
        """Adjust parameters based on the optimization progress."""
        F = self.F_min + (self.F_max - self.F_min) * progress
        CR = self.CR_max - (self.CR_max - self.CR_min) * progress
        w = self.w_max - (self.w_max - self.w_min) * progress
        c1 = self.c1_min + (self.c1_max - self.c1_min) * progress
        c2 = self.c2_max - (self.c2_max - self.c2_min) * progress
        return F, CR, w, c1, c2

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
            progress = evaluations / self.budget
            F, CR, w, c1, c2 = self.adaptive_parameters(progress)

            # Differential Evolution Phase
            for i in range(self.population_size):
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

            # Particle Swarm Optimization Phase
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (pbest[i] - pop[i]) +
                                 c2 * r2 * (gbest - pop[i]))
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