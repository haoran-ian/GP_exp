import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.F_min, self.F_max = 0.4, 0.9  # Adaptive DE scaling factor range
        self.CR_min, self.CR_max = 0.6, 0.9  # Adaptive crossover rate range
        self.w_min, self.w_max = 0.4, 0.9  # Adaptive inertia weight range
        self.c1, self.c2 = 1.5, 1.5  # Cognitive and social components

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        pbest = pop.copy()
        pbest_fitness = fitness.copy()
        gbest = pop[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)

        evaluations = population_size
        while evaluations < self.budget:
            # Dynamic parameter adjustment
            F = self.F_min + (self.F_max - self.F_min) * evaluations / self.budget
            CR = self.CR_max - (self.CR_max - self.CR_min) * evaluations / self.budget
            w = self.w_max - (self.w_max - self.w_min) * evaluations / self.budget

            # Differential Evolution Phase
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
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
            for i in range(population_size):
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

            # Dynamic population size adjustment
            if evaluations > self.budget * 0.5:
                population_size = int(self.initial_population_size * (self.budget - evaluations) / (self.budget * 0.5))
                if population_size < 10: 
                    population_size = 10
                pop = pop[:population_size]
                velocities = velocities[:population_size]
                fitness = fitness[:population_size]
                pbest = pbest[:population_size]
                pbest_fitness = pbest_fitness[:population_size]

        return gbest