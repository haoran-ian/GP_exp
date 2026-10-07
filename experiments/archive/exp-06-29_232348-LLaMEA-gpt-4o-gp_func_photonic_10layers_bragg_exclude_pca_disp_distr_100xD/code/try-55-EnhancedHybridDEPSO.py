import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Initial DE scaling factor
        self.CR = 0.9  # Initial crossover rate
        self.w = 0.5  # Initial inertia weight for PSO
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
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_adaptive = 0.5 + np.random.rand() * 0.5  # Adaptive scaling factor
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

            # Adaptive PSO Phase with Crowding Distance
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                w_adaptive = 0.5 + 0.2 * (evaluations / self.budget)  # Decreasing inertia
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

            # Crowding Distance-based Diversity Maintenance
            crowding_distances = np.zeros(self.population_size)
            sorted_indices = np.argsort(fitness)
            crowding_distances[sorted_indices[0]] = crowding_distances[sorted_indices[-1]] = float('inf')
            for j in range(1, self.population_size - 1):
                crowding_distances[sorted_indices[j]] = (
                    crowding_distances[sorted_indices[j - 1]] + crowding_distances[sorted_indices[j + 1]]
                ) / 2
            
            # Enhance diversity by retaining only diverse solutions
            diverse_indices = np.argsort(-crowding_distances)[:self.population_size]
            pop = pop[diverse_indices]
            fitness = fitness[diverse_indices]
            velocities = velocities[diverse_indices]
            pbest = pbest[diverse_indices]
            pbest_fitness = pbest_fitness[diverse_indices]

        return gbest