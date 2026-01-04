import numpy as np

class EnhancedAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // dim)
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.F0 = 0.5     # Initial DE mutation factor
        self.CR0 = 0.9    # Initial DE crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        pbest = pop.copy()
        pbest_fitness = np.array([func(ind) for ind in pop])
        gbest = pop[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        evaluations = self.population_size
        while evaluations < self.budget:
            # Adaptive inertia weight based on non-linear scheduling
            w = self.w_max - (self.w_max - self.w_min) * ((evaluations / self.budget) ** 2)

            # Update DE parameters with more sophisticated adaptation
            F = self.F0 + 0.1 * np.random.randn() * (gbest_fitness / np.mean(pbest_fitness))
            CR = self.CR0 + 0.05 * np.random.randn()
            F = np.clip(F, 0, 1)
            CR = np.clip(CR, 0, 1)

            # PSO Update with randomization in cognitive and social components
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = w * velocities + self.c1 * r1 * (pbest - pop) + self.c2 * r2 * (gbest - pop)
            pop = np.clip(pop + velocities, lb, ub)

            # DE Mutation and Crossover with probabilistic selection
            for i in range(self.population_size):
                if np.random.rand() < 0.5:  # Randomly decide between PSO and DE
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), lb, ub)
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pop[i])

                    # Evaluate trial
                    trial_fitness = func(trial)
                    evaluations += 1

                    # Selection
                    if trial_fitness < pbest_fitness[i]:
                        pbest[i] = trial
                        pbest_fitness[i] = trial_fitness
                        if trial_fitness < gbest_fitness:
                            gbest = trial
                            gbest_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            # Multi-phase exploitation-exploration strategy
            if evaluations % (self.budget // 10) == 0:  # Periodic strategic refocus
                if np.random.rand() < 0.2:  # Occasionally trigger a global exploration
                    for i in range(self.population_size):
                        idxs = [idx for idx in range(self.population_size) if idx != i]
                        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                        mutant = np.clip(a + F * (b - c) * np.random.uniform(0.5, 1.5), lb, ub)
                        cross_points = np.random.rand(self.dim) < CR
                        if not np.any(cross_points):
                            cross_points[np.random.randint(0, self.dim)] = True
                        trial = np.where(cross_points, mutant, pop[i])

                        # Evaluate trial
                        trial_fitness = func(trial)
                        evaluations += 1

                        # Selection
                        if trial_fitness < pbest_fitness[i]:
                            pbest[i] = trial
                            pbest_fitness[i] = trial_fitness
                            if trial_fitness < gbest_fitness:
                                gbest = trial
                                gbest_fitness = trial_fitness

                        if evaluations >= self.budget:
                            break

        return gbest