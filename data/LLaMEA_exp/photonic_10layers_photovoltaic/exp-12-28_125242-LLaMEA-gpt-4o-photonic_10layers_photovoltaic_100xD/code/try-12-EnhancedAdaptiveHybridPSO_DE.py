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
        self.elite_rate = 0.1  # Proportion of elite individuals

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
            # Dynamic inertia weight inspired by success rate
            success_rate = np.mean(pbest_fitness < gbest_fitness)
            w = self.w_max - (self.w_max - self.w_min) * success_rate

            # Update DE parameters adaptively
            F = self.F0 + 0.1 * np.random.randn()
            CR = self.CR0 + 0.05 * np.random.randn()
            F = np.clip(F, 0, 1)
            CR = np.clip(CR, 0, 1)

            # PSO Update with stochastic learning
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = w * velocities + self.c1 * r1 * (pbest - pop) + self.c2 * r2 * (gbest - pop)
            pop = np.clip(pop + velocities, lb, ub)

            # Evaluate population
            for i in range(self.population_size):
                fitness = func(pop[i])
                evaluations += 1

                # Update personal best
                if fitness < pbest_fitness[i]:
                    pbest[i] = pop[i]
                    pbest_fitness[i] = fitness

                    # Update global best
                    if fitness < gbest_fitness:
                        gbest = pop[i]
                        gbest_fitness = fitness

                if evaluations >= self.budget:
                    break

            # Elite mutation strategy
            elite_size = int(self.elite_rate * self.population_size)
            sorted_indices = np.argsort(pbest_fitness)[:elite_size]
            elite_pop = pbest[sorted_indices]

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                elite_idx = np.random.choice(elite_size)
                mutant = np.clip(elite_pop[elite_idx] + F * (pop[np.random.choice(idxs)] - pop[np.random.choice(idxs)]), lb, ub)
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