import numpy as np

class EnhancedAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // dim)
        self.c1_base = 1.7  # Base cognitive component
        self.c2_base = 1.3  # Base social component
        self.w_max = 0.9
        self.w_min = 0.4
        self.F0 = 0.5
        self.CR0 = 0.9
        self.elite_rate = 0.1
        self.diversity_factor = 0.15
        self.archive = []  # Archive for storing unsuccessful solutions

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
            success_rate = np.mean(pbest_fitness < gbest_fitness)
            self.c1 = self.c1_base * (1 + success_rate)  # Dynamic adjustment of c1
            self.c2 = self.c2_base * (1 - success_rate)  # Dynamic adjustment of c2
            w = self.w_max - (self.w_max - self.w_min) * success_rate

            F = self.F0 + 0.1 * np.random.randn()
            CR = self.CR0 + 0.05 * np.random.randn()
            F = np.clip(F, 0.3, 0.9)
            CR = np.clip(CR, 0, 1)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = w * velocities + self.c1 * r1 * (pbest - pop) + self.c2 * r2 * (gbest - pop)

            self.diversity_factor = 0.1 + 0.2 * (1 - np.tanh(success_rate))
            random_noise = self.diversity_factor * np.random.uniform(-1, 1, (self.population_size, self.dim))
            pop = np.clip(pop + velocities + random_noise, lb, ub)

            for i in range(self.population_size):
                fitness = func(pop[i])
                evaluations += 1

                if fitness < pbest_fitness[i]:
                    pbest[i] = pop[i]
                    pbest_fitness[i] = fitness

                    if fitness < gbest_fitness:
                        gbest = pop[i]
                        gbest_fitness = fitness
                else:
                    self.archive.append(pop[i])  # Store unsuccessful solution in archive

                if evaluations >= self.budget:
                    break

            # Archive-based mutation
            if len(self.archive) > 0:
                for i in range(self.population_size):
                    if np.random.rand() < 0.1:
                        archive_idx = np.random.randint(len(self.archive))
                        mutant = np.clip(self.archive[archive_idx] + F * (pop[np.random.choice(self.population_size)] - pop[np.random.choice(self.population_size)]), lb, ub)
                        trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
                        trial_fitness = func(trial)
                        evaluations += 1

                        if trial_fitness < pbest_fitness[i]:
                            pbest[i] = trial
                            pbest_fitness[i] = trial_fitness
                            if trial_fitness < gbest_fitness:
                                gbest = trial
                                gbest_fitness = trial_fitness

                        if evaluations >= self.budget:
                            break

            if evaluations < self.budget and np.random.rand() < 0.05:
                pop[np.random.randint(self.population_size)] = np.random.uniform(lb, ub, self.dim)

        return gbest