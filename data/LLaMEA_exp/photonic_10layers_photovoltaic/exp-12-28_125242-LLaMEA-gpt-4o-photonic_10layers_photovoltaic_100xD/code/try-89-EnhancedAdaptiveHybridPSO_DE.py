import numpy as np

class EnhancedAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // dim)
        self.c1 = 1.7
        self.c2 = 1.3
        self.w_max = 0.9
        self.w_min = 0.4
        self.F0 = 0.5
        self.CR0 = 0.9
        self.elite_rate = 0.1
        self.diversity_factor = 0.15

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        pbest = pop.copy()
        pbest_fitness = np.array([func(ind) for ind in pop])
        gbest = pop[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        evaluations = self.population_size
        chaos_param = 0.7  # Initial chaotic parameter
        while evaluations < self.budget:
            success_rate = np.mean(pbest_fitness < gbest_fitness)
            w = self.w_max - (self.w_max - self.w_min) * success_rate

            F = self.F0 + 0.1 * np.random.randn() * (1 - 0.5 * success_rate)  # Adaptive mutation scaling
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

                if evaluations >= self.budget:
                    break

            elite_size = int(self.elite_rate * (1 + 0.5 * success_rate) * self.population_size)
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

            if evaluations < self.budget and np.random.rand() < chaos_param:
                pop[np.random.randint(self.population_size)] = np.random.uniform(lb, ub, self.dim) * (1 - success_rate)  # Enhanced chaos initialization
                chaos_param *= 0.99  # Gradually reduce chaos influence

        return gbest