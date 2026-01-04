import numpy as np

class EnhancedAdaptiveHybridPSO_DE_MultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3  # Introducing multiple swarms
        self.population_size = min(100, budget // (dim * self.num_swarms))
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
        swarms = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        pbest = [swarm.copy() for swarm in swarms]
        pbest_fitness = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        gbest = [swarm[np.argmin(pf)] for swarm, pf in zip(swarms, pbest_fitness)]
        gbest_fitness = [np.min(pf) for pf in pbest_fitness]

        evaluations = self.population_size * self.num_swarms
        while evaluations < self.budget:
            for s in range(self.num_swarms):
                success_rate = np.mean(pbest_fitness[s] < gbest_fitness[s])
                w = self.w_max - (self.w_max - self.w_min) * success_rate

                F = self.F0 + 0.1 * np.random.randn()
                CR = self.CR0 + 0.05 * np.random.randn()
                F = np.clip(F, 0.3, 0.9)
                CR = np.clip(CR, 0, 1)

                r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
                velocities[s] = w * velocities[s] + self.c1 * r1 * (pbest[s] - swarms[s]) + self.c2 * r2 * (gbest[s] - swarms[s])

                self.diversity_factor = 0.1 + 0.2 * (1 - np.tanh(success_rate))
                random_noise = self.diversity_factor * np.random.uniform(-1, 1, (self.population_size, self.dim))
                swarms[s] = np.clip(swarms[s] + velocities[s] + random_noise, lb, ub)

                for i in range(self.population_size):
                    fitness = func(swarms[s][i])
                    evaluations += 1

                    if fitness < pbest_fitness[s][i]:
                        pbest[s][i] = swarms[s][i]
                        pbest_fitness[s][i] = fitness

                        if fitness < gbest_fitness[s]:
                            gbest[s] = swarms[s][i]
                            gbest_fitness[s] = fitness

                    if evaluations >= self.budget:
                        break

            # Inter-swarm elitism
            global_best_idx = np.argmin(gbest_fitness)
            for s in range(self.num_swarms):
                if s != global_best_idx:
                    elite_individual = gbest[global_best_idx]
                    swarms[s][np.random.randint(self.population_size)] = elite_individual

        return gbest[np.argmin(gbest_fitness)]