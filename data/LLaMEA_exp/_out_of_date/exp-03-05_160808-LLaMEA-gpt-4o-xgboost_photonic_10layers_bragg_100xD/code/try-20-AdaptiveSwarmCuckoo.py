import numpy as np

class AdaptiveSwarmCuckoo:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.nests = 25  # number of nests (solutions)
        self.pa = 0.3   # initial probability of abandoning a solution
        self.beta = 1.5  # Levy flight parameter
        self.lr = 0.8    # Local refinement probability
        self.pop = None
        self.fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.diversity_threshold = 0.1

    def levy_flight(self, size, scale=1.0):
        sigma = (np.math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                 (np.math.gamma((1 + self.beta) / 2) * self.beta *
                  2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, size=size) * scale
        v = np.random.normal(0, 1, size=size)
        step = u / np.abs(v) ** (1 / self.beta)
        return step

    def differential_evolution(self, target_idx, F=0.5, CR=0.8):
        idxs = [idx for idx in range(self.nests) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.pop[a] + F * (self.pop[b] - self.pop[c])
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.pop[target_idx])
        return np.clip(trial, self.lb, self.ub)

    def calculate_diversity(self):
        centroid = np.mean(self.pop, axis=0)
        diversity = np.mean(np.linalg.norm(self.pop - centroid, axis=1))
        return diversity

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.pop = np.random.uniform(self.lb, self.ub, (self.nests, self.dim))
        self.fitness = np.array([func(ind) for ind in self.pop])

        for _ in range(self.budget - self.nests):
            new_pop = self.pop.copy()
            diversity = self.calculate_diversity()
            
            if diversity < self.diversity_threshold:
                F = 0.9
                CR = 0.9
            else:
                F = 0.5
                CR = 0.8

            for i in range(self.nests):
                exploration_scale = 1 + (_ / self.budget)
                if np.random.rand() < self.lr:
                    candidate = self.differential_evolution(i, F=F, CR=CR)
                else:
                    step_size = self.levy_flight(self.dim, scale=exploration_scale) * (self.pop[i] - self.global_best)
                    candidate = self.pop[i] + step_size * np.random.uniform(-1, 1.2, self.dim)
                candidate = np.clip(candidate, self.lb, self.ub)
                f_candidate = func(candidate)
                
                if f_candidate < self.fitness[i]:
                    new_pop[i] = candidate
                    self.fitness[i] = f_candidate

                if f_candidate < self.global_best_fitness:
                    self.global_best_fitness = f_candidate
                    self.global_best = candidate

            # Dynamic pa value based on progress
            self.pa = 0.1 + 0.2 * (_ / self.budget)
            abandon = np.random.rand(self.nests) < self.pa
            for i in range(self.nests):
                if abandon[i] and _ < self.budget - self.nests:
                    new_pop[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    self.fitness[i] = func(new_pop[i])
            
            self.pop = new_pop
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[best_idx]
                self.global_best = self.pop[best_idx]

        return self.global_best