import numpy as np

class EnhancedHybridCuckooV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.nests = 25
        self.pa = 0.3
        self.beta = 1.5
        self.lr = 0.8
        self.pop = None
        self.fitness = None

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

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.pop = np.random.uniform(self.lb, self.ub, (self.nests, self.dim))
        self.fitness = np.array([func(ind) for ind in self.pop])
        best_idx = np.argmin(self.fitness)
        best_solution = self.pop[best_idx]
        fitness_rank = np.argsort(self.fitness)
        
        for _ in range(self.budget - self.nests):
            new_pop = self.pop.copy()
            for i in range(self.nests):
                exploration_scale = 1 + (_ / self.budget)
                rank_factor = 1 + (fitness_rank[i] / self.nests)
                diversity_factor = np.std(self.fitness) / (np.mean(self.fitness) + 1e-9)
                self.lr = 0.5 + 0.5 * (1 - np.exp(-diversity_factor))  # More dynamic lr value
                if np.random.rand() < self.lr:
                    candidate = self.differential_evolution(i, F=np.random.rand(), CR=np.random.rand())
                else:
                    step_size = self.levy_flight(self.dim, scale=exploration_scale * rank_factor * (1 - self.fitness[i] / (np.max(self.fitness) + 1e-9)))
                    bias = 0.5 * (best_solution - self.pop[i])
                    candidate = self.pop[i] + step_size * np.random.uniform(-1, 1.2, self.dim) + bias
                candidate = np.clip(candidate, self.lb, self.ub)
                f_candidate = func(candidate)
                
                if f_candidate < self.fitness[i]:
                    new_pop[i] = candidate
                    self.fitness[i] = f_candidate

            self.pa = 0.1 + 0.2 * (1 - diversity_factor)
            abandon = np.random.rand(self.nests) < self.pa
            for i in range(self.nests):
                if abandon[i] and _ < self.budget - self.nests:
                    new_pop[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    self.fitness[i] = func(new_pop[i])
            
            self.pop = new_pop
            best_idx = np.argmin(self.fitness)
            best_solution = self.pop[best_idx]
            fitness_rank = np.argsort(self.fitness)

        return best_solution