import numpy as np

class EnhancedHybridCuckooV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.nests = 25  # number of nests (solutions)
        self.pa = 0.3   # probability of abandoning a solution
        self.beta = 1.5  # Levy flight parameter
        self.lr = 0.8    # Local refinement probability
        self.pop = None
        self.fitness = None
        self.best_solution = None
        self.memory = []

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
        self.best_solution = self.pop[best_idx]
        self.memory.append(self.best_solution)
        fitness_rank = np.argsort(self.fitness)
        
        for _ in range(self.budget - self.nests):
            new_pop = self.pop.copy()
            for i in range(self.nests):
                exploration_scale = 1 + (_ / self.budget)
                rank_factor = 1 + (fitness_rank[i] / self.nests)
                diversity_factor = np.std(self.fitness) / (np.mean(self.fitness) + 1e-9)
                self.lr = 0.6 + 0.4 * diversity_factor  # Dynamic lr value proportional to diversity
                learning_rate = 0.1 + 0.9 * (self.budget - _) / self.budget  # adaptive learning rate
                if np.random.rand() < self.lr:
                    candidate = self.differential_evolution(i, F=np.random.rand(), CR=np.random.rand())
                else:
                    step_size = self.levy_flight(self.dim, scale=exploration_scale * rank_factor) * (self.pop[i] - self.best_solution)
                    memory_bias = np.mean(self.memory, axis=0) - self.pop[i]
                    candidate = self.pop[i] + learning_rate * (step_size + memory_bias)
                candidate = np.clip(candidate, self.lb, self.ub)
                f_candidate = func(candidate)
                
                if f_candidate < self.fitness[i]:
                    new_pop[i] = candidate
                    self.fitness[i] = f_candidate

            # Abandon some nests
            self.pa = 0.1 + 0.2 * (1 - diversity_factor)  # Dynamic pa value inversely proportional to diversity
            abandon = np.random.rand(self.nests) < self.pa
            for i in range(self.nests):
                if abandon[i] and _ < self.budget - self.nests:
                    new_pop[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    self.fitness[i] = func(new_pop[i])
            
            self.pop = new_pop
            best_idx = np.argmin(self.fitness)
            self.best_solution = self.pop[best_idx]
            self.memory.append(self.best_solution)
            if len(self.memory) > 10:  # Keep memory size manageable
                self.memory.pop(0)
            fitness_rank = np.argsort(self.fitness)

        return self.best_solution