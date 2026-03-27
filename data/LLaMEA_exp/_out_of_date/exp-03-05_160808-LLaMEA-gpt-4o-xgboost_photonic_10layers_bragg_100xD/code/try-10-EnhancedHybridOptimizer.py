import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.nests = 30  # increased number of nests for diversity
        self.pa = 0.25   # adjusted probability of abandoning a solution
        self.beta = 1.5  # Levy flight parameter
        self.lr = 0.85   # Local refinement probability
        self.pop = None
        self.fitness = None
        self.memory = []  # Memory to store elite solutions

    def levy_flight(self, size):
        sigma = (np.math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                 (np.math.gamma((1 + self.beta) / 2) * self.beta *
                  2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, size=size)
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

    def dual_space_exploration(self, candidate, best_solution):
        # Introduce dual-space exploration
        step_size = self.levy_flight(self.dim) * (candidate - best_solution)
        new_candidate = candidate + step_size * np.random.uniform(-1, 1.2, self.dim)
        return np.clip(new_candidate, self.lb, self.ub)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.pop = np.random.uniform(self.lb, self.ub, (self.nests, self.dim))
        self.fitness = np.array([func(ind) for ind in self.pop])
        best_idx = np.argmin(self.fitness)
        best_solution = self.pop[best_idx]
        
        for _ in range(self.budget - self.nests):
            new_pop = self.pop.copy()
            for i in range(self.nests):
                if np.random.rand() < self.lr:
                    candidate = self.differential_evolution(i, F=np.random.rand(), CR=np.random.rand())
                else:
                    candidate = self.dual_space_exploration(self.pop[i], best_solution)
                f_candidate = func(candidate)
                
                if f_candidate < self.fitness[i]:
                    new_pop[i] = candidate
                    self.fitness[i] = f_candidate

            # Abandon some nests and revisit historical solutions
            abandon = np.random.rand(self.nests) < self.pa
            for i in range(self.nests):
                if abandon[i] and _ < self.budget - self.nests:
                    if self.memory and np.random.rand() < 0.5:
                        new_pop[i] = self.memory[np.random.randint(0, len(self.memory))]
                    else:
                        new_pop[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    self.fitness[i] = func(new_pop[i])
            
            # Update memory with elite solutions
            elite_idx = np.argsort(self.fitness)[:5]
            self.memory.extend(self.pop[elite_idx])
            self.memory = sorted(self.memory, key=lambda x: func(x))[:10]  # Retain top 10

            self.pop = new_pop
            best_idx = np.argmin(self.fitness)
            best_solution = self.pop[best_idx]

        return best_solution