import numpy as np

class AdaptiveCMSAHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_size = max(30, dim * 2)
        self.hmcr = 0.9
        self.par = 0.45
        self.bw = 0.02
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.memory_size, self.dim))
        self.memory_fitness = None
        self.C = np.eye(dim)  # Covariance matrix

    def initialize_memory(self, func):
        self.memory_fitness = np.apply_along_axis(func, 1, self.memory)

    def generate_new_harmony(self):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(0, self.memory_size)
                new_harmony[i] = self.memory[idx, i]
                if np.random.rand() < self.par:
                    self.par = 0.45 * (1 - (np.min(self.memory_fitness) / (np.max(self.memory_fitness) + 1e-6)))
                    dynamic_bw = self.bw * (1 - (np.min(self.memory_fitness) / (np.max(self.memory_fitness) + 1e-6)))
                    new_harmony[i] += np.random.uniform(-dynamic_bw, dynamic_bw)
            else:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        noise = np.random.multivariate_normal(np.zeros(self.dim), self.C)
        new_harmony = np.clip(new_harmony + noise, self.lower_bound, self.upper_bound)
        return new_harmony

    def update_memory(self, new_harmony, new_fitness):
        worst_idx = np.argmax(self.memory_fitness)
        if new_fitness < self.memory_fitness[worst_idx]:
            self.memory[worst_idx] = new_harmony
            self.memory_fitness[worst_idx] = new_fitness
            self.C = self._update_covariance_matrix(new_harmony)

    def _update_covariance_matrix(self, new_harmony):
        mean_harmony = np.mean(self.memory, axis=0)
        diffs = self.memory - mean_harmony
        return np.cov(diffs.T) + np.outer(new_harmony - mean_harmony, new_harmony - mean_harmony)

    def __call__(self, func):
        self.initialize_memory(func)
        evaluations = self.memory_size

        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony()
            new_fitness = func(new_harmony)
            evaluations += 1
            self.update_memory(new_harmony, new_fitness)

        best_idx = np.argmin(self.memory_fitness)
        return self.memory[best_idx], self.memory_fitness[best_idx]