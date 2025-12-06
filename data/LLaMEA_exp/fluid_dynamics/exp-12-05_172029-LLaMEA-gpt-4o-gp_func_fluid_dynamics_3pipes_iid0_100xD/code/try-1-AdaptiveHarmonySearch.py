import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_size = 50
        self.hmcr = 0.9
        self.par = 0.3
        self.dynamic_bw_init = 0.1  # Dynamic bandwidth initialization
        self.bw_decay = 0.99  # Bandwidth decay factor
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.memory_size, self.dim))
        self.memory_fitness = None
        self.dynamic_memory = []

    def initialize_memory(self, func):
        self.memory_fitness = np.apply_along_axis(func, 1, self.memory)

    def generate_new_harmony(self):
        new_harmony = np.zeros(self.dim)
        dynamic_bw = self.dynamic_bw_init * (self.bw_decay ** (len(self.dynamic_memory) // 10))
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(0, self.memory_size)
                new_harmony[i] = self.memory[idx, i]
                if np.random.rand() < self.par:
                    new_harmony[i] += np.random.uniform(-dynamic_bw, dynamic_bw)
            else:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
        return new_harmony

    def update_memory(self, new_harmony, new_fitness):
        worst_idx = np.argmax(self.memory_fitness)
        if new_fitness < self.memory_fitness[worst_idx]:
            self.memory[worst_idx] = new_harmony
            self.memory_fitness[worst_idx] = new_fitness
        self.dynamic_memory.append((new_harmony, new_fitness))
        if len(self.dynamic_memory) > self.memory_size:
            self.dynamic_memory.pop(0)

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