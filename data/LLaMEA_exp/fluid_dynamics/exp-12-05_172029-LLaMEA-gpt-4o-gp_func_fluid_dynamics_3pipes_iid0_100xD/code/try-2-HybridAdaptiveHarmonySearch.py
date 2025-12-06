import numpy as np

class HybridAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_size = 50
        self.hmcr = 0.9  
        self.par = 0.3  
        self.bw = 0.01  
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.memory_size, self.dim))
        self.memory_fitness = None

    def initialize_memory(self, func):
        self.memory_fitness = np.apply_along_axis(func, 1, self.memory)

    def generate_new_harmony(self):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(0, self.memory_size)
                new_harmony[i] = self.memory[idx, i]
                if np.random.rand() < self.par:
                    new_harmony[i] += np.random.normal(0, self.bw)  # Changed from uniform to normal distribution
            else:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return np.clip(new_harmony, self.lower_bound, self.upper_bound)

    def local_refinement(self, harmony, func):  # Added new local refinement phase
        refined_harmony = harmony + np.random.normal(0, 0.05, self.dim)
        refined_harmony = np.clip(refined_harmony, self.lower_bound, self.upper_bound)
        refined_fitness = func(refined_harmony)
        return refined_harmony, refined_fitness

    def update_memory(self, new_harmony, new_fitness):
        worst_idx = np.argmax(self.memory_fitness)
        if new_fitness < self.memory_fitness[worst_idx]:
            self.memory[worst_idx] = new_harmony
            self.memory_fitness[worst_idx] = new_fitness

    def __call__(self, func):
        self.initialize_memory(func)
        evaluations = self.memory_size

        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony()
            new_harmony, new_fitness = self.local_refinement(new_harmony, func)  # Incorporate local refinement
            evaluations += 1
            self.update_memory(new_harmony, new_fitness)

        best_idx = np.argmin(self.memory_fitness)
        return self.memory[best_idx], self.memory_fitness[best_idx]