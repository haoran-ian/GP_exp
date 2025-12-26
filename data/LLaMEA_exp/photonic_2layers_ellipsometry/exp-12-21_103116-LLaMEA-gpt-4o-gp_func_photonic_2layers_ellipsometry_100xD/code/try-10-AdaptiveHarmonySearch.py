import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 20
        self.harmony_memory_consideration_rate = 0.9
        self.adjusting_rate = 0.5
        self.bandwidth = 0.05
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        harmony_values = np.array([func(h) for h in harmony_memory])
        self.evaluations += self.harmony_memory_size

        best_harmony_index = np.argmin(harmony_values)
        best_harmony = harmony_memory[best_harmony_index]
        best_value = harmony_values[best_harmony_index]

        while self.evaluations < self.budget:
            new_harmony = self.create_new_harmony(harmony_memory, lb, ub)
            new_value = func(new_harmony)
            self.evaluations += 1

            worst_index = np.argmax(harmony_values)
            if new_value < harmony_values[worst_index]:
                harmony_memory[worst_index] = new_harmony
                harmony_values[worst_index] = new_value

                if new_value < best_value:
                    best_harmony = new_harmony
                    best_value = new_value

            self.adapt_bandwidth(harmony_values)

        return best_harmony, best_value

    def create_new_harmony(self, harmony_memory, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size)][i]
                if np.random.rand() < self.adjusting_rate:
                    new_harmony[i] += self.bandwidth * np.random.uniform(-1, 1)
            else:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        return np.clip(new_harmony, lb, ub)

    def adapt_bandwidth(self, harmony_values):
        diversity = np.mean(np.std(harmony_values))
        self.bandwidth = min(max(0.001, self.bandwidth * (1 + 0.2 * (diversity - 0.1))), 0.1)