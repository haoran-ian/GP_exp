import numpy as np

class DynamicAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony memory considering rate
        self.PAR = 0.3   # Pitch adjusting rate
        self.HMS = 20    # Increased harmony memory size for better diversity
        self.FES = 0     # Function evaluations spent

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.HMS, self.dim))
        harmony_fitness = np.array([func(harmony) for harmony in harmony_memory])
        self.FES += self.HMS

        while self.FES < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.HMCR:
                    idx = np.random.randint(self.HMS)
                    new_harmony[i] = harmony_memory[idx, i]
                    if np.random.rand() < self.PAR:
                        adjustment_factor = np.random.uniform(-0.5, 0.5) * (ub[i] - lb[i])
                        new_harmony[i] += adjustment_factor * (np.random.rand() - 0.5)
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
            
            new_fitness = func(new_harmony)
            self.FES += 1

            if new_fitness < harmony_fitness.max():
                worst_idx = harmony_fitness.argmax()
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness

            # Dynamic adaptation of rates
            self.HMCR = 0.5 + 0.4 * np.cos(np.pi * self.FES / self.budget)
            self.PAR = 0.2 + 0.3 * np.sin(np.pi * self.FES / (2 * self.budget))

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]