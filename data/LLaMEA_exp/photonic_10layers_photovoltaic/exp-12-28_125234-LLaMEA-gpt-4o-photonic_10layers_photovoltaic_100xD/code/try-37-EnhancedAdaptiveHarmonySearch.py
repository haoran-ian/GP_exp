import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony memory considering rate
        self.PAR = 0.3   # Pitch adjusting rate
        self.HMS = 10    # Harmony memory size
        self.FES = 0     # Function evaluations spent
        self.diversity_threshold = 0.1
        self.learning_rate = 0.1  # Dynamic learning rate for adaptive PAR and HMCR

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
                        scale = self.learning_rate * (1 - self.FES / self.budget)
                        new_harmony[i] += np.random.uniform(-scale, scale) * (ub[i] - lb[i])
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
            
            new_fitness = func(new_harmony)
            self.FES += 1

            if new_fitness < harmony_fitness.max():
                worst_idx = harmony_fitness.argmax()
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness

            # Adaptive mechanisms
            self.HMCR = 0.7 + 0.3 * (1 - self.FES / self.budget)
            self.PAR = max(0.1, self.PAR * 0.995)

            # Adaptive diversity-driven reinitialization mechanism
            if self.FES % (self.HMS // 2) == 0:
                diversity = np.var(harmony_memory, axis=0).mean()
                if diversity < self.diversity_threshold * (1 + (self.FES / self.budget)):
                    idx_replace = np.random.choice(self.HMS, size=int(self.HMS * 0.2), replace=False)
                    for idx in idx_replace:
                        harmony_memory[idx] = np.random.uniform(lb, ub, self.dim)
                        harmony_fitness[idx] = func(harmony_memory[idx])
                    self.FES += len(idx_replace)

            # Slight change in adaptive diversity threshold
            self.diversity_threshold *= (1.0 + 0.2 * (self.FES / self.budget))  # Modified line

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]