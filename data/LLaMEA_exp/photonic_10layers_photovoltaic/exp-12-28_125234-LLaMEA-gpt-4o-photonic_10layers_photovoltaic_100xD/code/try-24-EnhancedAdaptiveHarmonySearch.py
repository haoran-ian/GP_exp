import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony memory considering rate
        self.PAR = 0.3   # Pitch adjusting rate
        self.HMS = 10    # Initial harmony memory size
        self.FES = 0     # Function evaluations spent
        self.diversity_threshold = 0.1

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
                        new_harmony[i] += np.random.uniform(-1, 1) * (ub[i] - lb[i]) * np.random.uniform(0.05, 0.2)
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
            self.HMCR = 0.7 + 0.3 * (self.budget - self.FES) / self.budget
            self.PAR *= 0.995

            # Dynamic harmony memory size adjustment
            if self.FES % (self.HMS * 5) == 0 and self.HMS < 50:
                self.HMS += 1
                new_harmony = np.random.uniform(lb, ub, self.dim)
                new_fitness = func(new_harmony)
                harmony_memory = np.vstack([harmony_memory, new_harmony])
                harmony_fitness = np.append(harmony_fitness, new_fitness)
                self.FES += 1
            
            # Adaptive diversity rejuvenation
            if self.FES % (self.HMS * 2) == 0:
                diversity = np.std(harmony_memory, axis=0).mean()
                if diversity < self.diversity_threshold:
                    idx_replace = np.random.choice(self.HMS, size=int(self.HMS * 0.2), replace=False)
                    for idx in idx_replace:
                        harmony_memory[idx] = np.random.uniform(lb, ub, self.dim)
                        harmony_fitness[idx] = func(harmony_memory[idx])
                    self.FES += len(idx_replace)

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]