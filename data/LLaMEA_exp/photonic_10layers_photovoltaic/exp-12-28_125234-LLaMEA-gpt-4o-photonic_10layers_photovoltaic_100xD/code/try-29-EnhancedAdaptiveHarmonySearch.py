import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR_initial = 0.9  # Initial Harmony memory considering rate
        self.PAR_initial = 0.3   # Initial Pitch adjusting rate
        self.HMS = 10            # Harmony memory size
        self.FES = 0             # Function evaluations spent
        self.diversity_threshold = 0.1
        self.learning_rate = 0.1
        self.HMCR = self.HMCR_initial
        self.PAR = self.PAR_initial

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

            # Adaptive mechanisms with feedback
            if self.FES > self.budget * 0.1:  # Start adjusting after 10% of budget
                improvement = (harmony_fitness.min() - new_fitness) / abs(harmony_fitness.min())
                self.HMCR = min(1.0, max(self.HMCR_initial * (1 + improvement), 0.7))
                self.PAR = min(0.5, max(0.1, self.PAR_initial * (1 - improvement)))

            # Enhanced diversity-driven reinitialization mechanism
            if self.FES % max(1, (self.HMS // 2)) == 0:
                diversity = np.var(harmony_memory, axis=0).mean()
                if diversity < self.diversity_threshold:
                    idx_replace = np.random.choice(self.HMS, size=int(self.HMS * 0.2), replace=False)
                    for idx in idx_replace:
                        harmony_memory[idx] = np.random.uniform(lb, ub, self.dim)
                        harmony_fitness[idx] = func(harmony_memory[idx])
                    self.FES += len(idx_replace)

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]