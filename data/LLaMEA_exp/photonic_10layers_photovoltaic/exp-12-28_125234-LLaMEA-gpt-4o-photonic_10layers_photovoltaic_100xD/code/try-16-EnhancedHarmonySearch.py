import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony memory considering rate
        self.PAR = 0.3   # Pitch adjusting rate
        self.HMS = 10    # Harmony memory size
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
                        new_harmony[i] += np.random.normal(0, (ub[i] - lb[i]) * 0.01)
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
            
            new_fitness = func(new_harmony)
            self.FES += 1

            if new_fitness < harmony_fitness.max():
                worst_idx = harmony_fitness.argmax()
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness

            # Dynamic PAR adjustment
            self.PAR = 0.3 * (1 - self.FES / self.budget)

            # Diversity-driven repair and elitism strategy
            if self.FES % self.HMS == 0:
                diversity = np.std(harmony_memory, axis=0).mean()
                if diversity < self.diversity_threshold:
                    idx_replace = np.random.choice(self.HMS, size=int(self.HMS * 0.2), replace=False)
                    for idx in idx_replace:
                        harmony_memory[idx] = np.random.uniform(lb, ub, self.dim)
                        harmony_fitness[idx] = func(harmony_memory[idx])
                    self.FES += len(idx_replace)

                # Introducing elitism: preserve the best solution
                best_idx = harmony_fitness.argmin()
                best_solution = harmony_memory[best_idx]
                harmony_memory[np.random.randint(self.HMS)] = best_solution

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]