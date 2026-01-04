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
        self.elite_fraction = 0.1  # Fraction of elite harmonies to retain

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

            # Dynamic memory adjustment
            diversity = np.std(harmony_memory, axis=0).mean()
            if diversity < self.diversity_threshold and self.FES % self.HMS == 0:
                num_elites = int(self.HMS * self.elite_fraction)
                elite_indices = np.argpartition(harmony_fitness, num_elites)[:num_elites]
                new_harmony_memory = np.random.uniform(lb, ub, (self.HMS - num_elites, self.dim))
                new_harmony_fitness = np.array([func(harmony) for harmony in new_harmony_memory])
                self.FES += self.HMS - num_elites
                harmony_memory = np.vstack((harmony_memory[elite_indices], new_harmony_memory))
                harmony_fitness = np.concatenate((harmony_fitness[elite_indices], new_harmony_fitness))

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]