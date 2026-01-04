import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony memory considering rate
        self.PAR = 0.3   # Pitch adjusting rate
        self.initial_HMS = 10  # Initial harmony memory size
        self.FES = 0     # Function evaluations spent
        self.diversity_threshold = 0.1
        self.learning_rate = 0.1  # Dynamic learning rate for adaptive PAR and HMCR
        self.exploration_intensity = 0.2  # Exploration intensity factor
        self.exploration_size = max(1, int(self.initial_HMS * 0.1))  # Initial exploration size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        HMS = self.initial_HMS
        harmony_memory = np.random.uniform(lb, ub, (HMS, self.dim))
        harmony_fitness = np.array([func(harmony) for harmony in harmony_memory])
        self.FES += HMS

        while self.FES < self.budget:
            # Dynamically adjust HMS based on remaining budget
            if self.FES < self.budget / 2:
                HMS = max(self.initial_HMS, int(HMS * 1.1))
            else:
                HMS = min(self.initial_HMS, int(HMS * 0.95))
            
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.HMCR:
                    idx = np.random.randint(HMS)
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

            # Adaptive parameters update
            self.HMCR = 0.7 + 0.3 * (1 - self.FES / self.budget)
            self.PAR = max(0.1, self.PAR * 0.995)

            # Adaptive diversity-driven reinitialization mechanism
            if self.FES % (HMS // 2) == 0:
                diversity = np.var(harmony_memory, axis=0).mean()
                if diversity < self.diversity_threshold * (1 + (self.FES / self.budget)):
                    idx_replace = np.random.choice(HMS, size=int(HMS * 0.2), replace=False)
                    for idx in idx_replace:
                        harmony_memory[idx] = np.random.uniform(lb, ub, self.dim)
                        harmony_fitness[idx] = func(harmony_memory[idx])
                    self.FES += len(idx_replace)

            # Slight change in adaptive diversity threshold
            self.diversity_threshold *= (0.9 + 0.2 * (self.FES / self.budget))

            # Dynamic exploration bursts
            if self.FES % (self.budget // 5) == 0:
                exploration_size = min(HMS, max(1, int(HMS * self.exploration_intensity)))
                for _ in range(exploration_size):
                    exploration_idx = np.random.randint(HMS)
                    harmony_memory[exploration_idx] = np.random.uniform(lb, ub, self.dim)
                    harmony_fitness[exploration_idx] = func(harmony_memory[exploration_idx])

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]