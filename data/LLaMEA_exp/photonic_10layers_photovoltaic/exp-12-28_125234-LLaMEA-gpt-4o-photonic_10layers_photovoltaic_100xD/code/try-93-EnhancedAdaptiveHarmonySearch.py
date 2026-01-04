import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9
        self.PAR = 0.3
        self.HMS = 10
        self.FES = 0
        self.diversity_threshold = 0.1
        self.learning_rate = 0.1
        self.global_phase_ratio = 0.5
        self.local_phase_ratio = 0.3
        self.scaling_factor = 0.1
        self.reset_counter = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.HMS, self.dim))
        harmony_fitness = np.array([func(harmony) for harmony in harmony_memory])
        self.FES += self.HMS
        global_phase_end = int(self.budget * self.global_phase_ratio)
        local_phase_end = int(self.budget * (self.global_phase_ratio + self.local_phase_ratio))
        
        while self.FES < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.HMCR:
                    idx = np.random.randint(self.HMS)
                    new_harmony[i] = harmony_memory[idx, i]
                    if np.random.rand() < self.PAR:
                        scale = self.learning_rate * (1 - self.FES / self.budget)
                        new_harmony[i] += np.random.uniform(-scale, scale) * (ub[i] - lb[i])
                else:
                    exploration_scale = self.scaling_factor * (1 - self.FES / self.budget)**2
                    if self.FES < global_phase_end:
                        new_harmony[i] = np.random.uniform(lb[i], ub[i])
                    else:
                        best_idx = harmony_fitness.argmin()
                        neighborhood_radius = 0.1 * (1 - self.FES / self.budget)
                        new_harmony[i] = np.clip(harmony_memory[best_idx, i] + np.random.uniform(-neighborhood_radius, neighborhood_radius) * (ub[i] - lb[i]), lb[i], ub[i])
            
            new_fitness = func(new_harmony)
            self.FES += 1

            if new_fitness < harmony_fitness.max():
                worst_idx = harmony_fitness.argmax()
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness

            self.HMCR = 0.7 + 0.3 * (1 - self.FES / self.budget)
            self.PAR = max(0.1, self.PAR * 0.995)

            if self.FES % (self.HMS // 2) == 0:
                diversity = np.var(harmony_memory, axis=0).mean()
                if diversity < self.diversity_threshold * (1 + (self.FES / self.budget)):
                    idx_replace = np.random.choice(self.HMS, size=int(self.HMS * 0.2), replace=False)
                    for idx in idx_replace:
                        harmony_memory[idx] = np.random.uniform(lb, ub, self.dim)
                        harmony_fitness[idx] = func(harmony_memory[idx])
                    self.FES += len(idx_replace)

            self.diversity_threshold *= (0.9 + 0.2 * (self.FES / self.budget))

            if self.FES % (self.budget // 10) == 0:
                exploration_idx = np.random.randint(self.HMS)
                harmony_memory[exploration_idx] = np.random.uniform(lb, ub, self.dim)
                harmony_fitness[exploration_idx] = func(harmony_memory[exploration_idx])

            if self.FES > local_phase_end and self.reset_counter < 3:
                best_idx = harmony_fitness.argmin()
                harmony_memory = np.random.uniform(lb, ub, (self.HMS, self.dim))
                harmony_memory[0] = harmony_memory[best_idx]
                harmony_fitness = np.array([func(harmony) for harmony in harmony_memory])
                self.FES += self.HMS
                self.reset_counter += 1

        best_idx = harmony_fitness.argmin()
        return harmony_memory[best_idx]