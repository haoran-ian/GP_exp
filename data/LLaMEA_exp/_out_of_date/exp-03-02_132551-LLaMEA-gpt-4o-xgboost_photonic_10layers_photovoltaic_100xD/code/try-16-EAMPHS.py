import numpy as np

class EAMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmonies = []
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.phase_switch_rate = 0.5  # Ratio to switch between phases
        self.phase_threshold = budget // 3  # Switch phases based on budget
        self.global_to_local_switch_factor = 0.5  # Factor to switch focus
        self.differential_weight = 0.8  # Weight for differential mutation

    def initialize_harmonies(self, bounds):
        for _ in range(self.harmony_memory_size):
            harmony = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            fitness = self.evaluate_harmony(harmony, func)
            self.harmonies.append((harmony, fitness))

    def evaluate_harmony(self, harmony, func):
        return func(harmony)

    def update_harmony_memory(self, new_harmony, new_fitness):
        worst_index = np.argmax([h[1] for h in self.harmonies])
        if new_fitness < self.harmonies[worst_index][1]:
            self.harmonies[worst_index] = (new_harmony, new_fitness)

    def adaptive_memory_consideration(self):
        return 0.85 + 0.15 * np.random.rand()

    def adaptive_pitch_adjustment(self):
        return 0.65 + 0.1 * np.random.rand()

    def differential_mutation(self, bounds):
        indices = np.random.choice(self.harmony_memory_size, 3, replace=False)
        base_harmony = self.harmonies[indices[0]][0]
        diff_vector = self.harmonies[indices[1]][0] - self.harmonies[indices[2]][0]
        mutant_vector = base_harmony + self.differential_weight * diff_vector
        return np.clip(mutant_vector, bounds.lb, bounds.ub)

    def refine_harmony(self, bounds, phase):
        if phase == "global":
            return self.differential_mutation(bounds)
        else:
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    chosen_harmony = self.harmonies[np.random.randint(self.harmony_memory_size)][0]
                    new_harmony[d] = chosen_harmony[d]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[d] += np.random.uniform(-1, 1)
                else:
                    new_harmony[d] = np.random.uniform(bounds.lb[d], bounds.ub[d])
            return np.clip(new_harmony, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmonies(bounds)

        remaining_budget = self.budget - self.harmony_memory_size
        phase = "global"  # Start with global exploration

        while remaining_budget > 0:
            self.harmony_memory_consideration_rate = self.adaptive_memory_consideration()
            self.pitch_adjustment_rate = self.adaptive_pitch_adjustment()

            if remaining_budget <= self.phase_threshold * self.phase_switch_rate:
                phase = "local"

            new_harmony = self.refine_harmony(bounds, phase)
            new_fitness = self.evaluate_harmony(new_harmony, func)
            self.update_harmony_memory(new_harmony, new_fitness)

            remaining_budget -= 1

        best_harmony = min(self.harmonies, key=lambda h: h[1])
        return best_harmony[0]