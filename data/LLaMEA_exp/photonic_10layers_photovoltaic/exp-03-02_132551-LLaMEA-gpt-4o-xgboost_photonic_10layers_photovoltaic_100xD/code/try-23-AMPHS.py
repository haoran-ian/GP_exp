import numpy as np

class AMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmonies = []
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.global_phase_rate = 0.8  # Initial focus on exploration
        self.local_phase_rate = 0.2  # Initial focus on exploitation
        self.phase_threshold = budget // 3  # Switch phases after one-third budget

    def initialize_harmonies(self, bounds):
        for _ in range(self.harmony_memory_size):
            harmony = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            fitness = None
            self.harmonies.append((harmony, fitness))

    def evaluate_harmony(self, harmony, func):
        return func(harmony)

    def update_harmony_memory(self, new_harmony, new_fitness):
        worst_index = np.argmax([h[1] for h in self.harmonies])
        if new_fitness < self.harmonies[worst_index][1]:
            self.harmonies[worst_index] = (new_harmony, new_fitness)

    def adaptive_memory_consideration(self, remaining_budget):
        max_rate = 0.95
        min_rate = 0.85
        return min_rate + (max_rate - min_rate) * (remaining_budget / self.budget)

    def adaptive_pitch_adjustment(self):
        return 0.65 + 0.1 * np.random.rand()

    def refine_harmony(self, bounds, phase):
        new_harmony = np.zeros(self.dim)
        for d in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                chosen_harmony = self.harmonies[np.random.randint(self.harmony_memory_size)][0]
                new_harmony[d] = chosen_harmony[d]
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[d] += np.random.uniform(-1, 1) * (phase == "local")
            else:
                new_harmony[d] = np.random.uniform(bounds.lb[d], bounds.ub[d]) * (phase == "global")
        
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmonies(bounds)

        remaining_budget = self.budget
        initial_evaluations = [self.evaluate_harmony(h[0], func) for h in self.harmonies]
        for i, (harmony, _) in enumerate(self.harmonies):
            self.harmonies[i] = (harmony, initial_evaluations[i])

        remaining_budget -= len(self.harmonies)
        phase = "global"  # Start with global exploration

        while remaining_budget > 0:
            self.harmony_memory_consideration_rate = self.adaptive_memory_consideration(remaining_budget)
            self.pitch_adjustment_rate = self.adaptive_pitch_adjustment()

            if remaining_budget <= self.phase_threshold:
                phase = "local"

            new_harmony = self.refine_harmony(bounds, phase)
            new_fitness = self.evaluate_harmony(new_harmony, func)
            self.update_harmony_memory(new_harmony, new_fitness)

            remaining_budget -= 1

        best_harmony = min(self.harmonies, key=lambda h: h[1])
        return best_harmony[0]