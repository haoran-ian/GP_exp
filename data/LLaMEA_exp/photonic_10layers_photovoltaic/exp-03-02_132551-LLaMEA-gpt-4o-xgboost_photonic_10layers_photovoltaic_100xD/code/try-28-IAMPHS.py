import numpy as np

class IAMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmonies = []
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.exploration_phase_rate = 0.8  # Initial strong focus on exploration
        self.exploitation_phase_rate = 0.2  # Initial lesser focus on exploitation
        self.dynamic_phase_threshold = budget // 2  # Dynamic switching by budget use

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

    def adaptive_memory_consideration(self):
        return 0.85 + 0.15 * np.random.rand()

    def adaptive_pitch_adjustment(self, diversity_factor):
        return 0.5 + 0.5 * np.random.rand() * diversity_factor

    def calculate_diversity_factor(self):
        harmonies_array = np.array([h[0] for h in self.harmonies])
        diversity = np.std(harmonies_array, axis=0).mean()  # Mean standard deviation across dimensions
        return max(0.1, min(1.0, diversity))  # Ensure diversity factor is between 0.1 and 1.0

    def refine_harmony(self, bounds, phase, diversity_factor):
        new_harmony = np.zeros(self.dim)
        for d in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                chosen_harmony = self.harmonies[np.random.randint(self.harmony_memory_size)][0]
                new_harmony[d] = chosen_harmony[d]
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[d] += np.random.uniform(-0.5, 0.5) * (phase == "local") * diversity_factor  # Changed range from -1, 1 to -0.5, 0.5
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
            self.harmony_memory_consideration_rate = self.adaptive_memory_consideration()
            diversity_factor = self.calculate_diversity_factor()
            self.pitch_adjustment_rate = self.adaptive_pitch_adjustment(diversity_factor)

            if remaining_budget <= self.dynamic_phase_threshold:
                phase = "local"

            new_harmony = self.refine_harmony(bounds, phase, diversity_factor)
            new_fitness = self.evaluate_harmony(new_harmony, func)
            self.update_harmony_memory(new_harmony, new_fitness)

            remaining_budget -= 1

        best_harmony = min(self.harmonies, key=lambda h: h[1])
        return best_harmony[0]