import numpy as np

class IAMPHS_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmonies = []
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.dynamic_threshold = budget // 3  # Reduce threshold for earlier dynamics
        self.convergence_rate = []

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
        return 0.8 + 0.2 * np.random.rand()

    def adaptive_pitch_adjustment(self, diversity_factor, phase):
        phase_factor = 0.3 if phase == "global" else 1.2
        return 0.4 + 0.6 * np.random.rand() * diversity_factor * phase_factor

    def calculate_diversity_factor(self):
        harmonies_array = np.array([h[0] for h in self.harmonies])
        diversity = np.std(harmonies_array, axis=0).mean()
        return max(0.1, min(1.0, diversity))

    def refine_harmony(self, bounds, phase, diversity_factor):
        new_harmony = np.zeros(self.dim)
        for d in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                chosen_harmony = self.harmonies[np.random.randint(self.harmony_memory_size)][0]
                new_harmony[d] = chosen_harmony[d]
                if np.random.rand() < self.pitch_adjustment_rate:
                    adjustment = np.random.uniform(-1, 1) * diversity_factor
                    new_harmony[d] += adjustment * (0.5 if phase == "local" else 1.0)
            else:
                new_harmony[d] = np.random.uniform(bounds.lb[d], bounds.ub[d])
        
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def dynamic_exploration_exploitation_swapping(self):
        recent_improvement = (self.convergence_rate[-2] - self.convergence_rate[-1]) if len(self.convergence_rate) > 1 else 1
        diversity_factor = self.calculate_diversity_factor()
        if recent_improvement > 0.01 and diversity_factor < 0.3:
            return "global"
        else:
            return "local"

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmonies(bounds)

        remaining_budget = self.budget
        initial_evaluations = [self.evaluate_harmony(h[0], func) for h in self.harmonies]
        for i, (harmony, _) in enumerate(self.harmonies):
            self.harmonies[i] = (harmony, initial_evaluations[i])

        remaining_budget -= len(self.harmonies)
        best_fitness = min([h[1] for h in self.harmonies])
        self.convergence_rate.append(best_fitness)

        while remaining_budget > 0:
            self.harmony_memory_consideration_rate = self.adaptive_memory_consideration()
            diversity_factor = self.calculate_diversity_factor()
            phase = self.dynamic_exploration_exploitation_swapping()
            self.pitch_adjustment_rate = self.adaptive_pitch_adjustment(diversity_factor, phase)

            new_harmony = self.refine_harmony(bounds, phase, diversity_factor)
            new_fitness = self.evaluate_harmony(new_harmony, func)
            self.update_harmony_memory(new_harmony, new_fitness)

            current_best_fitness = min([h[1] for h in self.harmonies])
            self.convergence_rate.append(current_best_fitness)
            remaining_budget -= 1

        best_harmony = min(self.harmonies, key=lambda h: h[1])
        return best_harmony[0]