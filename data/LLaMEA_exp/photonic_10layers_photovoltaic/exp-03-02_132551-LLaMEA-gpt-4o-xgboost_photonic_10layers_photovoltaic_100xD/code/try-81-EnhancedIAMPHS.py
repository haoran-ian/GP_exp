import numpy as np

class EnhancedIAMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmonies = []
        self.hierarchy_level = 2
        self.memory_levels = [[] for _ in range(self.hierarchy_level)]
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.exploration_phase_rate = 0.8
        self.exploitation_phase_rate = 0.2
        self.dynamic_phase_threshold = budget // (2 * self.hierarchy_level)
        self.convergence_rate = []
        self.exploration_bias = 1.5  

    def initialize_harmonies(self, bounds):
        for _ in range(self.harmony_memory_size):
            harmony = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            fitness = None
            self.harmonies.append((harmony, fitness))
        for level in range(self.hierarchy_level):
            for _ in range(self.harmony_memory_size):
                harmony = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                fitness = None
                self.memory_levels[level].append((harmony, fitness))

    def evaluate_harmony(self, harmony, func):
        return func(harmony)

    def update_harmony_memory(self, new_harmony, new_fitness, level):
        worst_index = np.argmax([h[1] for h in self.memory_levels[level]])
        if new_fitness < self.memory_levels[level][worst_index][1]:
            self.memory_levels[level][worst_index] = (new_harmony, new_fitness)

    def adaptive_memory_consideration(self):
        return 0.85 + 0.15 * np.random.rand()

    def adaptive_pitch_adjustment(self, diversity_factor, phase):
        phase_factor = 0.5 if phase == "global" else 1.0
        amplitude_adjustment = 1.2 if phase == "global" else 0.8
        return 0.5 + 0.5 * np.random.rand() * diversity_factor * phase_factor * amplitude_adjustment

    def calculate_diversity_factor(self):
        harmonies_array = np.array([h[0] for h in self.harmonies])
        diversity = np.std(harmonies_array, axis=0).mean()
        return max(0.05, min(1.0, diversity))

    def refine_harmony(self, bounds, phase, diversity_factor, level):
        new_harmony = np.zeros(self.dim)
        for d in range(self.dim):
            if np.random.rand() < self.harmony_memory_consideration_rate:
                chosen_harmony = self.memory_levels[level][np.random.randint(self.harmony_memory_size)][0]
                new_harmony[d] = chosen_harmony[d]
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[d] += np.random.uniform(-1, 1) * (phase == "local") * diversity_factor
            else:
                new_harmony[d] = np.random.uniform(bounds.lb[d], bounds.ub[d]) * (phase == "global")
        
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def dynamic_phase_transition(self, level):
        if len(self.convergence_rate) < 2:
            return "global"
        improvement = self.convergence_rate[-2] - self.convergence_rate[-1]
        if improvement < 1e-4:
            self.exploration_bias = min(3.0, self.exploration_bias + 0.2)
        else:
            self.exploration_bias = max(1.0, self.exploration_bias - 0.2)
        
        if self.exploration_bias > 2.0 / (level + 1):
            return "global"
        return "local"

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmonies(bounds)

        remaining_budget = self.budget
        for level in range(self.hierarchy_level):
            initial_evaluations = [self.evaluate_harmony(h[0], func) for h in self.memory_levels[level]]
            for i, (harmony, _) in enumerate(self.memory_levels[level]):
                self.memory_levels[level][i] = (harmony, initial_evaluations[i])

            remaining_budget -= len(self.memory_levels[level])
            best_fitness = min([h[1] for h in self.memory_levels[level]])
            self.convergence_rate.append(best_fitness)

            while remaining_budget > 0:
                self.harmony_memory_consideration_rate = self.adaptive_memory_consideration()
                diversity_factor = self.calculate_diversity_factor()
                phase = self.dynamic_phase_transition(level)
                self.pitch_adjustment_rate = self.adaptive_pitch_adjustment(diversity_factor, phase)

                new_harmony = self.refine_harmony(bounds, phase, diversity_factor, level)
                new_fitness = self.evaluate_harmony(new_harmony, func)
                self.update_harmony_memory(new_harmony, new_fitness, level)

                current_best_fitness = min([h[1] for h in self.memory_levels[level]])
                self.convergence_rate.append(current_best_fitness)
                remaining_budget -= 1

        best_harmony = min([(h, lvl) for lvl in self.memory_levels for h in lvl], key=lambda x: x[0][1])
        return best_harmony[0][0]