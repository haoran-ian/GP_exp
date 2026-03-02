import numpy as np

class EnhancedIAMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmonies = []
        self.exploration_exploitation_balance = 0.8  # Initially favor exploration
        self.dynamic_phase_threshold = budget // 3  # Trigger local refinement earlier

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

    def adjust_exploration_exploitation(self, progress):
        return 0.5 + 0.5 * (1 - progress)  # Decrease exploration as progress increases

    def calculate_diversity_factor(self):
        harmonies_array = np.array([h[0] for h in self.harmonies])
        diversity = np.std(harmonies_array, axis=0).mean()
        return max(0.1, min(1.0, diversity))

    def refine_harmony(self, bounds, phase, diversity_factor):
        new_harmony = np.zeros(self.dim)
        for d in range(self.dim):
            if np.random.rand() < self.exploration_exploitation_balance:
                chosen_harmony = self.harmonies[np.random.randint(self.harmony_memory_size)][0]
                new_harmony[d] = chosen_harmony[d]
                if np.random.rand() < 1.0 - phase:
                    distance_to_best = np.linalg.norm(chosen_harmony - self.best_harmony())
                    new_harmony[d] += np.random.uniform(-distance_to_best, distance_to_best) * diversity_factor
            else:
                new_harmony[d] = np.random.uniform(bounds.lb[d], bounds.ub[d])
        
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def best_harmony(self):
        return min(self.harmonies, key=lambda h: h[1])[0]

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmonies(bounds)

        remaining_budget = self.budget
        initial_evaluations = [self.evaluate_harmony(h[0], func) for h in self.harmonies]
        for i, (harmony, _) in enumerate(self.harmonies):
            self.harmonies[i] = (harmony, initial_evaluations[i])

        remaining_budget -= len(self.harmonies)
        phase = 1.0  # Start with exploration

        while remaining_budget > 0:
            progress = (self.budget - remaining_budget) / self.budget
            self.exploration_exploitation_balance = self.adjust_exploration_exploitation(progress)
            diversity_factor = self.calculate_diversity_factor()

            if remaining_budget <= self.dynamic_phase_threshold:
                phase = 0.0  # Trigger local refinement

            new_harmony = self.refine_harmony(bounds, phase, diversity_factor)
            new_fitness = self.evaluate_harmony(new_harmony, func)
            self.update_harmony_memory(new_harmony, new_fitness)

            remaining_budget -= 1

        return self.best_harmony()