import numpy as np

class ACH_S:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmonies = []
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.adaptive_cluster_threshold = 0.05

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

    def adaptive_cluster(self, bounds):
        distances = [np.linalg.norm(h1[0] - h2[0]) for i, h1 in enumerate(self.harmonies) for h2 in self.harmonies[i+1:]]
        if len(distances) > 0 and np.mean(distances) < self.adaptive_cluster_threshold:
            for harmony, _ in self.harmonies:
                cluster_center = np.mean([h[0] for h in self.harmonies], axis=0)
                harmony += np.random.uniform(-0.1, 0.1, self.dim) * (cluster_center - harmony)
                harmony = np.clip(harmony, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmonies(bounds)

        remaining_budget = self.budget
        initial_evaluations = [self.evaluate_harmony(h[0], func) for h in self.harmonies]
        for i, (harmony, _) in enumerate(self.harmonies):
            self.harmonies[i] = (harmony, initial_evaluations[i])

        remaining_budget -= len(self.harmonies)

        while remaining_budget > 0:
            new_harmony = np.zeros(self.dim)
            self.harmony_memory_consideration_rate = 0.9 + 0.1 * np.random.rand()  # Dynamic adjustment
            self.pitch_adjustment_rate = 0.6 + 0.1 * np.random.rand()  # Dynamic adjustment

            for d in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    chosen_harmony = self.harmonies[np.random.randint(self.harmony_memory_size)][0]
                    new_harmony[d] = chosen_harmony[d]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[d] += np.random.uniform(-1, 1)
                else:
                    new_harmony[d] = np.random.uniform(bounds.lb[d], bounds.ub[d])
            
            new_harmony = np.clip(new_harmony, bounds.lb, bounds.ub)
            new_fitness = self.evaluate_harmony(new_harmony, func)
            self.update_harmony_memory(new_harmony, new_fitness)

            remaining_budget -= 1
            if remaining_budget % (self.budget // 10) == 0:
                self.adaptive_cluster(bounds)

        best_harmony = min(self.harmonies, key=lambda h: h[1])
        return best_harmony[0]