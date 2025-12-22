import numpy as np

class MultiLayeredAdaptiveMemorySwarmSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_rate = 0.1
        self.memory_size = 5  # Memory for storing past best solutions
        self.memory_layers = [[], []]  # Two layers of memory
        self.inertia_weight_decay = (0.9 - 0.4) / self.budget

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        while self.evaluations < self.budget:
            previous_global_best_position = np.copy(global_best_position)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                    + self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                )
                positions[i] += velocities[i]

                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, 0.1, self.dim)
                    positions[i] += mutation

                positions[i] = np.clip(positions[i], lb, ub)

                score = func(positions[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best_position = positions[i]
                        global_best_score = score

            self.inertia_weight = max(0.4, self.inertia_weight - self.inertia_weight_decay)

            if func(previous_global_best_position) < global_best_score:
                global_best_position = previous_global_best_position

            self.social_coeff = min(2.0, self.social_coeff + (global_best_score / (np.mean(personal_best_scores) + 1e-8)) * 0.1)

            # Multi-layer Memory mechanism
            self.memory_layers[0].append((global_best_position, global_best_score))
            if len(self.memory_layers[0]) > self.memory_size:
                self.memory_layers[0].pop(0)
            
            if np.random.rand() < 0.5:  # 50% chance to promote to second layer
                self.memory_layers[1].extend(self.memory_layers[0])
                self.memory_layers[0] = []
                self.memory_layers[1] = sorted(self.memory_layers[1], key=lambda x: x[1])[:self.memory_size]

            # Use the best from all layers to improve exploration-exploitation
            all_memory = self.memory_layers[0] + self.memory_layers[1]
            if len(all_memory) > 1:
                memory_best_position = min(all_memory, key=lambda x: x[1])[0]
                for i in range(self.population_size):
                    positions[i] += np.random.rand() * (memory_best_position - positions[i])

        return global_best_position