import numpy as np

class DynamicQuantumAdaptiveMemorySwarmSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_rate = 0.1
        self.memory_size = 5  
        self.memory = []
        self.quantum_step = 0.05

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

            diversity = np.mean(np.std(positions, axis=0))
            self.inertia_weight = 0.4 + 0.5 * (1 - diversity) # Adaptive inertia weight based on diversity
            self.social_coeff = 1.5 + 0.5 * diversity # Adaptive social coefficient based on diversity

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
                    mutation = np.random.normal(0, 0.1 * np.std(positions, axis=0), self.dim)
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

            # Implement quantum-inspired position update
            for i in range(self.population_size):
                random_dim = np.random.randint(self.dim)
                positions[i][random_dim] += self.quantum_step * np.sign(global_best_position[random_dim] - positions[i][random_dim])
                positions[i] = np.clip(positions[i], lb, ub)

            if func(previous_global_best_position) < global_best_score:
                global_best_position = previous_global_best_position

            self.memory.append((global_best_position, global_best_score))
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
            
            if len(self.memory) > 1:
                memory_best_position = min(self.memory, key=lambda x: x[1])[0]
                for i in range(self.population_size):
                    positions[i] += np.random.rand() * (memory_best_position - positions[i])

            self.mutation_rate = max(0.01, 0.1 - 0.09 * (self.evaluations / self.budget))

        return global_best_position