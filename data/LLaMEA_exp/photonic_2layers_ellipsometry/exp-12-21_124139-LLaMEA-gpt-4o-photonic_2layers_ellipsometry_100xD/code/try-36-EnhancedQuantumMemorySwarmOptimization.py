import numpy as np

class EnhancedQuantumMemorySwarmOptimization:
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
        self.elite_ratio = 0.1
        self.elite_size = max(1, int(self.elite_ratio * self.population_size))
        self.inertia_weight_decay = (0.9 - 0.4) / self.budget
        self.quantum_step = 0.1  # Increased step size for quantum-inspired updates
        self.adaptive_learning = True

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
            # Preserve elite solutions
            elite_indices = np.argsort(personal_best_scores)[:self.elite_size]
            elite_positions = positions[elite_indices]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                    + self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                )

                # Apply quantum-inspired position update
                if np.random.rand() < 0.5:
                    random_dim = np.random.randint(self.dim)
                    velocities[i][random_dim] += self.quantum_step * np.sign(global_best_position[random_dim] - positions[i][random_dim])

                positions[i] += velocities[i] + np.random.normal(0, self.mutation_rate, self.dim)
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
            self.social_coeff = min(2.5, self.social_coeff + (global_best_score / (np.mean(personal_best_scores) + 1e-8)) * 0.2)

            # Update memory with elite solutions
            for elite_pos in elite_positions:
                self.memory.append((elite_pos, func(elite_pos)))
            self.memory.sort(key=lambda x: x[1])
            self.memory = self.memory[:self.memory_size]

            if len(self.memory) > 1:
                memory_best_position = self.memory[0][0]
                for i in range(self.population_size):
                    positions[i] += np.random.rand() * (memory_best_position - positions[i])

            if self.adaptive_learning:
                avg_score = np.mean(personal_best_scores)
                self.mutation_rate = max(0.01, 0.1 * (avg_score / (global_best_score + 1e-8)))

        return global_best_position