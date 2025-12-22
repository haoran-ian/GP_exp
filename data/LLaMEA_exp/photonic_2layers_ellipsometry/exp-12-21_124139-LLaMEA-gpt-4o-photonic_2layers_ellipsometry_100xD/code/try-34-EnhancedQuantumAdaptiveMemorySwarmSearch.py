import numpy as np

class EnhancedQuantumAdaptiveMemorySwarmSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.mutation_rate = 0.1
        self.memory_size = 5
        self.memory = []
        self.inertia_weight_decay = (self.initial_inertia_weight - self.final_inertia_weight) / self.budget
        self.quantum_step = 0.05
        self.diversity_threshold = 1e-5

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

                inertia_weight = self.initial_inertia_weight - (self.evaluations / self.budget) * (self.initial_inertia_weight - self.final_inertia_weight)
                cognitive_coeff_adaptive = self.cognitive_coeff * (1 - self.evaluations / self.budget)
                social_coeff_adaptive = self.social_coeff * (self.evaluations / self.budget)
                
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + cognitive_coeff_adaptive * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                    + social_coeff_adaptive * np.random.rand(self.dim) * (global_best_position - positions[i])
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

            # Quantum-inspired position update
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
                    if np.std(positions) < self.diversity_threshold:
                        positions[i] += np.random.rand() * (memory_best_position - positions[i])

            self.mutation_rate = max(0.01, 0.1 - 0.09 * (self.evaluations / self.budget))

        return global_best_position