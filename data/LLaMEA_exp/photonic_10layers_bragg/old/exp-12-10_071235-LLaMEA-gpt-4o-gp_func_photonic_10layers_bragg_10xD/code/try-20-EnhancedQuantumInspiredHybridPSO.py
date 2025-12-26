import numpy as np

class EnhancedQuantumInspiredHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.mutation_coefficient = 0.1
        self.quantum_coefficient = 0.1
        self.lower_bound = None
        self.upper_bound = None

    def __call__(self, func):
        self.lower_bound = np.array(func.bounds.lb)
        self.upper_bound = np.array(func.bounds.ub)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(population[i])
                evaluations += 1

                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = population[i]

                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best_position = population[i]

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Nonlinear inertia weight reduction over iterations
                current_inertia_weight = self.initial_inertia_weight - (
                    (self.initial_inertia_weight - self.final_inertia_weight) * (evaluations / self.budget)
                )

                velocities[i] = (
                    current_inertia_weight * velocities[i]
                    + self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                    + self.social_coefficient * r2 * (global_best_position - population[i])
                )

                population[i] += velocities[i]

                # Adaptive mutation based on dynamic neighborhood
                neighborhood_size = max(1, int(self.population_size * 0.1))
                neighborhood_indices = np.random.choice(self.population_size, neighborhood_size, replace=False)
                neighborhood_best_position = min(neighborhood_indices, key=lambda idx: personal_best_scores[idx])

                mutation_probability = self.mutation_coefficient * (
                    np.linalg.norm(population[i] - personal_best_positions[neighborhood_best_position])
                    / np.linalg.norm(self.upper_bound - self.lower_bound)
                )
                
                if np.random.rand() < mutation_probability:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    population[i] += mutation_vector * mutation_probability

                # Adaptive quantum tunneling for enhanced exploration
                adaptive_quantum_coefficient = self.quantum_coefficient * (1 - evaluations / self.budget)
                if np.random.rand() < adaptive_quantum_coefficient:
                    quantum_tunnel = np.random.normal(0, 1, self.dim)
                    population[i] = global_best_position + quantum_tunnel * adaptive_quantum_coefficient

                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

        return global_best_position