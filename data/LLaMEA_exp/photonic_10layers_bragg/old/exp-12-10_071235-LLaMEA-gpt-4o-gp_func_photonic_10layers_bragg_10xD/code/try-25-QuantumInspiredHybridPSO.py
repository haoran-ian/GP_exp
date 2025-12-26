import numpy as np

class QuantumInspiredHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
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

                # Dynamic adjustment of parameters
                self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
                self.cognitive_coefficient = max(1.5, self.cognitive_coefficient * 0.98)
                self.social_coefficient = max(1.5, self.social_coefficient * 0.98)

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                    + self.social_coefficient * r2 * (global_best_position - population[i])
                )

                population[i] += velocities[i]

                # Adaptive mutation based on distance to global best
                distance_to_best = np.linalg.norm(population[i] - global_best_position)
                mutation_probability = self.mutation_coefficient * (distance_to_best / (np.linalg.norm(self.upper_bound - self.lower_bound)))
                
                if np.random.rand() < mutation_probability:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    population[i] += mutation_vector * mutation_probability * 0.5  # Enhanced mutation strategy

                # Adaptive quantum tunneling
                adaptive_quantum_coeff = self.quantum_coefficient * (1 - evaluations / self.budget)
                if np.random.rand() < adaptive_quantum_coeff:
                    quantum_tunnel = np.random.normal(0, 1, self.dim)
                    population[i] = global_best_position + quantum_tunnel * adaptive_quantum_coeff

                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

        return global_best_position