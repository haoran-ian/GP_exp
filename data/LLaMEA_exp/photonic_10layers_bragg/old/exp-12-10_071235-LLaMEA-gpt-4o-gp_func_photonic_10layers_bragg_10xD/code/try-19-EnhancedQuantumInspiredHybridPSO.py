import numpy as np

class EnhancedQuantumInspiredHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
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

                # Adaptive inertia weight
                self.inertia_weight = max(0.4, 0.4 + 0.5 * (global_best_score - personal_best_scores[i]) / global_best_score)
                
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                    + self.social_coefficient * r2 * (global_best_position - population[i])
                )

                population[i] += velocities[i]

                # Self-learning mutation based on experience
                if np.random.rand() < self.mutation_coefficient:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    population[i] += mutation_vector * (personal_best_scores[i] - global_best_score)

                # Enhanced quantum tunneling
                if np.random.rand() < self.quantum_coefficient:
                    quantum_tunnel = np.random.normal(0, 1, self.dim)
                    population[i] = global_best_position + quantum_tunnel * self.quantum_coefficient

                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

            # Dynamic population adjustment
            if evaluations % (self.budget // 10) == 0:
                new_population_size = max(10, int(self.initial_population_size * (global_best_score / np.mean(personal_best_scores))))
                if new_population_size != self.population_size:
                    population = np.resize(population, (new_population_size, self.dim))
                    velocities = np.resize(velocities, (new_population_size, self.dim))
                    personal_best_positions = np.resize(personal_best_positions, (new_population_size, self.dim))
                    personal_best_scores = np.resize(personal_best_scores, new_population_size)
                    self.population_size = new_population_size

        return global_best_position