import numpy as np

class AdaptiveQuantumInspiredPSO:
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
                self.inertia_weight = max(0.4, 0.9 - 0.5 * (evaluations / self.budget))
                self.cognitive_coefficient = max(1.5, 2.0 - 0.5 * (evaluations / self.budget))
                self.social_coefficient = max(1.5, 2.0 - 0.5 * (evaluations / self.budget))

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                    + self.social_coefficient * r2 * (global_best_position - population[i])
                )

                population[i] += velocities[i]

                # Adaptive mutation based on diversity and time-dependent scaling
                diversity = np.std(population, axis=0).mean()
                mutation_probability = self.mutation_coefficient * (diversity + evaluations / self.budget)
                
                if np.random.rand() < mutation_probability:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    population[i] += mutation_vector * mutation_probability

                # Adaptive quantum tunneling for enhanced exploration
                if np.random.rand() < self.quantum_coefficient:
                    adaptive_quantum_tunnel = np.random.normal(0, 1, self.dim)
                    scaling_factor = np.exp(-evaluations / (2 * self.budget))
                    population[i] = global_best_position + adaptive_quantum_tunnel * self.quantum_coefficient * scaling_factor

                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

        return global_best_position