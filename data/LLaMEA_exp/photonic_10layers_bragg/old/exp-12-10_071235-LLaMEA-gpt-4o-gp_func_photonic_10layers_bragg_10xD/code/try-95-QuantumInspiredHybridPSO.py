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
        
        # Initialize neighborhood topology
        neighborhood_size = max(3, self.population_size // 5)
        neighborhood_indices = [np.random.choice(self.population_size, neighborhood_size, replace=False)
                                for _ in range(self.population_size)]

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
                
                # Use neighborhood best instead of global best
                neighborhood_best_score = np.inf
                neighborhood_best_position = np.copy(global_best_position)
                
                for idx in neighborhood_indices[i]:
                    if personal_best_scores[idx] < neighborhood_best_score:
                        neighborhood_best_score = personal_best_scores[idx]
                        neighborhood_best_position = personal_best_positions[idx]

                # Nonlinear decay of parameters
                inertia_decay = 0.99
                self.inertia_weight = max(0.4, self.inertia_weight * inertia_decay)
                cognitive_decay = 0.99
                self.cognitive_coefficient = max(1.5, self.cognitive_coefficient * cognitive_decay)
                social_decay = 0.99
                self.social_coefficient = max(1.5, self.social_coefficient * social_decay)

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                    + self.social_coefficient * r2 * (neighborhood_best_position - population[i])
                )

                population[i] += velocities[i]

                # Adaptive mutation and quantum tunneling
                relative_improvement = (personal_best_scores[i] - fitness) / max(1e-6, personal_best_scores[i])
                mutation_probability = self.mutation_coefficient * (relative_improvement + evaluations / self.budget)
                
                if np.random.rand() < mutation_probability:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    population[i] += mutation_vector * mutation_probability

                if np.random.rand() < self.quantum_coefficient:
                    adaptive_quantum_tunnel = np.random.normal(0, 1, self.dim)
                    scaling_factor = np.exp(-evaluations / (2 * self.budget))
                    population[i] = neighborhood_best_position + adaptive_quantum_tunnel * self.quantum_coefficient * scaling_factor

                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

        return global_best_position