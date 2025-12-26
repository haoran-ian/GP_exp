import numpy as np

class AdaptiveQuantumPSODCM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.velocity_scale = 0.1
        self.covariance_inflation = 1e-3
        self.qdelta = 0.001  # Quantum delta for position update

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * self.velocity_scale
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Calculate adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

            # Calculate population covariance for mutation adaptation
            cov_matrix = np.cov(population, rowvar=False) + self.covariance_inflation * np.eye(self.dim)

            for i in range(self.population_size):
                # Quantum position update
                if np.random.rand() < 0.5:
                    quantum_jump = np.random.uniform(-self.qdelta, self.qdelta, self.dim)
                    quantum_position = population[i] + quantum_jump
                    quantum_position = np.clip(quantum_position, lb, ub)
                    quantum_score = func(quantum_position)
                    evaluations += 1

                    # Accept quantum move if better
                    if quantum_score < personal_best_scores[i]:
                        population[i] = quantum_position
                        personal_best_scores[i] = quantum_score
                        personal_best_positions[i] = quantum_position
                        if quantum_score < global_best_score:
                            global_best_score = quantum_score
                            global_best_position = quantum_position

                # Update velocity and position
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                population[i] += velocities[i] * self.velocity_scale

                # Covariance-based mutation step
                if np.random.rand() < 0.1:
                    mutation_vector = np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                    population[i] += mutation_vector

                # Ensure within bounds
                population[i] = np.clip(population[i], lb, ub)

                # Evaluate new position
                score = func(population[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

        return global_best_score, global_best_position