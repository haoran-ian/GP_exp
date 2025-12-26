import numpy as np

class AdaptiveParticleSwarmGCM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_scale = 0.1
        self.covariance_inflation = 1e-3
        self.mutation_probability = 0.2  # Increased mutation probability

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
            # Calculate population covariance for mutation adaptation
            cov_matrix = np.cov(population, rowvar=False) + self.covariance_inflation * np.eye(self.dim)
            
            # Gradient approximation step for better direction
            gradients = np.zeros_like(population)
            for i, individual in enumerate(population):
                perturbed = individual + self.velocity_scale * velocities[i]
                gradients[i] = (func(perturbed) - personal_best_scores[i]) / self.velocity_scale

            for i in range(self.population_size):
                # Update velocity and position using gradient information
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                    - 0.01 * gradients[i]  # Incorporate gradient to adjust search direction
                )
                population[i] += velocities[i] * self.velocity_scale

                # Covariance-based mutation step with adaptive probability
                if np.random.rand() < self.mutation_probability:
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

                # Adjust velocity scale dynamically based on convergence
                self.velocity_scale = 0.1 + 0.9 * (evaluations / self.budget)

        return global_best_score, global_best_position