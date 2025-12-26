import numpy as np

class AdaptiveQuantumParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.hyper_sphere_shrink_factor = 0.99
        self.init_velocity_scale = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * self.init_velocity_scale
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity and position
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                population[i] += velocities[i]

                # Quantum-inspired mutation within a shrinking hyper-sphere
                if np.random.rand() < 0.1:
                    mutation_radius = np.linalg.norm(global_best_position - population.mean(axis=0)) * self.hyper_sphere_shrink_factor
                    mutation_vector = np.random.randn(self.dim)
                    mutation_vector /= np.linalg.norm(mutation_vector)
                    population[i] = global_best_position + mutation_radius * mutation_vector

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

            # Adaptive inertia weight to improve convergence
            self.w = 0.9 - 0.7 * (evaluations / self.budget)
        
        return global_best_score, global_best_position