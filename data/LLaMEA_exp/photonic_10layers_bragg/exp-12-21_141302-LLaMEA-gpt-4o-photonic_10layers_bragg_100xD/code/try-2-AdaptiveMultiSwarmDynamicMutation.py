import numpy as np

class AdaptiveMultiSwarmDynamicMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_count = 3  # Number of sub-swarms
        self.swarm_size = 10 * dim  # Heuristic size for each swarm
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.c3 = 0.5  # Communication coefficient between swarms
        self.mutation_rate = 0.1  # Initial mutation rate
        self.velocity_scale = 0.1  # Initial velocity scale factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        total_population_size = self.swarm_count * self.swarm_size
        population = np.random.uniform(lb, ub, (total_population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (total_population_size, self.dim)) * self.velocity_scale
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = np.copy(personal_best_positions[np.argmin(personal_best_scores)])
        global_best_score = np.min(personal_best_scores)

        evaluations = total_population_size

        while evaluations < self.budget:
            for swarm_id in range(self.swarm_count):
                start_idx = swarm_id * self.swarm_size
                end_idx = start_idx + self.swarm_size
                swarm_population = population[start_idx:end_idx]
                swarm_velocities = velocities[start_idx:end_idx]
                swarm_personal_best_positions = personal_best_positions[start_idx:end_idx]
                swarm_personal_best_scores = personal_best_scores[start_idx:end_idx]
                swarm_global_best_index = np.argmin(swarm_personal_best_scores)
                swarm_global_best_position = np.copy(swarm_personal_best_positions[swarm_global_best_index])

                for i in range(self.swarm_size):
                    velocities[start_idx + i] = (
                        self.w * swarm_velocities[i]
                        + self.c1 * np.random.rand(self.dim) * (swarm_personal_best_positions[i] - swarm_population[i])
                        + self.c2 * np.random.rand(self.dim) * (swarm_global_best_position - swarm_population[i])
                        + self.c3 * np.random.rand(self.dim) * (global_best_position - swarm_population[i])
                    )
                    population[start_idx + i] += velocities[start_idx + i] * self.velocity_scale

                    # Mutation step: Adaptive mutation rate based on proximity to global best
                    if np.random.rand() < self.mutation_rate:
                        mutation_vector = np.random.normal(0, (ub - lb) * 0.1, self.dim)
                        population[start_idx + i] += mutation_vector

                    # Ensure within bounds
                    population[start_idx + i] = np.clip(population[start_idx + i], lb, ub)

                    # Evaluate new position
                    score = func(population[start_idx + i])
                    evaluations += 1

                    # Update personal and global bests
                    if score < swarm_personal_best_scores[i]:
                        swarm_personal_best_scores[i] = score
                        swarm_personal_best_positions[i] = population[start_idx + i]
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = population[start_idx + i]

                # Adjust mutation rate and velocity scale dynamically per swarm
                diversity = np.mean(np.linalg.norm(swarm_population - swarm_global_best_position, axis=1))
                self.mutation_rate = 0.05 + 0.45 * (1 - evaluations / self.budget) * (diversity / np.linalg.norm(ub - lb))
                self.velocity_scale = 0.1 + 0.9 * (evaluations / self.budget)

        return global_best_score, global_best_position