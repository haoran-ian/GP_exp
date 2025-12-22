import numpy as np

class AdaptiveMultiSwarmHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        num_swarms = 3
        particles_per_swarm = 10
        total_particles = num_swarms * particles_per_swarm
        inertia_weight = 0.9
        cognitive_component = 1.5
        social_component = 1.5
        differential_weight = 0.8
        crossover_rate = 0.9

        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize all particles for each swarm
        all_particles = np.random.uniform(lb, ub, (total_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (total_particles, self.dim))
        personal_best_positions = np.copy(all_particles)
        personal_best_scores = np.array([float('inf')] * total_particles)

        # Global best initialization per swarm
        global_best_positions = np.copy(personal_best_positions[:num_swarms])
        global_best_scores = np.array([float('inf')] * num_swarms)

        eval_count = 0

        while eval_count < self.budget:
            for swarm_index in range(num_swarms):
                start_index = swarm_index * particles_per_swarm
                end_index = start_index + particles_per_swarm
                swarm_particles = all_particles[start_index:end_index]
                swarm_velocities = velocities[start_index:end_index]
                swarm_personal_best_positions = personal_best_positions[start_index:end_index]
                swarm_personal_best_scores = personal_best_scores[start_index:end_index]

                # Evaluate all particles in the current swarm
                for i in range(particles_per_swarm):
                    score = func(swarm_particles[i])
                    eval_count += 1

                    if score < swarm_personal_best_scores[i]:
                        swarm_personal_best_scores[i] = score
                        swarm_personal_best_positions[i] = swarm_particles[i]

                    if score < global_best_scores[swarm_index]:
                        global_best_scores[swarm_index] = score
                        global_best_positions[swarm_index] = swarm_particles[i]

                    if eval_count >= self.budget:
                        break

                # Update velocities and positions using PSO with adaptive inertia
                inertia_weight = 0.9 - (0.5 * eval_count / self.budget)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_velocities = (inertia_weight * swarm_velocities +
                                    cognitive_component * r1 * (swarm_personal_best_positions - swarm_particles) +
                                    social_component * r2 * (global_best_positions[swarm_index] - swarm_particles))
                swarm_particles += swarm_velocities
                swarm_particles = np.clip(swarm_particles, lb, ub)

                # Apply adaptive DE mutation and crossover to current swarm
                for i in range(particles_per_swarm):
                    if eval_count >= self.budget:
                        break
                    indices = list(range(particles_per_swarm))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False) + start_index
                    adaptive_differential_weight = differential_weight * (1 - eval_count / self.budget)
                    mutant_vector = personal_best_positions[a] + adaptive_differential_weight * (personal_best_positions[b] - personal_best_positions[c])
                    cross_points = np.random.rand(self.dim) < crossover_rate
                    trial_vector = np.where(cross_points, mutant_vector, swarm_particles[i])
                    trial_vector = np.clip(trial_vector, lb, ub)

                    trial_score = func(trial_vector)
                    eval_count += 1

                    if trial_score < swarm_personal_best_scores[i]:
                        swarm_personal_best_scores[i] = trial_score
                        swarm_personal_best_positions[i] = trial_vector

                    if trial_score < global_best_scores[swarm_index]:
                        global_best_scores[swarm_index] = trial_score
                        global_best_positions[swarm_index] = trial_vector

                # Update global best positions and scores
                personal_best_positions[start_index:end_index] = swarm_personal_best_positions
                personal_best_scores[start_index:end_index] = swarm_personal_best_scores
                all_particles[start_index:end_index] = swarm_particles
                velocities[start_index:end_index] = swarm_velocities

        # Find the best global position and score across all swarms
        best_global_index = np.argmin(global_best_scores)
        best_global_position = global_best_positions[best_global_index]
        best_global_score = global_best_scores[best_global_index]

        return best_global_position, best_global_score