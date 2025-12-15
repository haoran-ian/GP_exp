import numpy as np

class AdaptiveMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        num_particles = 30
        inertia_weight_initial = 0.9
        inertia_weight_final = 0.4
        cognitive_component_initial = 2.0
        cognitive_component_final = 1.0
        social_component_initial = 1.0
        social_component_final = 2.0
        differential_weight_initial = 0.9
        differential_weight_final = 0.5
        crossover_rate_initial = 0.9
        crossover_rate_final = 0.6

        # Initialize particles
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([float('inf')] * num_particles)
        
        # Global best and elite archive initialization
        global_best_position = None
        global_best_score = float('inf')
        elite_archive = []

        eval_count = 0

        while eval_count < self.budget:
            for i in range(num_particles):
                score = func(particles[i])
                eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]
                    if len(elite_archive) < num_particles:
                        elite_archive.append((particles[i], score))
                    else:
                        worst_in_elite = max(elite_archive, key=lambda x: x[1])
                        if score < worst_in_elite[1]:
                            elite_archive.remove(worst_in_elite)
                            elite_archive.append((particles[i], score))

                if eval_count >= self.budget:
                    break

            # Dynamically adjust parameters
            inertia_weight = inertia_weight_initial - (inertia_weight_initial - inertia_weight_final) * (eval_count / self.budget)
            cognitive_component = cognitive_component_initial - (cognitive_component_initial - cognitive_component_final) * (eval_count / self.budget)
            social_component = social_component_initial + (social_component_final - social_component_initial) * (eval_count / self.budget)
            differential_weight = differential_weight_initial - (differential_weight_initial - differential_weight_final) * (eval_count / self.budget)
            crossover_rate = crossover_rate_initial - (crossover_rate_initial - crossover_rate_final) * (eval_count / self.budget)

            # Update velocities and positions using PSO with adaptive parameters
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (inertia_weight * velocities +
                          cognitive_component * r1 * (personal_best_positions - particles) +
                          social_component * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, lb, ub)

            # Apply DE mutation and crossover with adaptive parameters
            for i in range(num_particles):
                if eval_count >= self.budget:
                    break
                indices = list(range(num_particles))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = personal_best_positions[a] + differential_weight * (personal_best_positions[b] - personal_best_positions[c])
                cross_points = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(cross_points, mutant_vector, particles[i])
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial_vector
                    if len(elite_archive) < num_particles:
                        elite_archive.append((trial_vector, trial_score))
                    else:
                        worst_in_elite = max(elite_archive, key=lambda x: x[1])
                        if trial_score < worst_in_elite[1]:
                            elite_archive.remove(worst_in_elite)
                            elite_archive.append((trial_vector, trial_score))

        return global_best_position, global_best_score