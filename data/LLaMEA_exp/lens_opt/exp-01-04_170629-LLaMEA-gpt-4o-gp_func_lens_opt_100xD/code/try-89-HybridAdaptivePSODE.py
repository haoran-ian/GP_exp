import numpy as np

class HybridAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.gaussian_scale = 0.1
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        # Initialize positions and velocities
        positions = np.random.uniform(lower_bound, upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocities with Gaussian perturbation
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])
                gaussian_perturbation = self.gaussian_scale * np.random.normal(0, 1, self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_velocity +
                                 social_velocity +
                                 gaussian_perturbation)

                # Differential Evolution mutation
                idxs = np.random.choice(self.num_particles, 3, replace=False)
                mutant_vector = positions[idxs[0]] + self.mutation_factor * (positions[idxs[1]] - positions[idxs[2]])
                mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, positions[i])

                # Evaluate trial vector
                trial_score = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial_vector

                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)

                # Stop if budget is exhausted
                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score