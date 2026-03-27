import numpy as np

class AdaptiveDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.gaussian_scale = 0.1
        self.topology_change_interval = 50  # How often to change the topology

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        positions = np.random.uniform(lower_bound, upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.num_particles
        iteration = 0

        while evaluations < self.budget:
            if iteration % self.topology_change_interval == 0:
                # Randomly connect particles to form a new topology
                neighbors = np.random.choice(self.num_particles, (self.num_particles, 3), replace=False)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                
                # Use local best based on dynamic topology
                local_best_position = min(neighbors[i], key=lambda n: personal_best_scores[n])
                social_velocity = self.social_param * r2 * (personal_best_positions[local_best_position] - positions[i])

                # Exponential decay of perturbation scale
                scaling_factor = np.exp(-iteration / self.budget)
                gaussian_perturbation = self.gaussian_scale * scaling_factor * np.random.normal(0, 1, self.dim)
                
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_velocity +
                                 social_velocity +
                                 gaussian_perturbation)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

            iteration += 1

        return global_best_position, global_best_score