import numpy as np

class EnhancedDynamicParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_factor = 1.5
        self.social_factor = 1.5
        self.vel_range_factor = 0.1

    def __call__(self, func):
        lb = np.array(func[0].bounds.lb)
        ub = np.array(func[0].bounds.ub)
        search_range = ub - lb
        velocity_range = self.vel_range_factor * search_range

        positions = lb + np.random.rand(self.swarm_size, self.dim) * search_range
        velocities = np.random.uniform(-velocity_range, velocity_range, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)

        global_best_score = np.inf
        global_best_position = None

        evaluations = 0
        perturbation_decay = 0.9
        perturbation_factor = 0.05  # Initial perturbation factor

        while evaluations < self.budget:
            scores = np.array([func[0](pos) for pos in positions])
            evaluations += self.swarm_size

            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], positions, personal_best_positions)

            min_score = np.min(scores)
            if min_score < global_best_score:
                global_best_score = min_score
                global_best_position = positions[np.argmin(scores)]

            # Adaptive inertia weight
            inertia_weight = self.inertia_weight_final + \
                             (self.inertia_weight_initial - self.inertia_weight_final) * \
                             ((self.budget - evaluations) / self.budget)

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = (
                inertia_weight * velocities +
                (self.cognitive_factor + 0.5) * r1 * (personal_best_positions - positions) +  # Slight increase in acceleration
                self.social_factor * r2 * (global_best_position - positions)
            )

            velocities = np.clip(velocities, -velocity_range, velocity_range)

            # Periodic velocity perturbation with decay
            if evaluations % (self.budget // 5) == 0:
                velocities += perturbation_factor * np.random.uniform(-velocity_range, velocity_range, (self.swarm_size, self.dim))
                perturbation_factor *= perturbation_decay

            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            # Enhanced local search for the global best
            if evaluations % (self.budget // 10) == 0:
                for _ in range(5):  # Perform multiple local search attempts
                    local_search_perturbation = np.random.uniform(-0.05, 0.05, self.dim) * velocity_range
                    new_global_position = global_best_position + local_search_perturbation
                    new_global_position = np.clip(new_global_position, lb, ub)
                    new_global_score = func[0](new_global_position)
                    evaluations += 1
                    if new_global_score < global_best_score:
                        global_best_score = new_global_score
                        global_best_position = new_global_position

        return global_best_position, global_best_score