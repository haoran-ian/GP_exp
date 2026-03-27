import numpy as np

class HybridAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = min(50, budget)
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

        positions = lb + np.random.rand(self.initial_swarm_size, self.dim) * search_range
        velocities = np.random.uniform(-velocity_range, velocity_range, (self.initial_swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.initial_swarm_size, np.inf)

        global_best_score = np.inf
        global_best_position = None

        evaluations = 0
        perturbation_decay = 0.9
        perturbation_factor = 0.05  # Initial perturbation factor

        while evaluations < self.budget:
            current_swarm_size = min(self.initial_swarm_size, self.budget - evaluations)
            scores = np.array([func[0](pos) for pos in positions[:current_swarm_size]])
            evaluations += current_swarm_size

            better_mask = scores < personal_best_scores[:current_swarm_size]
            personal_best_scores[:current_swarm_size] = np.where(better_mask, scores, personal_best_scores[:current_swarm_size])
            personal_best_positions[:current_swarm_size] = np.where(better_mask[:, np.newaxis], positions[:current_swarm_size], personal_best_positions[:current_swarm_size])

            min_score = np.min(scores)
            if min_score < global_best_score:
                global_best_score = min_score
                global_best_position = positions[np.argmin(scores)]

            # Adaptive inertia weight
            inertia_weight = self.inertia_weight_final + \
                             (self.inertia_weight_initial - self.inertia_weight_final) * \
                             ((self.budget - evaluations) / self.budget)

            r1 = np.random.rand(current_swarm_size, self.dim)
            r2 = np.random.rand(current_swarm_size, self.dim)
            velocities[:current_swarm_size] = (
                inertia_weight * velocities[:current_swarm_size] +
                self.cognitive_factor * r1 * (personal_best_positions[:current_swarm_size] - positions[:current_swarm_size]) +
                self.social_factor * r2 * (global_best_position - positions[:current_swarm_size])
            )

            velocities[:current_swarm_size] = np.clip(velocities[:current_swarm_size], -velocity_range, velocity_range)

            # Periodic velocity perturbation with decay
            if evaluations % (self.budget // 5) == 0:
                velocities[:current_swarm_size] += perturbation_factor * np.random.uniform(-velocity_range, velocity_range, (current_swarm_size, self.dim))
                perturbation_factor *= perturbation_decay

            positions[:current_swarm_size] = positions[:current_swarm_size] + velocities[:current_swarm_size]
            positions[:current_swarm_size] = np.clip(positions[:current_swarm_size], lb, ub)

            # Enhanced local search for the global best with differential mutation
            if evaluations % (self.budget // 10) == 0:
                for _ in range(5):  # Perform multiple local search attempts
                    local_search_perturbation = np.random.uniform(-0.05, 0.05, self.dim) * velocity_range
                    mutant = global_best_position + local_search_perturbation
                    base_vector = positions[np.random.randint(0, current_swarm_size)]
                    diff_vector = positions[np.random.randint(0, current_swarm_size)] - positions[np.random.randint(0, current_swarm_size)]
                    differential_perturbation = 0.5 * diff_vector
                    new_global_position = mutant + differential_perturbation
                    new_global_position = np.clip(new_global_position, lb, ub)
                    new_global_score = func[0](new_global_position)
                    evaluations += 1
                    if new_global_score < global_best_score:
                        global_best_score = new_global_score
                        global_best_position = new_global_position

        return global_best_position, global_best_score