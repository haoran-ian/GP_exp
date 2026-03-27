import numpy as np

class AdaptiveMultiSwarmParticleOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.main_swarm_size = min(30, budget // 2)
        self.sub_swarm_size = min(20, budget // 4)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_factor = 1.5
        self.social_factor = 1.5
        self.vel_range_factor = 0.1

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        search_range = ub - lb
        velocity_range = self.vel_range_factor * search_range

        main_positions = lb + np.random.rand(self.main_swarm_size, self.dim) * search_range
        sub_positions = lb + np.random.rand(self.sub_swarm_size, self.dim) * search_range
        main_velocities = np.random.uniform(-velocity_range, velocity_range, (self.main_swarm_size, self.dim))
        sub_velocities = np.random.uniform(-velocity_range, velocity_range, (self.sub_swarm_size, self.dim))

        main_personal_best_positions = np.copy(main_positions)
        sub_personal_best_positions = np.copy(sub_positions)
        main_personal_best_scores = np.full(self.main_swarm_size, np.inf)
        sub_personal_best_scores = np.full(self.sub_swarm_size, np.inf)

        global_best_score = np.inf
        global_best_position = None

        evaluations = 0
        perturbation_decay = 0.9
        perturbation_factor = 0.05  # Initial perturbation factor

        def update_swarm(positions, velocities, personal_best_positions, personal_best_scores, swarm_size):
            scores = np.array([func(pos) for pos in positions])
            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], positions, personal_best_positions)

            min_score = np.min(scores)
            nonlocal global_best_score, global_best_position
            if min_score < global_best_score:
                global_best_score = min_score
                global_best_position = positions[np.argmin(scores)]

            inertia_weight = self.inertia_weight_final + \
                             (self.inertia_weight_initial - self.inertia_weight_final) * \
                             ((self.budget - evaluations) / self.budget)

            r1 = np.random.rand(swarm_size, self.dim)
            r2 = np.random.rand(swarm_size, self.dim)
            velocities = (
                inertia_weight * velocities +
                self.cognitive_factor * r1 * (personal_best_positions - positions) +
                self.social_factor * r2 * (global_best_position - positions)
            )
            velocities = np.clip(velocities, -velocity_range, velocity_range)
            if evaluations % (self.budget // 5) == 0:
                velocities += perturbation_factor * np.random.uniform(-velocity_range, velocity_range, (swarm_size, self.dim))
            positions += velocities
            positions = np.clip(positions, lb, ub)
            return positions, velocities, personal_best_positions, personal_best_scores, min_score

        while evaluations < self.budget:
            main_positions, main_velocities, main_personal_best_positions, main_personal_best_scores, _ = \
                update_swarm(main_positions, main_velocities, main_personal_best_positions, main_personal_best_scores, self.main_swarm_size)
            evaluations += self.main_swarm_size

            sub_positions, sub_velocities, sub_personal_best_positions, sub_personal_best_scores, sub_min_score = \
                update_swarm(sub_positions, sub_velocities, sub_personal_best_positions, sub_personal_best_scores, self.sub_swarm_size)
            evaluations += self.sub_swarm_size

            if sub_min_score < global_best_score:
                global_best_score = sub_min_score
                global_best_position = sub_positions[np.argmin(sub_personal_best_scores)]

            if evaluations % (self.budget // 10) == 0:
                for _ in range(5):
                    local_search_perturbation = np.random.uniform(-0.05, 0.05, self.dim) * velocity_range
                    new_global_position = global_best_position + local_search_perturbation
                    new_global_position = np.clip(new_global_position, lb, ub)
                    new_global_score = func(new_global_position)
                    evaluations += 1
                    if new_global_score < global_best_score:
                        global_best_score = new_global_score
                        global_best_position = new_global_position

        return global_best_position, global_best_score