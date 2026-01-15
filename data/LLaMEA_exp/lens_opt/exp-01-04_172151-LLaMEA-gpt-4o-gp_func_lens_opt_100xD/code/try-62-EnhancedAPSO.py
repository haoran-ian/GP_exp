import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        num_particles = 20
        w_max = 0.9
        w_min = 0.4
        c1_max = 2.5
        c1_min = 0.5
        c2_max = 2.5
        c2_min = 0.5

        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf

        num_evaluations = 0

        def update_parameters(diversity, iter_fraction):
            w = w_min + (w_max - w_min) * (1 - iter_fraction**2)  # Smooth inertia reduction
            c1 = c1_min + (c1_max - c1_min) * (1 - diversity)
            c2 = c2_max - (c2_max - c2_min) * (1 - diversity)
            return w, c1, c2

        def find_local_best(i):
            # Neighborhood concept: consider neighbors in a simple ring topology
            left_neighbor = personal_best_positions[i - 1 if i > 0 else num_particles - 1]
            right_neighbor = personal_best_positions[i + 1 if i < num_particles - 1 else 0]
            neighbors = [left_neighbor, right_neighbor]
            best_neighbor = min(neighbors, key=lambda pos: func(pos))
            return best_neighbor

        while num_evaluations < self.budget:
            iter_fraction = num_evaluations / self.budget

            for i in range(num_particles):
                score = func(positions[i])
                num_evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            diversity = np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1))
            w, c1, c2 = update_parameters(diversity, iter_fraction)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                
                local_best_position = find_local_best(i)
                local_influence = c2 * np.random.rand(self.dim) * (local_best_position - positions[i])

                velocities[i] = w * velocities[i] + cognitive + social + local_influence
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

        return global_best_position, global_best_score