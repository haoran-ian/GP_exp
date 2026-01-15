import numpy as np

class EAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        num_particles = 20
        w_max = 0.9
        w_min = 0.4
        c1 = 2.05
        c2 = 2.05
        
        positions = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf
        neighborhood_best_positions = np.copy(personal_best_positions)

        num_evaluations = 0

        def adaptive_inertia(iter_fraction):
            return w_max - (w_max - w_min) * iter_fraction

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

            w = adaptive_inertia(iter_fraction)

            for i in range(num_particles):
                # Neighborhood best - considering adjacent particles
                left_neighbor = personal_best_positions[i - 1] if i > 0 else personal_best_positions[-1]
                right_neighbor = personal_best_positions[(i + 1) % num_particles]
                neighborhood_best_position = min([personal_best_positions[i], left_neighbor, right_neighbor], 
                                                 key=lambda pos: func(pos))
                neighborhood_best_positions[i] = neighborhood_best_position

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                r3 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                neighborhood_influence = c2 * r3 * (neighborhood_best_positions[i] - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social + neighborhood_influence
                positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], lb, ub)

        return global_best_position, global_best_score