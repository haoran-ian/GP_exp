import numpy as np

class APSO:
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
            w = w_max - (w_max - w_min) * iter_fraction
            c1 = c1_max - (c1_max - c1_min) * diversity
            c2 = c2_min + (c2_max - c2_min) * diversity
            return w, c1, c2
        
        def fuzzy_logic_control(score_diff, diversity):
            # Example fuzzy logic for parameter adjustment
            if score_diff < 0.1 and diversity < 0.1:
                return 0.5, 0.9
            elif score_diff > 0.5:
                return 0.9, 0.5
            else:
                return 0.7, 0.7

        def crowding_distance(positions):
            # Calculating crowding distance for maintaining diversity
            sorted_indices = np.argsort(positions, axis=0)
            distances = np.zeros((positions.shape[0], self.dim))
            
            for d in range(self.dim):
                sorted_pos = positions[sorted_indices[:, d], d]
                distances[sorted_indices[0, d], d] = np.inf
                distances[sorted_indices[-1, d], d] = np.inf
                
                for i in range(1, positions.shape[0] - 1):
                    distances[sorted_indices[i, d], d] = (sorted_pos[i+1] - sorted_pos[i-1]) / (np.max(sorted_pos) - np.min(sorted_pos))
                    
            return np.sum(distances, axis=1)

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

            diversity = np.mean(np.std(positions, axis=0))
            w, c1, c2 = update_parameters(diversity, iter_fraction)

            crowding_dist = crowding_distance(positions)
            score_diff = np.max(personal_best_scores) - np.min(personal_best_scores)
            w, c1 = fuzzy_logic_control(score_diff, diversity)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], lb, ub)

        return global_best_position, global_best_score