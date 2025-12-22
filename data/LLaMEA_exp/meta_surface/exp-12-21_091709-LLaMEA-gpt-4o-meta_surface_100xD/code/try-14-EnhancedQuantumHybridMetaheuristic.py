import numpy as np

class EnhancedQuantumHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = 50
        F = 0.8
        CR = 0.9
        positions = np.random.rand(pop_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(pop_size, self.dim) * 0.1
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(pop_size, np.inf)

        global_best_score = np.inf
        global_best_position = np.zeros(self.dim)

        eval_count = 0

        def quantum_variance(pos, global_best):
            beta = np.random.rand(self.dim)
            return pos + beta * (global_best - pos)

        def adaptive_grouping():
            group_size = np.random.randint(3, 6)
            return np.array_split(np.random.permutation(pop_size), group_size)

        def cooperative_update(group, global_best):
            for i in group:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (0.5 * velocities[i] +
                                 2 * r1 * (personal_best_positions[i] - positions[i]) +
                                 2 * r2 * (global_best - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

        while eval_count < self.budget:
            for i in range(pop_size):
                score = func(positions[i])
                eval_count += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()
                if eval_count >= self.budget:
                    break

            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                F = 0.5 + np.random.rand() * 0.5
                CR = 0.5 + np.random.rand() * 0.5
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(positions[a] + F * (positions[b] - positions[c]), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, positions[i])
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial.copy()
                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial.copy()

            grouped_indices = adaptive_grouping()
            for group in grouped_indices:
                cooperative_update(group, global_best_position)
                if np.random.rand() < 0.1:
                    for i in group:
                        positions[i] = quantum_variance(positions[i], global_best_position)

        return {'best_position': global_best_position, 'best_score': global_best_score}