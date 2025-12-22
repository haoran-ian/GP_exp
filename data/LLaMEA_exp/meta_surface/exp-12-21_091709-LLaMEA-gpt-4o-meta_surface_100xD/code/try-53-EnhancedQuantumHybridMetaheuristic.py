import numpy as np

class EnhancedQuantumHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = 50
        F_min, F_max = 0.5, 1.0
        CR_min, CR_max = 0.5, 1.0
        positions = np.random.rand(pop_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(pop_size, self.dim) * 0.1
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(pop_size, np.inf)

        global_best_score = np.inf
        global_best_position = np.zeros(self.dim)

        eval_count = 0

        def adaptive_quantum_variance(pos, global_best, iter, max_iter):
            beta = np.random.rand(self.dim)
            variance_factor = np.exp(-0.5 * (iter / max_iter))
            return pos + variance_factor * beta * (global_best - pos)

        while eval_count < self.budget:
            # Evaluate current population
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

            # Differential Evolution step with self-adaptive control parameters
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                F = F_min + np.random.rand() * (F_max - F_min) * (1 - (eval_count / self.budget))
                CR = CR_min + np.random.rand() * (CR_max - CR_min)
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

            # Particle Swarm step with adaptive quantum variance
            inertia_weight = 0.5 + 0.5 * np.random.rand()
            momentum = 0.1  # New momentum term
            for i in range(pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (momentum * velocities[i] +  # Incorporate momentum
                                 inertia_weight * velocities[i] +
                                 2 * r1 * (personal_best_positions[i] - positions[i]) +
                                 2 * r2 * (global_best_position - positions[i]))

                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
                # Apply adaptive quantum variance
                if np.random.rand() < 0.1:
                    iter_fraction = eval_count / self.budget
                    positions[i] = adaptive_quantum_variance(positions[i], global_best_position, eval_count, self.budget)

        return {'best_position': global_best_position, 'best_score': global_best_score}