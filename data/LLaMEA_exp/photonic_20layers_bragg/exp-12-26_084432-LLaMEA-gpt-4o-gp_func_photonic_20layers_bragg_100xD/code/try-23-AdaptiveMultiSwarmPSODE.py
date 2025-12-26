import numpy as np
from scipy.optimize import minimize

class AdaptiveMultiSwarmPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_component = 2.0
        self.social_component = 2.0
        self.f = 0.5
        self.cr = 0.9
        self.num_swarms = 3

    def __call__(self, func):
        np.random.seed(42)
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        swarm_size = self.population_size // self.num_swarms

        # Initialize particles
        positions = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), 
                                       (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive inertia weight with diversity
            diversity = np.mean(np.std(positions, axis=0))
            self.inertia_weight = self.inertia_weight_min + \
                                  (0.9 - self.inertia_weight_min) * (1 - evaluations / self.budget) * diversity

            # Multi-Swarm Particle Swarm Optimization update
            for k in range(self.num_swarms):
                swarm_indices = range(k * swarm_size, (k + 1) * swarm_size)
                r1, r2 = np.random.rand(2, swarm_size, self.dim)
                velocities[swarm_indices] = (self.inertia_weight * velocities[swarm_indices] +
                                             self.cognitive_component * r1 * (personal_best_positions[swarm_indices] - positions[swarm_indices]) +
                                             self.social_component * r2 * (global_best_position - positions[swarm_indices]))
                positions[swarm_indices] = positions[swarm_indices] + velocities[swarm_indices]
                positions[swarm_indices] = np.clip(positions[swarm_indices], lower_bound, upper_bound)

            # Evaluate new positions
            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                    if scores[i] < global_best_score:
                        global_best_score = scores[i]
                        global_best_position = positions[i]

            # Differential Evolution mutation and crossover with adaptive scaling
            self.f = 0.5 + 0.3 * (1 - evaluations / self.budget)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = positions[indices]
                mutant = np.clip(x1 + self.f * (x2 - x3), lower_bound, upper_bound)
                diversity_factor = np.mean(np.std(positions, axis=0))
                self.cr = 0.5 + 0.4 * diversity_factor
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, positions[i])
                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    positions[i] = trial
                    scores[i] = trial_score
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial

            # Local Search for best particles
            if evaluations < self.budget:
                best_idx = np.argmin(scores)
                res = minimize(func, positions[best_idx], bounds=[(lb, ub) for lb, ub in zip(lower_bound, upper_bound)], method='L-BFGS-B')
                if res.fun < global_best_score:
                    global_best_score = res.fun
                    global_best_position = res.x
                evaluations += res.nfev

        return global_best_position, global_best_score