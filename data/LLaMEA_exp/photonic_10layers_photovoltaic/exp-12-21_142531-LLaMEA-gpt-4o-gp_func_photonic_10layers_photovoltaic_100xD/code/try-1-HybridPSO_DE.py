import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.5  # inertia weight
        self.diff_weight = 0.8  # differential weight for DE
        self.cross_prob = 0.9  # crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize swarm
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(swarm)
        personal_best_scores = np.array([func(p) for p in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            self.w = 0.9 - 0.5 * (evaluations / self.budget)  # Dynamically adjust inertia weight
            # Particle Swarm Optimization step
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (
                    self.w * velocity[i] +
                    self.c1 * r1 * (personal_best[i] - swarm[i]) +
                    self.c2 * r2 * (global_best - swarm[i])
                )
                swarm[i] += velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
          
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_scores[i]:
                    personal_best_scores[i] = f_value
                    personal_best[i] = swarm[i]
                    if f_value < global_best_score:
                        global_best_score = f_value
                        global_best = swarm[i]

            # Differential Evolution step
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.diff_weight * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cross_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, swarm[i])
                f_trial = func(trial)
                evaluations += 1
                if f_trial < personal_best_scores[i]:
                    personal_best_scores[i] = f_trial
                    personal_best[i] = trial
                    if f_trial < global_best_score:
                        global_best_score = f_trial
                        global_best = trial

        return global_best