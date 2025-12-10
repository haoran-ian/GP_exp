import numpy as np

class AdaptiveChaoticParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(40, budget // 3)
        self.w = 0.5 + np.random.rand() / 2  # Inertia weight
        self.c1 = 1.5 + np.random.rand()  # Cognitive coefficient
        self.c2 = 1.5 + np.random.rand()  # Social coefficient
        self.func_evals = 0

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best = swarm.copy()
        personal_best_scores = np.array([func(x) for x in personal_best])
        self.func_evals += self.swarm_size

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = personal_best_scores.min()

        while self.func_evals < self.budget:
            chaotic_sequence = 0.7 * np.random.rand() * (1.0 - np.random.rand(self.swarm_size, self.dim) ** 2)
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)

            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best - swarm) +
                          self.c2 * r2 * (global_best - swarm) +
                          chaotic_sequence)

            swarm = swarm + velocities
            swarm = np.clip(swarm, lb, ub)

            scores = np.array([func(x) for x in swarm])
            self.func_evals += self.swarm_size

            for i in range(self.swarm_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best[i] = swarm[i]
                    personal_best_scores[i] = scores[i]

            min_idx = personal_best_scores.argmin()
            if personal_best_scores[min_idx] < global_best_score:
                global_best = personal_best[min_idx]
                global_best_score = personal_best_scores[min_idx]

        return global_best