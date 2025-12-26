import numpy as np

class AdaptiveCooperativeSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget // 3)
        self.w = 0.7  # Increased initial inertia weight for better exploration
        self.c1 = 1.5 + np.random.rand()  # Cognitive coefficient
        self.c2 = 1.5 + np.random.rand()  # Social coefficient
        self.func_evals = 0

    def chaotic_perturbation(self, size):
        beta = 1.3  # Slightly adjusted Levy distribution parameter for more aggressive search
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, size) * sigma
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def cooperative_init(self, lb, ub):
        x = np.random.rand(self.swarm_size, self.dim)
        r = 2 + 2 * (1 - (self.func_evals / self.budget) ** 2)  # Cooperative dynamic parameter
        for _ in range(100):
            x = r * x * (1 - x)
        return lb + 0.75 * (ub - lb) * x  # Modified scaling to enhance diversity

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm = self.cooperative_init(lb, ub)
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        personal_best = swarm.copy()
        personal_best_scores = np.array([func(x) for x in personal_best])
        self.func_evals += self.swarm_size

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = personal_best_scores.min()

        while self.func_evals < self.budget:
            self.w = 0.7 - 0.5 * (self.func_evals / self.budget)  # Linear inertia weight adjustment
            self.c1 = 1.5 + np.cos(np.pi * self.func_evals / self.budget)  # Cooperative cognitive coefficient
            self.c2 = 1.5 + np.sin(np.pi * self.func_evals / self.budget)  # Cooperative social coefficient
            cooperation_factor = (1 - np.cos(2 * np.pi * self.func_evals / self.budget))
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)

            velocities = (0.9 * (self.w * velocities) +  # Added dynamic velocity scaling
                          self.c1 * r1 * (personal_best - swarm) +
                          self.c2 * r2 * (global_best - swarm))

            swarm = swarm + velocities
            perturbation = self.chaotic_perturbation((self.swarm_size, self.dim)) * np.sqrt(cooperation_factor)
            swarm += perturbation
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

            if np.random.rand() < 0.05 * cooperation_factor:
                random_indexes = np.random.choice(self.swarm_size, size=self.swarm_size // 4, replace=False)
                new_positions = self.cooperative_init(lb, ub)
                swarm[random_indexes] = new_positions[random_indexes]

        return global_best