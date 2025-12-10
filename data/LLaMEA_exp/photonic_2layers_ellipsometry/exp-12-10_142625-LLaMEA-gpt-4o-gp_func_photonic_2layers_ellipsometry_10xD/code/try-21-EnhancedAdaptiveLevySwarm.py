import numpy as np

class EnhancedAdaptiveLevySwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(40, budget // 3)
        self.w = 0.5  # Initial inertia weight
        self.c1 = 1.5 + np.random.rand()  # Cognitive coefficient
        self.c2 = 1.5 + np.random.rand()  # Social coefficient
        self.func_evals = 0

    def levy_flight(self, size, scale):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, size) * sigma
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / beta)
        return step * scale

    def chaotic_init(self, lb, ub):
        x = np.random.rand(self.swarm_size, self.dim)
        r = 4.0 - 2.0 * (self.func_evals / self.budget)  # Dynamic chaos parameter
        for _ in range(100):
            x = r * x * (1 - x)
        return lb + (ub - lb) * x

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm = self.chaotic_init(lb, ub)
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best = swarm.copy()
        personal_best_scores = np.array([func(x) for x in personal_best])
        self.func_evals += self.swarm_size

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = personal_best_scores.min()

        while self.func_evals < self.budget:
            self.w = 0.5 + 0.5 * np.sin(np.pi * self.func_evals / self.budget)  # Sinusoidal inertia weight adjustment
            self.c1 = 1.5 + 1.5 * np.sin(np.pi * self.func_evals / self.budget)
            self.c2 = 1.5 + 1.5 * np.cos(np.pi * self.func_evals / self.budget)
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)

            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best - swarm) +
                          self.c2 * r2 * (global_best - swarm))

            swarm = swarm + velocities
            scale = np.sqrt(1.0 - self.func_evals / self.budget)
            swarm += self.levy_flight((self.swarm_size, self.dim), scale)  # Dynamic Levy flight scaling
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