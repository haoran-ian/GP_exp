import numpy as np

class EnhancedAdaptiveCooperativeSwarmV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget // 3)
        self.w = 0.7  # Initial inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.func_evals = 0

    def chaotic_perturbation(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, 1, size) * sigma
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def cooperative_init(self, lb, ub):
        x = np.random.rand(self.swarm_size, self.dim)
        r = 4 - 3 * (self.func_evals / self.budget)
        for _ in range(100):
            x = r * x * (1 - x)
        return lb + 0.5 * (ub - lb) * x

    def calculate_entropy(self, swarm):
        prob = np.mean(swarm, axis=0)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        return entropy

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
            self.w = 0.7 - 0.3 * (self.func_evals / self.budget)
            self.c1 = 2.5 - (1.5 * np.cos(np.pi * self.func_evals / self.budget))
            self.c2 = 2.5 - (1.5 * np.sin(np.pi * self.func_evals / self.budget))

            entropy = self.calculate_entropy(swarm)
            mutation_rate = 0.1 + 0.4 * (1 - entropy / np.log(self.swarm_size))

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)

            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best - swarm) +
                          self.c2 * r2 * (global_best - swarm))

            perturbation = self.chaotic_perturbation((self.swarm_size, self.dim)) * mutation_rate
            swarm += velocities + perturbation
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

            # Differential evolution-inspired crossover
            if np.random.rand() < mutation_rate:
                random_indexes = np.random.choice(self.swarm_size, size=self.swarm_size // 4, replace=False)
                for idx in random_indexes:
                    ind1, ind2, ind3 = np.random.choice(self.swarm_size, size=3, replace=False)
                    mutant = swarm[ind1] + 0.8 * (swarm[ind2] - swarm[ind3])
                    trial = np.where(np.random.rand(self.dim) < 0.5, mutant, swarm[idx])
                    trial = np.clip(trial, lb, ub)
                    trial_score = func(trial)
                    self.func_evals += 1
                    if trial_score < scores[idx]:
                        swarm[idx] = trial
                        personal_best[idx] = trial
                        personal_best_scores[idx] = trial_score

        return global_best