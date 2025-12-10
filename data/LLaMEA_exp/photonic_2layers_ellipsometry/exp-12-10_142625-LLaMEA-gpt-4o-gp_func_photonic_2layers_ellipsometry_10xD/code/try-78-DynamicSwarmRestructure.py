import numpy as np

class DynamicSwarmRestructure:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget // 3)
        self.w = 0.7  # Initial inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.memory = np.random.rand(self.swarm_size, self.dim)
        self.func_evals = 0

    def chaotic_perturbation(self, size, diversity_factor):
        """Enhanced chaotic perturbation based on current diversity."""
        beta = 1.5 + 0.5 * diversity_factor  # Adjusted for diversity
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

    def update_memory(self, swarm, scores, personal_best_scores):
        for i in range(self.swarm_size):
            if scores[i] < personal_best_scores[i]:
                self.memory[i] = swarm[i]

    def restructure_swarm(self, swarm, scores):
        """Dynamic restructuring of the swarm based on current performance."""
        top_performers = np.argsort(scores)[:self.swarm_size // 2]
        new_positions = np.random.rand(self.swarm_size // 2, self.dim)
        swarm[top_performers] = new_positions
        return swarm

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
            self.w = 0.7 - 0.5 * (self.func_evals / self.budget)
            self.c1 = 2.0 - (1.5 * np.cos(np.pi * self.func_evals / self.budget))
            self.c2 = 2.0 - (1.5 * np.sin(np.pi * self.func_evals / self.budget))
            entropy = self.calculate_entropy(swarm)
            mutation_rate = 0.1 + 0.4 * (1 - entropy / np.log(self.swarm_size))

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)

            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best - swarm) +
                          self.c2 * r2 * (global_best - swarm) + np.random.normal(0, 0.1, velocities.shape))

            memory_influence = np.random.rand(self.swarm_size, self.dim)
            swarm = swarm + velocities + memory_influence * (self.memory - swarm)
            perturbation = self.chaotic_perturbation((self.swarm_size, self.dim), entropy) * mutation_rate
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

            self.update_memory(swarm, scores, personal_best_scores)

            if np.random.rand() < mutation_rate:
                swarm = self.restructure_swarm(swarm, scores)

        return global_best