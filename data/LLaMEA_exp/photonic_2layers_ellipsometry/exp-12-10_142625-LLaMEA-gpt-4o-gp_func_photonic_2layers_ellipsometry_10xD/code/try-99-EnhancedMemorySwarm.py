import numpy as np

class EnhancedMemorySwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget // 3)
        self.w = 0.7  # Initial inertia weight
        self.c1 = 2.0  # Initial cognitive coefficient
        self.c2 = 2.0  # Initial social coefficient
        self.memory = np.random.rand(self.swarm_size, self.dim)  # Memory for adaptive learning
        self.func_evals = 0

    def chaotic_perturbation(self, size):
        beta = 1.8 + 0.2 * np.sin(self.func_evals / self.budget * np.pi)  # Adaptive beta for dynamic intensity
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

    def adaptive_learning_rates(self):
        self.c1 = 2.0 * (1 - self.func_evals / self.budget)  # Decreasing cognitive coefficient
        self.c2 = 2.0 * (self.func_evals / self.budget)  # Increasing social coefficient

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
            self.w = 0.9 - (0.9 - 0.4) / (1 + np.exp(-10 * (self.func_evals / self.budget - 0.5)))
            self.adaptive_learning_rates()
            entropy = self.calculate_entropy(swarm)
            mutation_rate = 0.1 + 0.4 * (1 - entropy / np.log(self.swarm_size)) * np.sin((self.func_evals / self.budget) * np.pi)

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)

            local_exploitation = r1 * (personal_best - swarm)
            global_exploration = r2 * (global_best - swarm) + self.chaotic_perturbation((self.swarm_size, self.dim)) * mutation_rate
            
            velocities = (self.w * velocities + self.c1 * local_exploitation + self.c2 * global_exploration) * 0.99 + np.random.normal(0, 0.01, velocities.shape)
            
            memory_influence = np.random.rand(self.swarm_size, self.dim)
            swarm = swarm + velocities + memory_influence * (self.memory - swarm)
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
                random_indexes = np.random.choice(self.swarm_size, size=self.swarm_size // 4, replace=False)
                new_positions = self.cooperative_init(lb, ub)
                swarm[random_indexes] = new_positions[random_indexes]

        return global_best