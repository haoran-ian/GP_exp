import numpy as np

class AdaptiveMultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.swarm_size = min(30, budget // (2 * self.num_swarms))
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
        multi_global_best_score = float('inf')
        multi_global_best = None

        swarms = [self.cooperative_init(lb, ub) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best = [swarm.copy() for swarm in swarms]
        personal_best_scores = [np.array([func(x) for x in pb]) for pb in personal_best]
        self.func_evals += self.num_swarms * self.swarm_size

        global_bests = [pb[np.argmin(pbs)] for pb, pbs in zip(personal_best, personal_best_scores)]
        global_best_scores = [pbs.min() for pbs in personal_best_scores]

        while self.func_evals < self.budget:
            updates = []
            for i in range(self.num_swarms):
                self.w = 0.7 - 0.5 * (self.func_evals / self.budget)
                self.c1 = 2.0 - (1.5 * np.cos(np.pi * self.func_evals / self.budget))
                self.c2 = 2.0 - (1.5 * np.sin(np.pi * self.func_evals / self.budget))
                
                entropy = self.calculate_entropy(swarms[i])
                mutation_rate = 0.1 + 0.4 * (1 - entropy / np.log(self.swarm_size))

                r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)

                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - swarms[i]) +
                                 self.c2 * r2 * (global_bests[i] - swarms[i]) + 
                                 np.random.normal(0, 0.1, velocities[i].shape))

                swarms[i] = swarms[i] + velocities[i]
                perturbation = self.chaotic_perturbation((self.swarm_size, self.dim)) * mutation_rate
                swarms[i] += perturbation
                swarms[i] = np.clip(swarms[i], lb, ub)

                scores = np.array([func(x) for x in swarms[i]])
                self.func_evals += self.swarm_size

                for j in range(self.swarm_size):
                    if scores[j] < personal_best_scores[i][j]:
                        personal_best[i][j] = swarms[i][j]
                        personal_best_scores[i][j] = scores[j]

                min_idx = personal_best_scores[i].argmin()
                if personal_best_scores[i][min_idx] < global_best_scores[i]:
                    global_bests[i] = personal_best[i][min_idx]
                    global_best_scores[i] = personal_best_scores[i][min_idx]

                updates.append((global_best_scores[i], global_bests[i]))

            for score, best in updates:
                if score < multi_global_best_score:
                    multi_global_best_score = score
                    multi_global_best = best

            if np.random.rand() < mutation_rate:
                for i in range(self.num_swarms):
                    random_indexes = np.random.choice(self.swarm_size, size=self.swarm_size // 4, replace=False)
                    new_positions = self.cooperative_init(lb, ub)
                    swarms[i][random_indexes] = new_positions[random_indexes]

        return multi_global_best