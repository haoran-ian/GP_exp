import numpy as np

class QuantumAdaptiveSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget // 3)
        self.w = 0.7
        self.c1 = 2.0
        self.c2 = 2.0
        self.func_evals = 0

    def quantum_perturbation(self, size):
        delta = 1e-4
        alpha = np.random.uniform(0, 2 * np.pi, size)
        return delta * np.tan(alpha)

    def adaptive_neighborhood(self, swarm, scores):
        adj_matrix = np.zeros((self.swarm_size, self.swarm_size))
        for i in range(self.swarm_size):
            distances = np.linalg.norm(swarm - swarm[i], axis=1)
            nearest_idx = np.argsort(distances)[1:4]
            adj_matrix[i, nearest_idx] = 1
        return adj_matrix

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        personal_best = swarm.copy()
        personal_best_scores = np.array([func(x) for x in personal_best])
        self.func_evals += self.swarm_size

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = personal_best_scores.min()

        while self.func_evals < self.budget:
            self.w = 0.5 + 0.4 * (np.sin(np.pi * self.func_evals / self.budget))  # Dynamic inertia weight
            adj_matrix = self.adaptive_neighborhood(swarm, personal_best_scores)

            for i in range(self.swarm_size):
                if adj_matrix[i].sum() > 0:
                    local_best_idx = np.argmin(personal_best_scores[adj_matrix[i] == 1])
                    local_best = personal_best[local_best_idx]
                else:
                    local_best = global_best

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)

                velocities[i] = (self.w * velocities[i] +
                                self.c1 * r1 * (personal_best[i] - swarm[i]) +
                                self.c2 * r2 * (local_best - swarm[i]))

            swarm += velocities + self.quantum_perturbation((self.swarm_size, self.dim))
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