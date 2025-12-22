import numpy as np

class EnhancedQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.initial_c1 = 2.0
        self.initial_c2 = 2.0
        self.initial_w = 0.9
        self.quantum_factor = 0.05
        self.cooling_rate = 0.98
        self.swarm_count = 3  # Number of swarms for multi-swarm strategy
        self.swarm_sizes = [self.initial_population_size // self.swarm_count] * self.swarm_count

    def adaptive_inertia_weight(self, evals):
        return 0.4 + 0.5 * (1 - evals / self.budget)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evals = 0
        swarms = [np.random.uniform(low=lb, high=ub, size=(size, self.dim))
                  for size in self.swarm_sizes]
        velocities = [np.random.uniform(size=(size, self.dim)) * (ub - lb) / 20.0
                      for size in self.swarm_sizes]
        personal_best = [swarm.copy() for swarm in swarms]
        personal_best_scores = [np.array([func(p) for p in swarm]) for swarm in personal_best]
        global_best_index = [np.argmin(scores) for scores in personal_best_scores]
        global_best = [personal_best[i][global_best_index[i]] for i in range(self.swarm_count)]
        global_best_score = [personal_best_scores[i][global_best_index[i]] for i in range(self.swarm_count)]
        evals += sum(self.swarm_sizes)

        while evals < self.budget:
            for swarm_id in range(self.swarm_count):
                for i in range(self.swarm_sizes[swarm_id]):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    w = self.adaptive_inertia_weight(evals)
                    c1 = self.initial_c1 * (1 - evals / self.budget)
                    c2 = self.initial_c2 * (evals / self.budget)
                    velocities[swarm_id][i] = (w * velocities[swarm_id][i] +
                                               c1 * r1 * (personal_best[swarm_id][i] - swarms[swarm_id][i]) +
                                               c2 * r2 * (global_best[swarm_id] - swarms[swarm_id][i]))
                    swarms[swarm_id][i] = np.clip(swarms[swarm_id][i] + velocities[swarm_id][i], lb, ub)

                    if np.random.rand() < self.quantum_factor:
                        swarms[swarm_id][i] = np.clip(global_best[swarm_id] + np.random.normal(size=self.dim) * (ub - lb) / 10.0, lb, ub)

                    score = func(swarms[swarm_id][i])
                    evals += 1
                    if evals >= self.budget:
                        break

                    if score < personal_best_scores[swarm_id][i]:
                        personal_best[swarm_id][i] = swarms[swarm_id][i]
                        personal_best_scores[swarm_id][i] = score

                    if score < global_best_score[swarm_id]:
                        global_best[swarm_id] = swarms[swarm_id][i]
                        global_best_score[swarm_id] = score

            if evals % 100 == 0:  # Swap global best across swarms periodically
                for i in range(self.swarm_count):
                    for j in range(i + 1, self.swarm_count):
                        if global_best_score[j] < global_best_score[i]:
                            global_best[i], global_best[j] = global_best[j], global_best[i]
                            global_best_score[i], global_best_score[j] = global_best_score[j], global_best_score[i]

        return min(global_best, key=lambda b: func(b))