import numpy as np

class AdaptiveMemoryPSOGWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = 30
        self.w_init = 0.9
        self.w_min = 0.4
        self.c1_init = 2.0
        self.c2_init = 2.0
        self.alpha, self.beta, self.delta = None, None, None
        self.best_cost = float('inf')
        self.best_solution = None
        self.memory_size = 5
        self.memory = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x = np.random.uniform(lb, ub, (self.particles, self.dim))
        v = np.random.uniform(-1, 1, (self.particles, self.dim))
        personal_best = x.copy()
        personal_best_cost = np.array([float('inf')] * self.particles)

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.particles):
                if evaluations >= self.budget:
                    break

                cost = func(x[i])
                evaluations += 1

                if cost < personal_best_cost[i]:
                    personal_best_cost[i] = cost
                    personal_best[i] = (personal_best[i] + x[i]) / 2

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = x[i].copy()

            sorted_indices = np.argsort(personal_best_cost)
            self.alpha = personal_best[sorted_indices[0]]
            self.beta = personal_best[sorted_indices[1]]
            self.delta = personal_best[sorted_indices[2]]

            self.memory.append(self.alpha)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

            for i in range(self.particles):
                w = self.w_min + (self.w_init - self.w_min) * (1 - evaluations / self.budget)
                c1 = self.c1_init * (1 - evaluations / self.budget)
                c2 = self.c2_init * (evaluations / self.budget)

                r1, r2 = np.random.rand(), np.random.rand()
                v[i] = (w * v[i] 
                        + c1 * r1 * (personal_best[i] - x[i]) 
                        + c2 * r2 * (self.best_solution - x[i]))
                x[i] = np.clip(x[i] + v[i], lb, ub)

                leader = self.best_solution if evaluations >= self.budget / 2 else self.alpha

                A1, A2, A3 = 2 * np.random.rand(3, self.dim) - 1
                C1, C2, C3 = 2 * np.random.rand(3, self.dim)

                D_leader = np.abs(C1 * leader - x[i])
                D_beta = np.abs(C2 * self.beta - x[i])
                D_delta = np.abs(C3 * self.delta - x[i])

                X1 = leader - A1 * D_leader
                X2 = self.beta - A2 * D_beta
                X3 = self.delta - A3 * D_delta

                candidate_solution = (X1 + X2 + X3 + x[i]) / 4
                candidate_solution = np.clip(candidate_solution, lb, ub)
                candidate_cost = func(candidate_solution)
                evaluations += 1

                if candidate_cost < personal_best_cost[i]:
                    personal_best_cost[i] = candidate_cost
                    personal_best[i] = candidate_solution.copy()
                if candidate_cost < self.best_cost:
                    self.best_cost = candidate_cost
                    self.best_solution = candidate_solution.copy()

                if np.random.rand() < 0.1:
                    memory_leader = self.memory[np.random.randint(len(self.memory))]
                    random_walk = np.random.uniform(-1, 1, self.dim)
                    walk_solution = np.clip(memory_leader + random_walk, lb, ub)
                    walk_cost = func(walk_solution)
                    evaluations += 1

                    if walk_cost < personal_best_cost[i]:
                        personal_best_cost[i] = walk_cost
                        personal_best[i] = walk_solution.copy()
                    if walk_cost < self.best_cost:
                        self.best_cost = walk_cost
                        self.best_solution = walk_solution.copy()