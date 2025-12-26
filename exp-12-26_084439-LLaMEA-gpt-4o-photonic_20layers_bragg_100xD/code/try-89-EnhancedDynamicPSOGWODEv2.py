import numpy as np

class EnhancedDynamicPSOGWODEv2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = 30
        self.w = 0.9  # adaptive inertia
        self.c1_init = 2.0  # initial cognitive coefficient
        self.c2_init = 2.0  # initial social coefficient
        self.best_cost = float('inf')
        self.best_solution = None

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
                    personal_best[i] = x[i].copy()

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = x[i].copy()

            sorted_indices = np.argsort(personal_best_cost)
            alpha = personal_best[sorted_indices[0]]
            beta = personal_best[sorted_indices[1]]
            delta = personal_best[sorted_indices[2]]

            for i in range(self.particles):
                self.w = 0.9 - 0.7 * (evaluations / self.budget)
                c1 = self.c1_init * (1 - evaluations / self.budget) * 0.5
                c2 = self.c2_init * (evaluations / self.budget)

                r1, r2 = np.random.rand(), np.random.rand()
                v[i] = (self.w * v[i]
                        + c1 * r1 * (personal_best[i] - x[i])
                        + c2 * r2 * (self.best_solution - x[i]))
                x[i] = np.clip(x[i] + v[i], lb, ub)

                if np.random.rand() < 0.5:
                    leader = alpha
                else:
                    # Introduce stochastic exploration around beta and delta
                    if np.random.rand() < 0.5:
                        leader = beta
                    else:
                        leader = delta

                A1, A2, A3 = 2 * np.random.rand(3, self.dim) - 1
                C1, C2, C3 = 2 * np.random.rand(3, self.dim)

                D_leader = np.abs(C1 * leader - x[i])
                D_beta = np.abs(C2 * beta - x[i])
                D_delta = np.abs(C3 * delta - x[i])

                X1 = leader - A1 * D_leader
                X2 = beta - A2 * D_beta
                X3 = delta - A3 * D_delta

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

                # Introduce vortex strategy with randomness
                vortex_vector = np.random.normal(0, 1, self.dim)
                intensity = np.exp(-2 * (candidate_cost - self.best_cost))
                vortex_solution = x[i] + vortex_vector * intensity * np.random.rand()
                vortex_solution = np.clip(vortex_solution, lb, ub)
                vortex_cost = func(vortex_solution)
                evaluations += 1

                if vortex_cost < personal_best_cost[i]:
                    personal_best_cost[i] = vortex_cost
                    personal_best[i] = vortex_solution.copy()
                if vortex_cost < self.best_cost:
                    self.best_cost = vortex_cost
                    self.best_solution = vortex_solution.copy()

                # Refined differential evolution-inspired mutation
                if np.random.rand() < 0.5:
                    indices = np.arange(self.particles)
                    np.random.shuffle(indices)
                    idx1, idx2, idx3 = indices[:3]
                    F = np.random.rand() * 0.5 + 0.5
                    mutant = personal_best[idx1] + F * (personal_best[idx2] - personal_best[idx3])
                    mutant = np.clip(mutant, lb, ub)
                    mutant_cost = func(mutant)
                    evaluations += 1

                    if mutant_cost < personal_best_cost[i]:
                        personal_best_cost[i] = mutant_cost
                        personal_best[i] = mutant.copy()
                    if mutant_cost < self.best_cost:
                        self.best_cost = mutant_cost
                        self.best_solution = mutant.copy()