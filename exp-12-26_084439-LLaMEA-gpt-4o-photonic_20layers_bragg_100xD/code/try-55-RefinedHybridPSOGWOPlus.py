import numpy as np

class RefinedHybridPSOGWOPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = 30
        self.w = 0.9  # adaptive inertia
        self.c1_init = 2.0  # initial cognitive coefficient
        self.c2_init = 2.0  # initial social coefficient
        self.alpha, self.beta, self.delta = None, None, None
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
                    personal_best[i] = (personal_best[i] + x[i]) / 2

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = x[i].copy()

            sorted_indices = np.argsort(personal_best_cost)
            self.alpha = personal_best[sorted_indices[0]]
            self.beta = personal_best[sorted_indices[1]]
            self.delta = personal_best[sorted_indices[2]]

            for i in range(self.particles):
                self.w = 0.9 - 0.65 * (evaluations / self.budget)
                c1 = self.c1_init * (1 - evaluations / self.budget)
                c2 = self.c2_init * (evaluations / self.budget)
                
                r1, r2 = np.random.rand(), np.random.rand()
                v[i] = (self.w * v[i] 
                        + c1 * r1 * (personal_best[i] - x[i]) 
                        + c2 * r2 * (self.best_solution - x[i]))
                x[i] = np.clip(x[i] + v[i], lb, ub)

                # Grey Wolf Optimization component
                A1, A2, A3 = 2 * np.random.rand(3, self.dim) - 1
                C1, C2, C3 = 2 * np.random.rand(3, self.dim)
                
                D_alpha = np.abs(C1 * self.alpha - x[i])
                D_beta = np.abs(C2 * self.beta - x[i])
                D_delta = np.abs(C3 * self.delta - x[i])
                
                X1 = self.alpha - A1 * D_alpha
                X2 = self.beta - A2 * D_beta
                X3 = self.delta - A3 * D_delta
                
                candidate_solution = (X1 + X2 + X3) / 3
                candidate_solution = np.clip(candidate_solution, lb, ub)
                candidate_cost = func(candidate_solution)
                evaluations += 1

                if candidate_cost < personal_best_cost[i]:
                    personal_best_cost[i] = candidate_cost
                    personal_best[i] = candidate_solution.copy()
                if candidate_cost < self.best_cost:
                    self.best_cost = candidate_cost
                    self.best_solution = candidate_solution.copy()

                # Adaptive particle clustering and phase-shifted vortex intensification
                cluster_intensity = np.exp(-2 * np.abs(candidate_cost - np.median(personal_best_cost)))
                vortex_vector = np.random.normal(0, 1, self.dim) * cluster_intensity
                vortex_phase_shift = np.sin(2 * np.pi * evaluations / self.budget)
                vortex_solution = x[i] + vortex_vector * vortex_phase_shift
                vortex_solution = np.clip(vortex_solution, lb, ub)
                vortex_cost = func(vortex_solution)
                evaluations += 1

                if vortex_cost < personal_best_cost[i]:
                    personal_best_cost[i] = vortex_cost
                    personal_best[i] = vortex_solution.copy()
                if vortex_cost < self.best_cost:
                    self.best_cost = vortex_cost
                    self.best_solution = vortex_solution.copy()