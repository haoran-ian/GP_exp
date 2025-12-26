import numpy as np

class LevyDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = 30
        self.w = 0.9  # adaptive inertia weight
        self.c1_init = 2.0  # initial cognitive coefficient
        self.c2_init = 2.0  # initial social coefficient
        self.best_cost = float('inf')
        self.best_solution = None

    def levy_flight(self, size):
        # Implementing a simple Levy flight with beta = 1.5
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / beta))
        return step

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

            # Update inertia weight adaptively
            self.w = 0.5 + 0.4 * np.cos(np.pi * evaluations / self.budget)

            # Determine neighborhood dynamically
            for i in range(self.particles):
                # Update coefficients
                c1 = self.c1_init * (1 - evaluations / self.budget)
                c2 = self.c2_init * (evaluations / self.budget)

                r1, r2 = np.random.rand(), np.random.rand()
                v[i] = (self.w * v[i] 
                        + c1 * r1 * (personal_best[i] - x[i]) 
                        + c2 * r2 * (self.best_solution - x[i]))
                
                # Levy flight step
                if np.random.rand() < 0.3:
                    levy_step = self.levy_flight(self.dim)
                    x[i] = x[i] + levy_step * (x[i] - self.best_solution)
                else:
                    x[i] = np.clip(x[i] + v[i], lb, ub)

                # Evaluate new position
                cost = func(x[i])
                evaluations += 1

                if cost < personal_best_cost[i]:
                    personal_best_cost[i] = cost
                    personal_best[i] = x[i].copy()

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = x[i].copy()