import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = 30
        self.w_max = 0.9  # increased inertia for exploration
        self.w_min = 0.4  # decreased inertia for exploitation
        self.c1 = 2.0  # cognitive (particle)
        self.c2 = 2.0  # social (swarm)
        self.temperature = 1000
        self.cooling_rate = 0.995
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
            
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)  # adaptive inertia weight

            for i in range(self.particles):
                r1, r2 = np.random.rand(), np.random.rand()
                v[i] = (w * v[i] 
                        + self.c1 * r1 * (personal_best[i] - x[i]) 
                        + self.c2 * r2 * (self.best_solution - x[i]))
                x[i] = np.clip(x[i] + v[i], lb, ub)

                # Simulated Annealing component
                candidate_solution = x[i] + np.random.normal(0, 1, self.dim) * (self.temperature / 1000) # mutation scale
                candidate_solution = np.clip(candidate_solution, lb, ub)
                candidate_cost = func(candidate_solution)
                evaluations += 1

                if candidate_cost < cost or np.random.rand() < np.exp((cost - candidate_cost) / self.temperature):
                    x[i] = candidate_solution
                    if candidate_cost < personal_best_cost[i]:
                        personal_best_cost[i] = candidate_cost
                        personal_best[i] = candidate_solution.copy()
                    if candidate_cost < self.best_cost:
                        self.best_cost = candidate_cost
                        self.best_solution = candidate_solution.copy()

            self.temperature *= self.cooling_rate