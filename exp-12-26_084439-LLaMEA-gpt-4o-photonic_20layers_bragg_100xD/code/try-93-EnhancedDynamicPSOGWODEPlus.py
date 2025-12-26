import numpy as np

class EnhancedDynamicPSOGWODEPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = max(10, dim * 2)  # adaptive swarm size
        self.w = 0.9
        self.c1_init = 2.0
        self.c2_init = 2.0
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
                    personal_best[i] = x[i].copy()

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = x[i].copy()

            sorted_indices = np.argsort(personal_best_cost)
            self.alpha = personal_best[sorted_indices[0]]
            self.beta = personal_best[sorted_indices[1]]
            self.delta = personal_best[sorted_indices[2]]

            for i in range(self.particles):
                self.w = 0.9 - 0.7 * (evaluations / self.budget)
                c1 = self.c1_init * (1 - evaluations / self.budget) * 0.5
                c2 = self.c2_init * (evaluations / self.budget)

                r1, r2 = np.random.rand(), np.random.rand()
                v[i] = (self.w * v[i]
                        + c1 * r1 * (personal_best[i] - x[i])
                        + c2 * r2 * (self.best_solution - x[i]))
                x[i] = np.clip(x[i] + v[i], lb, ub)

                if evaluations < self.budget / 2:
                    leader = self.alpha
                else:
                    leader = self.best_solution

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

                vortex_vector = np.random.normal(0, 1, self.dim)
                intensity = np.exp(-2 * (candidate_cost - self.best_cost))
                vortex_solution = x[i] + vortex_vector * intensity * (evaluations / self.budget)
                vortex_solution = np.clip(vortex_solution, lb, ub)
                vortex_cost = func(vortex_solution)
                evaluations += 1

                if vortex_cost < personal_best_cost[i]:
                    personal_best_cost[i] = vortex_cost
                    personal_best[i] = vortex_solution.copy()
                if vortex_cost < self.best_cost:
                    self.best_cost = vortex_cost
                    self.best_solution = vortex_solution.copy()

                # Differential Evolution-inspired mutation
                if np.random.rand() < 0.5:
                    indices = np.arange(self.particles)
                    np.random.shuffle(indices)
                    idx1, idx2, idx3 = indices[:3]
                    F = 0.5 + 0.5 * (evaluations / self.budget)
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

                # Genetic Algorithm-inspired crossover
                if np.random.rand() < 0.3:
                    idx1, idx2 = np.random.choice(self.particles, 2, replace=False)
                    crossover_point = np.random.randint(1, self.dim - 1)
                    offspring1 = np.concatenate((personal_best[idx1][:crossover_point], personal_best[idx2][crossover_point:]))
                    offspring2 = np.concatenate((personal_best[idx2][:crossover_point], personal_best[idx1][crossover_point:]))
                    
                    for offspring in [offspring1, offspring2]:
                        offspring = np.clip(offspring, lb, ub)
                        offspring_cost = func(offspring)
                        evaluations += 1

                        if offspring_cost < self.best_cost:
                            self.best_cost = offspring_cost
                            self.best_solution = offspring.copy()