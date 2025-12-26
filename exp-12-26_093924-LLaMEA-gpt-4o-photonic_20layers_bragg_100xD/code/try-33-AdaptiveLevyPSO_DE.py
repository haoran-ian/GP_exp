import numpy as np

class AdaptiveLevyPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_min = 0.3
        self.w_max = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F_base = 0.5
        self.CR_base = 0.5

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = population.copy()
        p_best_scores = np.array([float('inf')] * self.population_size)

        for i in range(self.population_size):
            score = func(population[i])
            p_best_scores[i] = score

        g_best = population[np.argmin(p_best_scores)]
        g_best_score = min(p_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            dynamic_pressure = (self.budget - evaluations) / self.budget
            self.w = self.w_min + (self.w_max - self.w_min) * np.random.rand() * dynamic_pressure
            r1, r2 = np.random.rand(2)
            velocity = (self.w * velocity +
                        self.c1 * r1 * (p_best - population) +
                        self.c2 * r2 * (g_best - population))
            population = np.clip(population + velocity, lb, ub)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.F = self.F_base + 0.2 * np.random.rand() * dynamic_pressure
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                self.CR = self.CR_base + 0.3 * np.random.rand() * (g_best_score / (p_best_scores[i] + 1e-9))
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Levy flight perturbation
                if np.random.rand() < 0.1:
                    trial += self.levy_flight(trial.shape)
                    trial = np.clip(trial, lb, ub)
                
                trial_score = func(trial)
                evaluations += 1
                if trial_score < p_best_scores[i]:
                    p_best[i] = trial
                    p_best_scores[i] = trial_score
                    if trial_score < g_best_score:
                        g_best = trial
                        g_best_score = trial_score
                if evaluations >= self.budget:
                    break

            self.population_size = max(10, int(20 * dynamic_pressure))

        return g_best