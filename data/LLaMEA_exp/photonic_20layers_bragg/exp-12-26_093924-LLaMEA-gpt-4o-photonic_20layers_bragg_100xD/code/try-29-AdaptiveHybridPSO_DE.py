import numpy as np

class AdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_min = 0.3
        self.w_max = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.F_base = 0.5
        self.CR_base = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = population.copy()
        p_best_scores = np.array([func(ind) for ind in population])
        g_best = population[np.argmin(p_best_scores)]
        g_best_score = min(p_best_scores)
        evaluations = self.population_size

        while evaluations < self.budget:
            phase_ratio = evaluations / self.budget
            self.w = self.w_max - (self.w_max - self.w_min) * phase_ratio
            r1, r2 = np.random.rand(2)
            velocity = (self.w * velocity +
                        self.c1 * r1 * (p_best - population) +
                        self.c2 * r2 * (g_best - population))
            population = np.clip(population + velocity, lb, ub)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.F = self.F_base + 0.5 * np.random.rand() * phase_ratio
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                self.CR = self.CR_base + 0.4 * np.random.rand() * (1 - phase_ratio)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
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

            self.population_size = max(10, int(20 * (1 - phase_ratio)))

        return g_best