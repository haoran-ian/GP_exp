import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.min_population_size = 10
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.5  # cognitive coefficient for PSO
        self.c2 = 1.5  # social coefficient for PSO
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.5
        self.CR_max = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = population.copy()
        p_best_scores = np.full(self.population_size, float('inf'))
        
        evaluations = 0
        for i in range(self.population_size):
            score = func(population[i])
            p_best_scores[i] = score
            evaluations += 1
            
        g_best = population[np.argmin(p_best_scores)]
        g_best_score = min(p_best_scores)

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            F = self.F_min + (self.F_max - self.F_min) * (1 - evaluations / self.budget)
            CR = self.CR_min + (self.CR_max - self.CR_min) * (1 - g_best_score / (min(p_best_scores) + 1e-9))
            
            r1, r2 = np.random.rand(2)
            velocity = (w * velocity +
                        self.c1 * r1 * (p_best - population) +
                        self.c2 * r2 * (g_best - population))
            population = np.clip(population + velocity, lb, ub)
            
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
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

            self.population_size = max(self.min_population_size, int(self.initial_population_size * (1 - (evaluations / self.budget))))

        return g_best