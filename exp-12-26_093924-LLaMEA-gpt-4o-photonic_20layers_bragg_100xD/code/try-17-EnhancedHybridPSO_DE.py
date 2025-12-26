import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.base_w = 0.5  # base inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.base_F = 0.8  # base scaling factor
        self.base_CR = 0.9  # base crossover probability

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
            # Adapt inertia weight dynamically
            dynamic_w = self.base_w + 0.1 * np.random.rand()
            # Use stochastic c1 and c2 to adjust exploration/exploitation balance
            stochastic_c1 = self.c1 * (0.5 + np.random.rand())
            stochastic_c2 = self.c2 * (0.5 + np.random.rand())

            r1, r2 = np.random.rand(2)
            velocity = (dynamic_w * velocity +
                        stochastic_c1 * r1 * (p_best - population) +
                        stochastic_c2 * r2 * (g_best - population))
            population = np.clip(population + velocity, lb, ub)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                # Adaptive scaling factor for DE
                adaptive_F = self.base_F + 0.3 * np.random.rand()
                mutant = np.clip(a + adaptive_F * (b - c), lb, ub)
                # Adaptive crossover probability
                adaptive_CR = self.base_CR * (0.5 + np.random.rand())
                cross_points = np.random.rand(self.dim) < adaptive_CR
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

        return g_best