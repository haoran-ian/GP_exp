import numpy as np
import skfuzzy as fuzz

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_population_size = self.population_size
        self.w = 0.5  # inertia weight for PSO
        self.c1 = 1.5 # cognitive coefficient for PSO
        self.c2 = 1.5 # social coefficient for PSO
        self.F = 0.8  # scaling factor for DE
        self.CR = 0.9 # crossover probability for DE

    def fuzzy_logic(self, g_best_score, evaluations_ratio):
        # Fuzzy logic system to adjust parameters
        x_range = np.arange(0, 1.1, 0.1)
        g_score = fuzz.trapmf(x_range, [0, 0, 0.5, 0.7])
        eval_ratio = fuzz.trapmf(x_range, [0.3, 0.5, 0.7, 1])
        
        rule1 = np.fmin(g_score, eval_ratio)
        rule2 = np.fmin(g_score, np.fmax(np.ones_like(eval_ratio) - eval_ratio, 0))
        
        # Aggregate the two rules
        aggregated = np.fmax(rule1, rule2)

        # Defuzzify the aggregated output
        parameter_adjustment = fuzz.defuzz(x_range, aggregated, 'centroid')
        
        return parameter_adjustment

    def __call__(self, func):
        # Initialize the population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = population.copy()
        p_best_scores = np.array([float('inf')]*self.population_size)
        
        # Evaluate initial population
        for i in range(self.population_size):
            score = func(population[i])
            p_best_scores[i] = score

        g_best = population[np.argmin(p_best_scores)]
        g_best_score = min(p_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            evaluations_ratio = evaluations / self.budget
            adjustment = self.fuzzy_logic(g_best_score / (np.min(p_best_scores) + 1e-9), evaluations_ratio)

            # Update velocity and position for PSO
            r1, r2 = np.random.rand(2)
            self.w = 0.4 + 0.3 * adjustment
            velocity = (self.w * velocity +
                        self.c1 * r1 * (p_best - population) +
                        self.c2 * r2 * (g_best - population))
            population = np.clip(population + velocity, lb, ub)

            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.F = 0.5 + 0.3 * adjustment
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                self.CR = 0.5 + 0.4 * adjustment * (g_best_score / (p_best_scores[i] + 1e-9))
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_score = func(trial)
                evaluations += 1
                # Selection
                if trial_score < p_best_scores[i]:
                    p_best[i] = trial
                    p_best_scores[i] = trial_score
                    if trial_score < g_best_score:
                        g_best = trial
                        g_best_score = trial_score
                if evaluations >= self.budget:
                    break

            # Adjust population size using fuzzy logic
            self.population_size = max(10, int(self.initial_population_size * (1 - evaluations_ratio)))
        
        return g_best