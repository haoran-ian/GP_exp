import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w = 0.5  # inertia weight for PSO
        self.c1 = 1.5 # cognitive coefficient for PSO
        self.c2 = 1.5 # social coefficient for PSO
        self.F = 0.8  # scaling factor for DE
        self.CR = 0.9 # crossover probability for DE

    def __call__(self, func):
        # Initialize the population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.normal(0, 0.1, (self.population_size, self.dim))  # Stochastic initialization
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
            # Update velocity and position for PSO
            r1, r2 = np.random.rand(2)
            self.w = 0.4 + 0.1 * np.random.rand()  # Dynamic inertia weight update
            velocity = (self.w * velocity +
                        self.c1 * r1 * (p_best - population) +
                        self.c2 * r2 * (g_best - population))
            population = np.clip(population + velocity, lb, ub)

            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                # Adaptive scaling factor for DE
                self.F = 0.5 + 0.3 * np.random.rand()  
                mutant = np.clip(a + self.F * ((b - c) + (g_best - population[i])), lb, ub)  # Enhanced mutation strategy
                # Dynamic crossover probability
                self.CR = 0.5 + 0.4 * (g_best_score / (p_best_scores[i] + 1e-9)) 
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

            # Adaptive population size adjustment
            self.population_size = max(10, int(20 * (1 - (evaluations / self.budget))))

        return g_best