import numpy as np

class AdvancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.F = 0.8  # Mutation factor for DE
        self.CR = 0.9  # Crossover probability for DE
        self.learning_rate = 0.1  # Learning rate for adaptive changes
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        evaluations = self.population_size
        r_adaptive = np.random.rand(self.population_size, self.dim)

        while evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - 1.5) * evaluations / self.budget)
            c2 = self.c2_initial + ((2.0 - self.c2_initial) * evaluations / self.budget)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (w * velocity +
                        c1 * r1 * (personal_best - population) +
                        c2 * r2 * (global_best - population))
            population += velocity
            population = np.clip(population, lb, ub)
            
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                # Apply chaotic mutation
                r_adaptive = self.learning_rate * r_adaptive * (1 - r_adaptive)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i] + r_adaptive[i])
                trial_value = func(trial)
                evaluations += 1

                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if evaluations >= self.budget:
                    break

        return global_best