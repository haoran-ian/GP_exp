import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.5   # inertia weight
        self.F = 0.8   # mutation factor for DE
        self.CR = 0.9  # crossover probability for DE
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-np.abs(ub-lb), np.abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            self.w = 0.9 - 0.5 * (evaluations / self.budget)  # Adjust inertia weight over time
            # PSO update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, lb, ub)
            
            # Evaluate new positions
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size
            
            # DE mutation
            for i in range(self.population_size):
                indices = np.arange(self.population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.F = 0.5 + 0.5 * np.random.rand()  # Dynamic adaptation of F
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, lb, ub)
                
                # DE crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score
            
            # Update personal and global best
            improved = scores < personal_best_scores
            personal_best_positions[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if np.min(scores) < global_best_score:
                global_best_position = population[np.argmin(scores)]
                global_best_score = np.min(scores)
            
        return global_best_position, global_best_score