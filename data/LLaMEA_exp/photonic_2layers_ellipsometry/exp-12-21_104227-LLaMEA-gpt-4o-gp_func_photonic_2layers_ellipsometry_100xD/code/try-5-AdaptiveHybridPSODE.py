import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1_init = 2.0  # initial cognitive component
        self.c2_init = 2.0  # initial social component
        self.w_init = 0.9   # initial inertia weight
        self.w_min = 0.4    # minimum inertia weight
        self.F_init = 0.8   # initial mutation factor for DE
        self.CR_init = 0.9  # initial crossover probability for DE
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-np.abs(ub-lb), np.abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        c1, c2 = self.c1_init, self.c2_init
        w = self.w_init
        F = self.F_init
        CR = self.CR_init

        while evaluations < self.budget:
            # Dynamic adjustments
            if evaluations > self.budget / 2:
                c1 = max(1.0, c1 - 0.001)
                c2 = min(3.0, c2 + 0.001)
                w = max(self.w_min, w - 0.001)
                F = max(0.1, F - 0.001)
                CR = max(0.7, CR - 0.001)

            # PSO update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - population) +
                          c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, lb, ub)
            
            # Evaluate new positions
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size
            
            # DE mutation and crossover
            for i in range(self.population_size):
                indices = np.arange(self.population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, lb, ub)
                
                cross_points = np.random.rand(self.dim) < CR
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