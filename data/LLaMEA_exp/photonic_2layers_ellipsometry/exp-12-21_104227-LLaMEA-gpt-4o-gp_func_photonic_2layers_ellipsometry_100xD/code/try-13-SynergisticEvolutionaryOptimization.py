import numpy as np
from concurrent.futures import ThreadPoolExecutor

class SynergisticEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased population size for diversity
        self.c1_initial, self.c1_final = 2.5, 0.5  # Adaptive cognitive component
        self.c2_initial, self.c2_final = 0.5, 2.5  # Adaptive social component
        self.w_initial, self.w_final = 0.9, 0.4    # Linearly decreasing inertia weight
        self.F_initial, self.F_final = 0.2, 0.9    # Adaptive mutation factor for DE
        self.CR = 0.9                              # Crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-np.abs(ub-lb), np.abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        def evaluate(ind):
            return func(ind)

        while evaluations < self.budget:
            # Adaptive parameters
            t = evaluations / self.budget
            self.w = self.w_final + (self.w_initial - self.w_final) * (1 - t)**2
            self.c1 = self.c1_final + (self.c1_initial - self.c1_final) * (1 - t)
            self.c2 = self.c2_final + (self.c2_initial - self.c2_final) * t
            self.F = self.F_final * t + self.F_initial * (1 - t)

            # PSO update with adaptive parameters
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, lb, ub)
            
            # Parallel evaluation of new positions
            with ThreadPoolExecutor() as executor:
                scores = list(executor.map(evaluate, population))
            scores = np.array(scores)
            evaluations += self.population_size
            
            # DE mutation with dynamic selection pressure
            for i in range(self.population_size):
                indices = np.arange(self.population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
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