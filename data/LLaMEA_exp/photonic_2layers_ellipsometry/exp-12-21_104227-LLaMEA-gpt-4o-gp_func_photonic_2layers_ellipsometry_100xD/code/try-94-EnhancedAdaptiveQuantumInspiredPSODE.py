import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedAdaptiveQuantumInspiredPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.5   # inertia weight
        self.F = 0.8   # mutation factor for DE
        self.CR = 0.9  # crossover probability for DE
        self.alpha = 0.01  # scaling factor for Levy flight

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return self.alpha * step
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        q_population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-np.abs(ub-lb), np.abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        def evaluate(ind):
            return func(ind)

        while evaluations < self.budget:
            self.w = 0.9 - 0.5 * (evaluations / self.budget)
            self.F = 0.1 + 0.7 * (1 - evaluations / self.budget)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population += velocities * (0.9 + 0.1 * (evaluations / self.budget))
            population = np.clip(population, lb, ub)

            q_population = 0.5 * (population + global_best_position) + np.random.uniform(-1, 1, (self.population_size, self.dim)) * np.abs(population - global_best_position)
            q_population = np.clip(q_population, lb, ub)

            with ThreadPoolExecutor() as executor:
                scores = np.array(list(executor.map(evaluate, population)))
                q_scores = np.array(list(executor.map(evaluate, q_population)))
            evaluations += 2 * self.population_size

            improved_q = q_scores < scores
            population[improved_q] = q_population[improved_q]
            scores[improved_q] = q_scores[improved_q]

            for i in range(self.population_size):
                indices = np.arange(self.population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

            improved = scores < personal_best_scores
            personal_best_positions[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if np.min(scores) < global_best_score:
                global_best_position = population[np.argmin(scores)]
                global_best_score = np.min(scores)

            # Levy flights for exploration
            if evaluations < self.budget:
                levy_population = population + np.array([self.levy_flight() for _ in range(self.population_size)])
                levy_population = np.clip(levy_population, lb, ub)
                levy_scores = np.array([func(ind) for ind in levy_population])
                evaluations += self.population_size

                better_levy = levy_scores < scores
                population[better_levy] = levy_population[better_levy]
                scores[better_levy] = levy_scores[better_levy]

                # Update personal and global best again
                improved = scores < personal_best_scores
                personal_best_positions[improved] = population[improved]
                personal_best_scores[improved] = scores[improved]
                if np.min(scores) < global_best_score:
                    global_best_position = population[np.argmin(scores)]
                    global_best_score = np.min(scores]

        return global_best_position, global_best_score