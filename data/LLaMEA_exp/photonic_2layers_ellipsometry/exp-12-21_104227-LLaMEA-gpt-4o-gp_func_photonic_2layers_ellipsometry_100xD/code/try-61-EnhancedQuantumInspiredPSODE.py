import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedQuantumInspiredPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.9   # initial inertia weight
        self.min_w = 0.4  # minimum inertia weight
        self.max_w = 0.9  # maximum inertia weight
        self.F = 0.8   # initial mutation factor for DE
        self.CR = 0.9  # crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Quantum-inspired initialization
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
            # Dynamic adaptive inertia weight and mutation factor
            self.w = self.max_w - (self.max_w - self.min_w) * (evaluations / self.budget)
            self.F = np.random.uniform(0.5, 1)  # Randomized F for diversity
            
            # PSO update with quantum-inspired enhancement
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, lb, ub)

            # Quantum-inspired position update
            q_population = 0.5 * (population + global_best_position) + np.random.uniform(-1, 1, (self.population_size, self.dim)) * np.abs(population - global_best_position)
            q_population = np.clip(q_population, lb, ub)

            # Parallel evaluation of new positions
            with ThreadPoolExecutor() as executor:
                scores = np.array(list(executor.map(evaluate, population)))
                q_scores = np.array(list(executor.map(evaluate, q_population)))
            evaluations += 2 * self.population_size

            # Replace with quantum-inspired solutions if better
            improved_q = q_scores < scores
            population[improved_q] = q_population[improved_q]
            scores[improved_q] = q_scores[improved_q]

            # Parallel DE mutation and crossover
            def de_mutation(i):
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
                return trial, trial_score
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(de_mutation, range(self.population_size)))
            
            for i, (trial, trial_score) in enumerate(results):
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