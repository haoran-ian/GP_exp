import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedAdaptiveQuantumInspiredPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.5   # inertia weight
        self.F = 0.8   # mutation factor for DE
        self.CR = 0.9  # crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        q_population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-np.abs(ub-lb), np.abs(ub-lb), (population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = population_size

        def evaluate(ind):
            return func(ind)

        while evaluations < self.budget:
            # Adaptive inertia weight and mutation factor
            self.w = 0.9 - 0.5 * (evaluations / self.budget)
            self.F = 0.1 + 0.7 * (1 - evaluations / self.budget)
            # Adaptive crossover probability
            self.CR = 0.7 + 0.3 * (evaluations / self.budget)
            # Dynamic population size adjustment
            population_size = self.initial_population_size + int((self.budget - evaluations) / self.budget * 10)
            # PSO update with quantum-inspired enhancement
            r1, r2 = np.random.rand(population_size, self.dim), np.random.rand(population_size, self.dim)
            velocities[:population_size] = (self.w * velocities[:population_size] +
                                            self.c1 * r1 * (personal_best_positions[:population_size] - population[:population_size]) +
                                            self.c2 * r2 * (global_best_position - population[:population_size]))
            population[:population_size] += velocities[:population_size] * (0.9 + 0.1 * (evaluations / self.budget))
            population[:population_size] = np.clip(population[:population_size], lb, ub)

            # Quantum-inspired position update
            q_population[:population_size] = 0.5 * (population[:population_size] + global_best_position) + \
                                             np.random.uniform(-1, 1, (population_size, self.dim)) * np.abs(population[:population_size] - global_best_position)
            q_population[:population_size] = np.clip(q_population[:population_size], lb, ub)

            # Parallel evaluation of new positions
            with ThreadPoolExecutor() as executor:
                scores = np.array(list(executor.map(evaluate, population[:population_size])))
                q_scores = np.array(list(executor.map(evaluate, q_population[:population_size])))
            evaluations += 2 * population_size

            # Replace with quantum-inspired solutions if better
            improved_q = q_scores < scores
            population[:population_size][improved_q] = q_population[:population_size][improved_q]
            scores[improved_q] = q_scores[improved_q]

            # DE mutation
            for i in range(population_size):
                indices = np.arange(population_size)
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
            personal_best_positions[:population_size][improved] = population[:population_size][improved]
            personal_best_scores[improved] = scores[improved]
            if np.min(scores) < global_best_score:
                global_best_position = population[np.argmin(scores)]
                global_best_score = np.min(scores)
            
        return global_best_position, global_best_score