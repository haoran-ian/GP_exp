import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedAdaptiveQuantumInspiredPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.min_population_size = 5
        self.max_population_size = 40
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.5   # inertia weight
        self.F = 0.8   # mutation factor for DE
        self.CR = 0.9  # crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Population initialization
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
            # Dynamically adjust population size
            population_size = int(self.min_population_size + (self.max_population_size - self.min_population_size) * (1 - evaluations / self.budget))
            population_size = max(self.min_population_size, min(self.max_population_size, population_size))
            # Resize arrays based on new population size
            if len(population) != population_size:
                indices = np.random.choice(len(population), population_size, replace=False)
                population = population[indices]
                q_population = q_population[indices]
                velocities = velocities[indices]
                personal_best_positions = personal_best_positions[indices]
                personal_best_scores = personal_best_scores[indices]

            # Adaptive inertia weight and mutation factor
            self.w = 0.9 - 0.5 * (evaluations / self.budget)
            self.F = 0.1 + 0.7 * (1 - evaluations / self.budget)

            # PSO update with quantum-inspired enhancement
            r1, r2 = np.random.rand(population_size, self.dim), np.random.rand(population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, lb, ub)

            # Quantum-inspired position update
            q_population = 0.5 * (population + global_best_position) + np.random.uniform(-1, 1, (population_size, self.dim)) * np.abs(population - global_best_position)
            q_population = np.clip(q_population, lb, ub)

            # Parallel evaluation of new positions
            with ThreadPoolExecutor() as executor:
                scores = np.array(list(executor.map(evaluate, population)))
                q_scores = np.array(list(executor.map(evaluate, q_population)))
            evaluations += 2 * population_size

            # Replace with quantum-inspired solutions if better
            improved_q = q_scores < scores
            population[improved_q] = q_population[improved_q]
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
            personal_best_positions[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if np.min(scores) < global_best_score:
                global_best_position = population[np.argmin(scores)]
                global_best_score = np.min(scores)
            
        return global_best_position, global_best_score