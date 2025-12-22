import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HybridQuantumInspiredSelfAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1_init = 2.5   # initial cognitive component
        self.c2_init = 0.5   # initial social component
        self.inertia_weight_init = 0.9
        self.inertia_weight_final = 0.4
        self.F = 0.8   # mutation factor for DE
        self.CR = 0.9  # crossover probability for DE

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
            # Dynamic inertia weight, cognitive and social components
            progress = evaluations / self.budget
            inertia_weight = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * progress
            c1 = self.c1_init * (1 - progress) + 0.5 * progress
            c2 = self.c2_init * (1 - progress) + 2.5 * progress

            # PSO update with quantum-inspired enhancement
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          c1 * r1 * (personal_best_positions - population) +
                          c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, lb, ub)

            # Quantum-inspired position update
            q_population = 0.5 * (population + global_best_position) + np.random.uniform(-1, 1, (self.population_size, self.dim)) * np.abs(population - global_best_position)
            q_population = np.clip(q_population, lb, ub)

            with ThreadPoolExecutor() as executor:
                scores = np.array(list(executor.map(evaluate, population)))
                q_scores = np.array(list(executor.map(evaluate, q_population)))
            evaluations += 2 * self.population_size

            # Replace with quantum-inspired solutions if better
            improved_q = q_scores < scores
            population[improved_q] = q_population[improved_q]
            scores[improved_q] = q_scores[improved_q]

            # DE mutation and crossover
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

            # Update personal and global best
            improved = scores < personal_best_scores
            personal_best_positions[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if np.min(scores) < global_best_score:
                global_best_position = population[np.argmin(scores)]
                global_best_score = np.min(scores)

        return global_best_position, global_best_score