import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize population
        population_size = 10
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Differential Evolution parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        def adaptive_FCR(evals):
            # Adapt F and CR based on remaining budget
            return 0.5 + 0.3 * (1 - evals / self.budget), 0.8 + 0.1 * (evals / self.budget)

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                F, CR = adaptive_FCR(evaluations)

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Pattern-based Local Search
            if evaluations < self.budget:
                base = population[np.argmin(fitness)]
                perturbation = np.random.uniform(-0.1, 0.1, self.dim) * (func.bounds.ub - func.bounds.lb)
                trial = np.clip(base + perturbation, func.bounds.lb, func.bounds.ub)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = trial_fitness
                    population[worst_idx] = trial

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]