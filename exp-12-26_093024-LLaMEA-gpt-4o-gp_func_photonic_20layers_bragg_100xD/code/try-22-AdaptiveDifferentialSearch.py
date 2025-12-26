import numpy as np

class AdaptiveDifferentialSearch:
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
        F_base = 0.5  # Base differential weight
        CR = 0.9  # Crossover probability

        def chaotic_local_search(individual):
            # Apply a simple chaotic local search for fine-tuning
            tau = 0.1  # Learning rate
            perturbation = np.random.randn(self.dim) * tau
            return np.clip(individual + perturbation, func.bounds.lb, func.bounds.ub)

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                
                # Adaptive mutation strategy
                F = F_base + np.random.rand() * 0.2
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

            # Chaotic local search on the best individual
            if evaluations < self.budget:
                best_idx = np.argmin(fitness)
                refined_candidate = chaotic_local_search(population[best_idx])
                refined_fitness = func(refined_candidate)
                evaluations += 1

                if refined_fitness < fitness[best_idx]:
                    population[best_idx] = refined_candidate
                    fitness[best_idx] = refined_fitness

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]