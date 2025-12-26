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
        F_base = 0.5  # Base differential weight
        CR = 0.9  # Crossover probability
        elitism_rate = 0.1

        while evaluations < self.budget:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Apply elitism
            elite_size = max(1, int(elitism_rate * population_size))
            elites = population[:elite_size]
            elites_fitness = fitness[:elite_size]

            for i in range(elite_size, population_size):
                if evaluations >= self.budget:
                    break

                # Adaptive mutation strategy
                F = F_base + np.random.rand() * (1.0 - F_base)
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

            # Random Search
            if evaluations < self.budget:
                rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[-1]:
                    fitness[-1] = rand_fitness
                    population[-1] = rand_ind

            # Reinforce elitism
            population[:elite_size] = elites
            fitness[:elite_size] = elites_fitness

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]