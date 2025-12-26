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

        # Adaptive Differential Evolution parameters
        F = 0.5 + np.random.rand() * 0.5  # Adaptive differential weight
        CR = 0.5 + np.random.rand() * 0.5  # Adaptive crossover probability

        def local_search(individual):
            local_step_size = 0.1 * (func.bounds.ub - func.bounds.lb)
            perturbation = np.random.uniform(-local_step_size, local_step_size, self.dim)
            neighbor = np.clip(individual + perturbation, func.bounds.lb, func.bounds.ub)
            return neighbor

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

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

                # Local search
                if evaluations < self.budget:
                    neighbor = local_search(population[i])
                    neighbor_fitness = func(neighbor)
                    evaluations += 1

                    if neighbor_fitness < fitness[i]:
                        population[i] = neighbor
                        fitness[i] = neighbor_fitness

            # Random Search
            if evaluations < self.budget:
                rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = rand_fitness
                    population[worst_idx] = rand_ind

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]