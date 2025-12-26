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

        # Adaptive parameters
        rs_factor = 0.2  # Initial probability of performing random search
        improvement_threshold = 0.01  # Threshold for performance improvement
        last_best_fitness = np.min(fitness)

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

            # Guided Random Search based on adaptive probability
            if evaluations < self.budget and np.random.rand() < rs_factor:
                rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = rand_fitness
                    population[worst_idx] = rand_ind

            # Adapt rs_factor based on the improvement of the best solution
            current_best_fitness = np.min(fitness)
            if current_best_fitness < last_best_fitness - improvement_threshold:
                rs_factor *= 0.9  # Reduce random search probability if improving
            else:
                rs_factor = min(0.5, rs_factor * 1.1)  # Increase it if not improving

            last_best_fitness = current_best_fitness

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]