import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize population
        population_size = 10
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Adaptive parameters
        F_base = 0.8  # Initial differential weight
        CR_base = 0.9  # Initial crossover probability
        diversity_threshold = 0.1

        def calculate_diversity():
            centroid = np.mean(population, axis=0)
            diversity = np.mean(np.linalg.norm(population - centroid, axis=1))
            return diversity

        while evaluations < self.budget:
            best_idx = np.argmin(fitness)  # Track best solution index
            diversity = calculate_diversity()

            if diversity < diversity_threshold:
                F = F_base * (1 + np.random.rand() * 0.5)  # Increase exploration
                CR = CR_base * (1 - np.random.rand() * 0.5)  # Decrease exploitation
            else:
                F = F_base
                CR = CR_base * (diversity / (diversity + 1))  # Adjust CR based on diversity
            
            # Dynamic population adjustment
            if evaluations % 50 == 0 and evaluations > 0:
                convergence_rate = np.std(fitness) / np.mean(fitness)
                if convergence_rate < 0.01:
                    population = np.vstack((population, np.random.uniform(func.bounds.lb, func.bounds.ub, (5, self.dim))))
                    fitness = np.concatenate((fitness, [func(ind) for ind in population[-5:]]))
                    evaluations += 5
                    population_size += 5
                   
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

            # Elitism: Preserve best solution
            if evaluations < self.budget:
                rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = rand_fitness
                    population[worst_idx] = rand_ind

            # Ensure best solution is retained
            population[best_idx] = population[np.argmin(fitness)]
            fitness[best_idx] = min(fitness)

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]