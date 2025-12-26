import numpy as np

class SynergisticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        population_size = 20 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Memory for best solutions
        best_mem = []
        success_mem = np.array([])
        
        # Initialize archive for guided mutation
        archive = []

        while eval_count < self.budget:
            success_mem = np.append(success_mem, fitness)
            if len(success_mem) > 10 * population_size:
                success_mem = success_mem[-10 * population_size:]

            # Calculate adaptive parameters
            F = 0.5 + 0.3 * np.random.rand(population_size)
            CR = np.clip(0.3 + 0.6 * np.random.randn(population_size), 0.1, 0.9)

            for i in range(population_size):
                # Mutation and crossover
                indices = np.random.choice(population_size, 5, replace=False)
                a, b, c, d, e = population[indices]
                if len(archive) > 0 and np.random.rand() < 0.4:  # Use archive-guided mutation
                    archive_indx = np.random.randint(len(archive))
                    mutant = a + F[i] * (b - c + archive[archive_indx] - d)
                else:
                    mutant = a + F[i] * (b - c + d - e)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate the trial candidate
                f_trial = func(trial)
                eval_count += 1

                # Selection and success memory update
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    archive.append(trial)  # Add successful candidate to archive
                    archive = archive[-population_size:]  # Limit archive size

                    # Update success memory
                    best_mem.append(f_trial)
                    best_mem = sorted(best_mem)[:5]

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]