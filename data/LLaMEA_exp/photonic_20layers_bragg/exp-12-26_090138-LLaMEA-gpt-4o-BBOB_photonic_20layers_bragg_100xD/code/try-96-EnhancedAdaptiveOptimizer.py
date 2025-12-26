import numpy as np

class EnhancedAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        population_size = 15 * self.dim
        F_base = 0.6
        CR_base = 0.8
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Memory for best solutions and success statistics
        best_mem = []
        success_rates = np.zeros(population_size)
        
        # Initialize elitist archive
        archive = []
        
        # Clustering parameters
        niche_threshold = 0.1 * (bounds[:, 1] - bounds[:, 0])

        while eval_count < self.budget:
            # Dynamic clustering to maintain diversity
            clusters = self.cluster_population(population, niche_threshold)

            for i in range(population_size):
                # Dynamic parameter adjustment based on success rate
                F = F_base + np.tanh(success_rates[i] - 0.5)
                CR = CR_base + np.tanh(success_rates[i] - 0.5)

                # Mutation and crossover
                cluster_indices = clusters[i]
                indices = np.random.choice(cluster_indices, 3, replace=False)
                a, b, c = population[indices]
                if archive and np.random.rand() < 0.2:  # Use archive-guided mutation
                    archive_indx = np.random.randint(len(archive))
                    mutant = np.clip(a + F * (b - c + archive[archive_indx] - a), bounds[:, 0], bounds[:, 1])
                else:
                    mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate the trial candidate
                f_trial = func(trial)
                eval_count += 1

                # Selection and success rate update
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_rates[i] += 1 / max(1, np.sum(success_rates))
                    if len(best_mem) < 5 or f_trial < np.max(best_mem):
                        best_mem.append(f_trial)
                        best_mem = sorted(best_mem)[:5]
                        archive.append(trial)  # Add to archive
                else:
                    success_rates[i] *= 0.9  # Decay unsuccessful attempts

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]

    def cluster_population(self, population, niche_threshold):
        num_individuals = len(population)
        clusters = [[] for _ in range(num_individuals)]
        for i in range(num_individuals):
            for j in range(num_individuals):
                if i != j and np.linalg.norm(population[i] - population[j]) < niche_threshold:
                    clusters[i].append(j)
            if not clusters[i]:  # Ensure at least self presence
                clusters[i].append(i)
        return clusters