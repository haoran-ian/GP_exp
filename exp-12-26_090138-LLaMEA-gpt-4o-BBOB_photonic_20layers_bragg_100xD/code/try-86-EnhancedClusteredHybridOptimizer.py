import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

class EnhancedClusteredHybridOptimizer:
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
        
        # Memory for best solutions
        best_mem = []
        success_mem = []
        
        # Initialize elitist archive
        archive = []
        
        def update_archive(candidate):
            archive.append(candidate)
            if len(archive) > 10 * self.dim:  # Limit the size of the archive
                archive.pop(0)  # Remove the oldest entry

        while eval_count < self.budget:
            # Hierarchical clustering for population reorganization
            if eval_count % (3 * population_size) == 0:
                distance_matrix = squareform(pdist(population))
                clustering = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='complete')
                labels = clustering.fit_predict(distance_matrix)
                
                # Balance clusters by moving individuals if necessary
                for cluster_id in range(5):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    if len(cluster_indices) < 2:  # Ensure minimum cluster size
                        continue
                    sub_population = population[cluster_indices]
                    sub_fitness = fitness[cluster_indices]
                    
                    best_sub_idx = np.argmin(sub_fitness)
                    best_sub_candidate = sub_population[best_sub_idx]
                    
                    if len(archive) > 0:
                        archive_candidate = archive[np.random.randint(len(archive))]
                        if func(archive_candidate) < func(best_sub_candidate):
                            best_sub_candidate = archive_candidate
                    
                    for idx in cluster_indices:
                        if idx != cluster_indices[best_sub_idx]:
                            population[idx] = best_sub_candidate + np.random.normal(0, 0.1, self.dim)
                            population[idx] = np.clip(population[idx], bounds[:, 0], bounds[:, 1])
                            fitness[idx] = func(population[idx])
                            eval_count += 1
                    
            for i in range(population_size):
                # Adaptive parameter selection based on success history
                if success_mem:
                    F = np.mean(success_mem)
                    F = np.clip(F + np.random.normal(0, 0.05), 0.4, 1.0)
                else:
                    F = F_base + np.random.uniform(-0.1, 0.3)

                CR = CR_base + np.random.uniform(-0.1, 0.1)

                # Mutation and crossover
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                if len(archive) > 0 and np.random.rand() < 0.2:  # Use archive-guided mutation
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

                # Selection and success memory update
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_mem.append(F)
                    if len(success_mem) > 5:
                        success_mem.pop(0)
                    if len(best_mem) < 5 or f_trial < np.max(best_mem):
                        best_mem.append(f_trial)
                        best_mem = sorted(best_mem)[:5]
                        update_archive(trial)

                # Memory-based local search
                if eval_count < self.budget and np.random.rand() < 0.3:
                    perturbation_scale = 0.05 * np.random.uniform(0.5, 1.5)
                    perturbation = np.random.normal(0, perturbation_scale, self.dim)
                    new_trial = population[i] + perturbation
                    new_trial = np.clip(new_trial, bounds[:, 0], bounds[:, 1])
                    f_new_trial = func(new_trial)
                    eval_count += 1
                    if f_new_trial < f_trial:
                        population[i] = new_trial
                        fitness[i] = f_new_trial
                        if len(best_mem) < 5 or f_new_trial < np.max(best_mem):
                            best_mem.append(f_new_trial)
                            best_mem = sorted(best_mem)[:5]
                            update_archive(new_trial)

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]