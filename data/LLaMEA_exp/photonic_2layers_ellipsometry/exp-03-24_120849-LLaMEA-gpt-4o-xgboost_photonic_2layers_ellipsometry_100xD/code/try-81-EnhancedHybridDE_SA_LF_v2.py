import numpy as np
from scipy.stats import levy
from sklearn.cluster import KMeans

class EnhancedHybridDE_SA_LF_v2:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, population_factor=10, cluster_count=3, crossover_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.T0 = T0
        self.alpha = alpha
        self.population_factor = population_factor
        self.cluster_count = cluster_count
        self.crossover_decay = crossover_decay

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        population = lb + (ub - lb) * np.random.beta(0.5, 0.5, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        
        T = self.T0
        CR = self.CR

        while eval_count < self.budget:
            kmeans = KMeans(n_clusters=self.cluster_count)
            population_labels = kmeans.fit_predict(population)
            
            for cluster in range(self.cluster_count):
                cluster_indices = np.where(population_labels == cluster)[0]
                for i in cluster_indices:
                    # Adaptive mutation scaling based on fitness variance
                    F_adaptive = self.F * (1 - eval_count / self.budget) + 0.1 * np.std(fitness[cluster_indices])
                    indices = np.random.choice(cluster_indices, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = np.clip(x0 + F_adaptive * (x1 - x2), lb, ub)
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])

                    # Evaluate trial individual
                    trial_fitness = func(trial)
                    eval_count += 1

                    # Selection and fitness diversity check
                    if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                        population[i] = trial
                        fitness[i] = trial_fitness

                        if trial_fitness < best_fitness:
                            best = trial
                            best_fitness = trial_fitness
                
                    if np.random.rand() < 0.1:
                        levy_step = levy.rvs(size=self.dim)
                        new_position = np.clip(population[i] + levy_step, lb, ub)
                        new_fitness = func(new_position)
                        eval_count += 1

                        if new_fitness < fitness[i]:
                            population[i] = new_position
                            fitness[i] = new_fitness

                            if new_fitness < best_fitness:
                                best = new_position
                                best_fitness = new_fitness

            T *= self.alpha
            CR *= self.crossover_decay

        return best, best_fitness