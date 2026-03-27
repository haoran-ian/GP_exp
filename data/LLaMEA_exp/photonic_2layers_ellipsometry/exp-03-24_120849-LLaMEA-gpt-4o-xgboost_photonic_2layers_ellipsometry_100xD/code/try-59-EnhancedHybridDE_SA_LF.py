import numpy as np
from scipy.stats import levy
from sklearn.cluster import KMeans

class EnhancedHybridDE_SA_LF:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, population_factor=10, cluster_count=3, crossover_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.F = F  # Differential evolution parameter
        self.CR = CR  # Crossover probability
        self.T0 = T0  # Initial temperature for Simulated Annealing
        self.alpha = alpha  # Cooling rate
        self.population_factor = population_factor  # Scaling factor for population size
        self.cluster_count = cluster_count  # Number of clusters for adaptive subpopulations
        self.crossover_decay = crossover_decay  # Decay factor for crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        population = lb + (ub - lb) * np.random.beta(1.5, 1.5, (pop_size, self.dim))  # Modified initial sampling
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0
        CR = self.CR

        while eval_count < self.budget:
            # Clustering for subpopulation management
            kmeans = KMeans(n_clusters=self.cluster_count)
            population_labels = kmeans.fit_predict(population)
            
            for cluster in range(self.cluster_count):
                cluster_indices = np.where(population_labels == cluster)[0]
                for i in cluster_indices:
                    # Adaptive Differential Evolution mutation and crossover
                    F_adaptive = self.F * (1 - eval_count / self.budget) + 0.1
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

                    # Selection and Simulated Annealing acceptance
                    if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                        population[i] = trial
                        fitness[i] = trial_fitness

                        if trial_fitness < best_fitness:
                            best = trial
                            best_fitness = trial_fitness
                
                    # Lévy Flight for enhanced exploration
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

            # Temperature cooling for simulated annealing
            T *= self.alpha ** (eval_count / self.budget)  # Adaptive temperature
            # Dynamic crossover decay
            CR *= self.crossover_decay

        return best, best_fitness