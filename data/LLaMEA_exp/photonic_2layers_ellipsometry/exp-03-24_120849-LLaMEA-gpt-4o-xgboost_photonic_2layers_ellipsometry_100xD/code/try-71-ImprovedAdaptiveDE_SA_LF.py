import numpy as np
from scipy.stats import levy
from sklearn.cluster import KMeans

class ImprovedAdaptiveDE_SA_LF:
    def __init__(self, budget, dim, F_init=0.9, CR_init=0.7, T0=1000, alpha=0.94, population_factor=12, max_cluster_count=5, crossover_decay=0.98, levy_prob=0.15):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init  # Initial Differential evolution parameter
        self.CR_init = CR_init  # Initial Crossover probability
        self.T0 = T0  # Initial temperature for Simulated Annealing
        self.alpha = alpha  # Cooling rate
        self.population_factor = population_factor  # Scaling factor for population size
        self.max_cluster_count = max_cluster_count  # Maximum number of clusters for adaptive subpopulations
        self.crossover_decay = crossover_decay  # Decay factor for crossover probability
        self.levy_prob = levy_prob  # Probability of performing Levy flights

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        population = lb + (ub - lb) * np.random.rand(pop_size, self.dim)  # Random initial sampling
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0
        F = self.F_init
        CR = self.CR_init

        while eval_count < self.budget:
            # Dynamic clustering for subpopulation management
            cluster_count = min(self.max_cluster_count, int(1 + np.floor(eval_count / self.budget * self.max_cluster_count)))
            kmeans = KMeans(n_clusters=cluster_count, n_init=3)
            population_labels = kmeans.fit_predict(population)
            
            for cluster in range(cluster_count):
                cluster_indices = np.where(population_labels == cluster)[0]
                for i in cluster_indices:
                    # Self-adaptive DE mutation and crossover
                    F_adaptive = F * (1 - eval_count / self.budget) + 0.1
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
                    if np.random.rand() < self.levy_prob:
                        levy_step = levy.rvs(size=self.dim) * 0.01  # Control the scale of Levy steps
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
            T *= self.alpha
            # Dynamic crossover decay
            CR = self.CR_init + (CR - self.CR_init) * self.crossover_decay
            # Adaptive scaling of F
            F = self.F_init + (F - self.F_init) * (1 - eval_count / self.budget)

        return best, best_fitness