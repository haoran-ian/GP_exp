import numpy as np
from sklearn.cluster import KMeans

class AdaptiveMultiPhaseOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Adjusted cooling rate for better exploitation
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1  # Introduced elitism to preserve top solutions
        self.diversity_threshold = 0.1
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        num_elites = int(self.elitism_rate * self.population_size)
        
        while budget_used < self.budget:
            # Sort for elitism
            sorted_indices = np.argsort(fitness)
            elites = population[sorted_indices[:num_elites]]
            
            # Differential Evolution: mutate and crossover
            for i in range(self.population_size):
                if i < num_elites:
                    continue  # Skip elites
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing: accept based on Metropolis criterion
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Diversity-based adaptive mutation factor adjustment
            km = KMeans(n_clusters=5)
            clusters = km.fit_predict(population)
            cluster_centers = km.cluster_centers_
            cluster_distances = [np.linalg.norm(population[clusters == k] - cluster_centers[k], axis=1).mean() for k in range(5)]
            diversity = np.mean(cluster_distances)
            if diversity < self.diversity_threshold * (ub - lb).mean():
                self.mutation_factor *= 1.1  # Adjust adaptation factor

            # Reintroduce elites
            population[:num_elites] = elites

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]