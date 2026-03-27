import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.92
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.neighborhood_size = 2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        def variable_neighborhood_search(ind):
            neighbors = np.random.uniform(max(lb, ind - self.neighborhood_size), 
                                          min(ub, ind + self.neighborhood_size), 
                                          (5, self.dim))
            neighbors_fitness = np.array([func(nei) for nei in neighbors])
            budget_used = 5
            best_neighbor_idx = np.argmin(neighbors_fitness)
            return neighbors[best_neighbor_idx], neighbors_fitness[best_neighbor_idx], budget_used

        while budget_used < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor * (1 - budget_used / self.budget)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.1 * np.cos(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
                
                # Apply variable neighborhood search to refine improvements
                new_ind, new_fitness, additional_budget = variable_neighborhood_search(population[i])
                budget_used += additional_budget
                if new_fitness < fitness[i]:
                    population[i] = new_ind
                    fitness[i] = new_fitness
            
            self.temperature *= self.cooling_rate

            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]