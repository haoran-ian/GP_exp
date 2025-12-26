import numpy as np

class EnhancedHybridDELocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 * dim
        self.population_size = self.base_population_size
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.num_evaluations = 0
        self.elite_rate = 0.1  # Rate of elite preservation

    def adapt_parameters(self):
        # Adaptive parameter adjustment based on current evaluations
        self.F = 0.5 + 0.5 * np.random.rand()
        self.CR = 0.8 + 0.2 * np.random.rand()
        # Adjust population size based on progress
        progress_ratio = self.num_evaluations / self.budget
        self.population_size = max(4 * self.dim, int(self.base_population_size * (1 - progress_ratio)))

    def crowding_distance(self, population, fitness):
        # Compute crowding distance to promote diversity
        distances = np.zeros(len(population))
        order = np.argsort(fitness)
        for i in range(self.dim):
            sorted_pop = population[order, i]
            sorted_fit = fitness[order]
            distances[order[0]] = distances[order[-1]] = np.inf
            for j in range(1, len(population) - 1):
                if sorted_fit[-1] > sorted_fit[0]:
                    distances[order[j]] += (sorted_pop[j + 1] - sorted_pop[j - 1]) / (sorted_fit[-1] - sorted_fit[0])
        return distances

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.num_evaluations += self.population_size
        
        while self.num_evaluations < self.budget:
            self.adapt_parameters()
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite_size = max(1, int(self.elite_rate * self.population_size))
            
            crowding_distances = self.crowding_distance(population, fitness)
            for i in range(elite_size, self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                selected_idxs = np.random.choice(idxs, 3, replace=False, p=crowding_distances / crowding_distances.sum())
                a, b, c = population[selected_idxs]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                
                if self.num_evaluations < self.budget:
                    sigma = 0.1 * (ub - lb)
                    local_candidates = trial + np.random.randn(5, self.dim) * sigma
                    local_candidates = np.clip(local_candidates, lb, ub)
                    local_fitnesses = np.array([func(cand) for cand in local_candidates])
                    self.num_evaluations += 5
                    best_local_idx = np.argmin(local_fitnesses)
                    if local_fitnesses[best_local_idx] < trial_fitness:
                        population[i], fitness[i] = local_candidates[best_local_idx], local_fitnesses[best_local_idx]

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness