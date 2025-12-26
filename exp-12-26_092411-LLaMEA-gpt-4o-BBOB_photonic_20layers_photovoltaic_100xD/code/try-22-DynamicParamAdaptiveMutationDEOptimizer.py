import numpy as np

class DynamicParamAdaptiveMutationDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.num_evaluations = 0
        self.elite_rate = 0.1  # Rate of elite preservation
        self.diversity_threshold = 0.05  # Threshold for diversity adaptation

    def adapt_parameters(self, diversity):
        # Adaptive parameter adjustment based on diversity
        if diversity < self.diversity_threshold:
            self.F = min(1.0, self.F + 0.05)
            self.CR = max(0.5, self.CR - 0.05)
        else:
            self.F = max(0.5, self.F - 0.05)
            self.CR = min(1.0, self.CR + 0.05)

    def calculate_diversity(self, population):
        # Calculate population diversity
        return np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.num_evaluations += self.population_size
        
        while self.num_evaluations < self.budget:
            diversity = self.calculate_diversity(population)
            self.adapt_parameters(diversity)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite_size = max(1, int(self.elite_rate * self.population_size))
            
            for i in range(elite_size, self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
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
                    sigma = 0.1 * (ub - lb) * (1 + diversity)  # Adjusted std deviation for mutation
                    local_candidates = trial + np.random.randn(3, self.dim) * sigma
                    local_candidates = np.clip(local_candidates, lb, ub)
                    local_fitnesses = np.array([func(cand) for cand in local_candidates])
                    self.num_evaluations += 3
                    best_local_idx = np.argmin(local_fitnesses)
                    if local_fitnesses[best_local_idx] < trial_fitness:
                        population[i], fitness[i] = local_candidates[best_local_idx], local_fitnesses[best_local_idx]

            best_idx = np.argmin(fitness)
            best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness