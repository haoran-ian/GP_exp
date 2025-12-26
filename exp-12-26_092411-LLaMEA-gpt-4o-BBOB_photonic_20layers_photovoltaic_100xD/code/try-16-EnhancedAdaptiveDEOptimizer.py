import numpy as np

class EnhancedAdaptiveDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.num_evaluations = 0
        self.elite_rate = 0.1

    def adapt_parameters(self, diversity):
        # Adaptive parameter adjustment based on diversity and evaluation progress
        progress = self.num_evaluations / self.budget
        self.F = 0.5 + 0.5 * np.random.rand() * (1 - progress)
        self.CR = 0.8 + 0.2 * np.random.rand() * (1 - diversity)

    def calculate_diversity(self, population):
        return np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))

    def resize_population(self, current_size):
        # Adaptive resizing based on progress
        progress = self.num_evaluations / self.budget
        return int(current_size * (1 - progress))

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
                    sigma = 0.1 * (ub - lb) * (1 + diversity)
                    local_candidates = trial + np.random.randn(3, self.dim) * sigma
                    local_candidates = np.clip(local_candidates, lb, ub)
                    local_fitnesses = np.array([func(cand) for cand in local_candidates])
                    self.num_evaluations += 3
                    best_local_idx = np.argmin(local_fitnesses)
                    if local_fitnesses[best_local_idx] < trial_fitness:
                        population[i], fitness[i] = local_candidates[best_local_idx], local_fitnesses[best_local_idx]

            self.population_size = max(elite_size, self.resize_population(self.initial_population_size))
            population = population[:self.population_size]
            fitness = fitness[:self.population_size]

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]
        return best_solution, best_fitness