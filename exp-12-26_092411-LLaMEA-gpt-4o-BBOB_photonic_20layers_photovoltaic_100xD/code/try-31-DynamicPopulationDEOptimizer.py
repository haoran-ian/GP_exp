import numpy as np

class DynamicPopulationDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.max_population_size = 20 * dim
        self.population_size = self.initial_population_size
        self.F = 0.8
        self.CR = 0.9
        self.num_evaluations = 0
        self.elite_rate = 0.1

    def adapt_parameters(self):
        # Adaptive parameter adjustment based on progress
        self.F = 0.5 + 0.5 * np.random.rand()
        self.CR = 0.8 + 0.2 * np.random.rand()

    def calculate_diversity(self, population):
        # Calculate population diversity
        return np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))

    def adjust_population_size(self, progress):
        if progress > 0.1:
            self.population_size = min(self.max_population_size, self.population_size + self.dim)
        elif progress < 0.01:
            self.population_size = max(self.initial_population_size, self.population_size - self.dim)
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.num_evaluations += self.population_size
        best_fitness = np.min(fitness)

        while self.num_evaluations < self.budget:
            self.adapt_parameters()
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite_size = max(1, int(self.elite_rate * self.population_size))
            
            for i in range(elite_size, self.population_size):
                idxs = [idx for idx in np.random.choice(self.population_size, 5, replace=False) if idx != i]
                a, b, c, d, e = population[idxs]
                mutant = np.clip(a + self.F * (b - c + d - e), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i], new_fitness[i] = trial, trial_fitness

            progress = np.abs(best_fitness - np.min(new_fitness)) / (1 + np.abs(best_fitness))
            self.adjust_population_size(progress)
            best_idx = np.argmin(new_fitness)
            best_fitness = new_fitness[best_idx]
            population, fitness = new_population, new_fitness

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]
        return best_solution, best_fitness