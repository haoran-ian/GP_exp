import numpy as np

class AdaptiveDEWithDynamicPopulation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.8
        self.CR = 0.9
        self.num_evaluations = 0
        self.elite_rate = 0.1
        self.resizing_factor = 0.8

    def adapt_parameters(self, diversity):
        self.F = 0.5 + 0.5 * np.random.rand()
        self.CR = 0.8 + 0.2 * np.random.rand()
        if diversity < 0.1:
            self.population_size = int(self.population_size * self.resizing_factor)
            self.population_size = max(4, self.population_size)
        else:
            self.population_size = min(int(self.population_size / self.resizing_factor), self.initial_population_size)

    def calculate_diversity(self, population):
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
            elite_size = max(1, int(self.elite_rate * len(population)))
            
            for i in range(elite_size, len(population)):
                idxs = [idx for idx in range(len(population)) if idx != i]
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

            if len(population) < self.initial_population_size:
                additional_population = np.random.rand(self.initial_population_size - len(population), self.dim) * (ub - lb) + lb
                additional_fitness = np.array([func(ind) for ind in additional_population])
                self.num_evaluations += len(additional_population)
                population = np.vstack((population, additional_population))
                fitness = np.hstack((fitness, additional_fitness))
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices[:self.initial_population_size]]
                fitness = fitness[sorted_indices[:self.initial_population_size]]
            
            best_idx = np.argmin(fitness)
            best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness