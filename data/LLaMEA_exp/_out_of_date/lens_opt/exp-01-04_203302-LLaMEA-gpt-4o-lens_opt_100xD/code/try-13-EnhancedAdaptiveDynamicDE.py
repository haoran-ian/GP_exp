import numpy as np

class EnhancedAdaptiveDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, int(budget / (10 * dim)))
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.func_evals = 0
        self.no_improvement_count = 0
        self.local_search_threshold = 5

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf

    def evaluate_population(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.func_evals += self.population_size
        return fitness

    def select_parents(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        return np.random.choice(candidates, 3, replace=False)

    def mutate(self, idx, parents):
        x1, x2, x3 = self.population[parents]
        return self.population[idx] + self.F * (x1 - x2 + x3 - self.population[idx])

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def adapt_parameters(self, iter_num):
        self.F = np.clip(0.5 + 0.3 * (np.sin(iter_num / 8)), 0.1, 0.9)
        self.CR = np.clip(0.8 + 0.2 * (np.cos(iter_num / 15)), 0.0, 1.0)

    def local_search(self, func, best_idx):
        perturbation_strength = 0.1 * (func.bounds.ub - func.bounds.lb)
        local_solution = self.population[best_idx] + np.random.uniform(-perturbation_strength, perturbation_strength)
        local_fitness = func(local_solution)
        self.func_evals += 1
        return local_solution, local_fitness

    def __call__(self, func):
        self.initialize_population(func.bounds)
        fitness = self.evaluate_population(func)
        
        while self.func_evals < self.budget:
            improvement = False
            for idx in range(self.population_size):
                parents = self.select_parents(idx)
                mutant = self.mutate(idx, parents)
                trial = self.crossover(self.population[idx], mutant)

                trial_fitness = func(trial)
                self.func_evals += 1
                if trial_fitness < fitness[idx]:
                    self.population[idx] = trial
                    fitness[idx] = trial_fitness
                    improvement = True

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

            if not improvement:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0

            if self.no_improvement_count >= self.local_search_threshold:
                best_idx = np.argmin(fitness)
                local_solution, local_fitness = self.local_search(func, best_idx)
                if local_fitness < self.best_fitness:
                    self.best_fitness = local_fitness
                    self.best_solution = local_solution
                self.no_improvement_count = 0

            self.adapt_parameters(self.func_evals // self.population_size)

        return self.best_solution, self.best_fitness