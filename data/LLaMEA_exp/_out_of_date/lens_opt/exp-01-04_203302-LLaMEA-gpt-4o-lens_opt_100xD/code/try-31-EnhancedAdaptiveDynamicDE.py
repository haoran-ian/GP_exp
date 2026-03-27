import numpy as np

class EnhancedAdaptiveDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(10, int(budget / (10 * dim)))
        self.population_size = self.initial_population_size
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.func_evals = 0
        self.fitness_history = []

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
        # Adaptive learning rates based on historical performance
        if len(self.fitness_history) > 1:
            improvement_rate = (self.fitness_history[-2] - self.fitness_history[-1]) / self.fitness_history[-2]
            self.F = np.clip(self.F + 0.1 * improvement_rate, 0.1, 0.9)
            self.CR = np.clip(self.CR + 0.05 * improvement_rate, 0.0, 1.0)
        
        # Dynamic population resizing based on iteration number
        self.population_size = max(5, self.initial_population_size - iter_num // (self.budget // 20))

    def __call__(self, func):
        self.initialize_population(func.bounds)
        fitness = self.evaluate_population(func)
        self.fitness_history.append(np.mean(fitness))
        
        while self.func_evals < self.budget:
            for idx in range(self.population_size):
                parents = self.select_parents(idx)
                mutant = self.mutate(idx, parents)
                trial = self.crossover(self.population[idx], mutant)
                
                trial_fitness = func(trial)
                self.func_evals += 1
                if trial_fitness < fitness[idx]:
                    self.population[idx] = trial
                    fitness[idx] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

            self.fitness_history.append(np.mean(fitness))
            self.adapt_parameters(self.func_evals // self.population_size)

        return self.best_solution, self.best_fitness