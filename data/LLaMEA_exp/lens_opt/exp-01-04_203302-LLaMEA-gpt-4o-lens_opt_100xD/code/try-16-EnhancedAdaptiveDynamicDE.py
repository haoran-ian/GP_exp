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

    def opposition_based_learning(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        opp_population = lb + ub - self.population
        return opp_population

    def __call__(self, func):
        self.initialize_population(func.bounds)
        fitness = self.evaluate_population(func)
        
        while self.func_evals < self.budget:
            # Incorporate opposition-based learning
            if self.func_evals < self.budget / 2:
                opp_population = self.opposition_based_learning(func.bounds)
                opp_fitness = np.apply_along_axis(func, 1, opp_population)
                self.func_evals += self.population_size
                for idx in range(self.population_size):
                    if opp_fitness[idx] < fitness[idx]:
                        self.population[idx] = opp_population[idx]
                        fitness[idx] = opp_fitness[idx]
                        if opp_fitness[idx] < self.best_fitness:
                            self.best_fitness = opp_fitness[idx]
                            self.best_solution = opp_population[idx]
            
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

            self.adapt_parameters(self.func_evals // self.population_size)

        return self.best_solution, self.best_fitness