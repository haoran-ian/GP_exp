import numpy as np

class DynamicNeighborhoodADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(10, int(budget / (10 * dim)))
        self.population_size = self.initial_population_size
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

    def select_dynamic_parents(self, idx, fitness):
        num_neighbors = max(3, int(self.population_size * 0.1))  # 10% neighbors
        distances = np.linalg.norm(self.population - self.population[idx], axis=1)
        neighbor_indices = np.argsort(distances)[:num_neighbors]
        selected = neighbor_indices[neighbor_indices != idx]
        best_indices = selected[np.argsort(fitness[selected])]
        return best_indices[:3] if len(best_indices) >= 3 else np.random.choice(selected, 3, replace=False)

    def mutate(self, idx, parents):
        x1, x2, x3 = self.population[parents]
        mutant1 = x1 + self.F * (x2 - x3)
        mutant2 = self.population[idx] + self.F * (x1 - x2 + x3 - self.population[idx])
        return mutant1 if np.random.rand() < 0.5 else mutant2

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def adapt_parameters(self, iter_num, fitness):
        fitness_diversity = np.std(fitness) / np.mean(fitness)
        self.F = np.clip(0.5 + 0.3 * (np.sin(iter_num / 8)) + 0.2 * fitness_diversity, 0.3, 0.9)
        self.CR = np.clip(0.8 + 0.2 * (np.cos(iter_num / 15)), 0.0, 1.0)
        self.population_size = max(5, self.initial_population_size - iter_num // (self.budget // 20))

    def __call__(self, func):
        self.initialize_population(func.bounds)
        fitness = self.evaluate_population(func)
        
        while self.func_evals < self.budget:
            for idx in range(self.population_size):
                parents = self.select_dynamic_parents(idx, fitness)
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

            self.adapt_parameters(self.func_evals // self.population_size, fitness)

        return self.best_solution, self.best_fitness