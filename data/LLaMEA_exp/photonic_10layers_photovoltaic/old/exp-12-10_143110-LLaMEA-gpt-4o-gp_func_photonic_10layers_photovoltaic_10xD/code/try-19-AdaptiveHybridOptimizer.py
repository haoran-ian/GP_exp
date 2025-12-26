import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 + 5 * dim
        self.cr = 0.9  # Crossover probability
        self.mutation_factor = 0.8
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.elite_size = max(1, int(0.1 * self.initial_pop_size))  # 10% elites

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.initial_pop_size, self.dim) * (ub - lb) + lb

    def differential_evolution_step(self, bounds, func):
        new_population = []
        for i in range(self.current_pop_size):
            indices = np.random.choice(range(self.current_pop_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            mutant_vector = x0 + self.mutation_factor * (x1 - x2)
            mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness < func(self.population[i]):
                new_population.append((trial_vector, trial_fitness))
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = trial_fitness
            else:
                new_population.append((self.population[i], func(self.population[i])))
        new_population.sort(key=lambda x: x[1])
        self.population = np.array([x[0] for x in new_population[:self.current_pop_size]])

    def gradient_based_local_search(self, bounds, func):
        for i in range(self.current_pop_size):
            candidate = np.copy(self.population[i])
            gradient = self.estimate_gradient(candidate, func)
            step_size = 0.01 * (bounds.ub - bounds.lb)
            candidate -= step_size * gradient
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < func(self.population[i]):
                self.population[i] = candidate
                if candidate_fitness < self.best_fitness:
                    self.best_solution = candidate
                    self.best_fitness = candidate_fitness

    def estimate_gradient(self, x, func, epsilon=1e-8):
        gradient = np.zeros_like(x)
        fx = func(x)
        for i in range(self.dim):
            x_step = np.copy(x)
            x_step[i] += epsilon
            gradient[i] = (func(x_step) - fx) / epsilon
        return gradient

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)

        evaluations = 0
        while evaluations < self.budget:
            self.current_pop_size = int(self.initial_pop_size * (1 - evaluations / self.budget))
            self.differential_evolution_step(bounds, func)
            evaluations += self.current_pop_size
            if evaluations < self.budget:
                self.gradient_based_local_search(bounds, func)
                evaluations += self.current_pop_size
        return self.best_solution