import numpy as np

class EnhancedHyperAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_pop_size = 10 + 5 * dim
        self.pop_size = self.init_pop_size
        self.cr = np.random.uniform(0.5, 0.9, self.pop_size)
        self.mutation_factor = np.random.uniform(0.5, 0.9, self.pop_size)
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb

    def adaptive_differential_evolution_step(self, bounds, func):
        new_population = np.copy(self.population)
        for i in range(self.pop_size):
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            cr_i = self.cr[i]
            mut_factor_i = self.mutation_factor[i]
            # Multi-mutation strategy
            mutant_vector_1 = x0 + mut_factor_i * (x1 - x2)
            mutant_vector_2 = x0 + mut_factor_i * (x2 - x1)
            mutant_vector = mutant_vector_1 if np.random.rand() > 0.5 else mutant_vector_2
            mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < cr_i
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness < func(self.population[i]):
                new_population[i] = trial_vector
                self.cr[i] = min(self.cr[i] + 0.05, 0.9)  # Adjusted increment to 0.05
                self.mutation_factor[i] = min(self.mutation_factor[i] + 0.05, 0.9)  # Adjusted increment to 0.05
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = trial_fitness
            else:
                self.cr[i] = max(self.cr[i] - 0.05, 0.5)  # Adjusted decrement to 0.05
                self.mutation_factor[i] = max(self.mutation_factor[i] - 0.05, 0.5)  # Adjusted decrement to 0.05
        self.population = new_population

    def simulated_annealing_local_search(self, bounds, func):
        for i in range(self.pop_size):
            candidate = np.copy(self.population[i])
            gradient = self.estimate_gradient(candidate, func)
            stochastic_step = np.random.uniform(0.005, 0.02) * (bounds.ub - bounds.lb)
            candidate -= stochastic_step * gradient
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            acceptance_prob = np.exp(-(candidate_fitness - func(self.population[i])) / self.temperature)
            if candidate_fitness < func(self.population[i]) or np.random.rand() < acceptance_prob:
                self.population[i] = candidate
                if candidate_fitness < self.best_fitness:
                    self.best_solution = candidate
                    self.best_fitness = candidate_fitness
        self.temperature *= self.cooling_rate

    def estimate_gradient(self, x, func, epsilon=1e-8):
        gradient = np.zeros_like(x)
        fx = func(x)
        for i in range(self.dim):
            x_step = np.copy(x)
            x_step[i] += epsilon
            gradient[i] = (func(x_step) - fx) / epsilon
        return gradient

    def dynamic_population_resizing(self, evaluations):
        if evaluations > self.budget / 2:
            self.pop_size = max(self.init_pop_size // 2, 4)
            self.cr = np.resize(self.cr, self.pop_size)
            self.mutation_factor = np.resize(self.mutation_factor, self.pop_size)
            self.population = self.population[:self.pop_size]

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        evaluations = 0
        while evaluations < self.budget:
            self.adaptive_differential_evolution_step(bounds, func)
            evaluations += self.pop_size
            if evaluations < self.budget:
                self.simulated_annealing_local_search(bounds, func)
                evaluations += self.pop_size
            self.dynamic_population_resizing(evaluations)
        return self.best_solution