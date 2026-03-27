import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f_min = 0.5  # Minimum scaling factor
        self.f_max = 0.9  # Maximum scaling factor
        self.cr = 0.9  # Crossover probability
        self.current_evals = 0

    def chaotic_initialization(self, bounds):
        # Using logistic map for chaotic sequence generation for initialization
        x = np.random.rand(self.population_size, self.dim)
        r = 3.9  # Logistic map parameter
        for _ in range(100):  # Iterating the chaotic map
            x = r * x * (1 - x)
        return bounds.lb + x * (bounds.ub - bounds.lb)

    def adaptive_differential_evolution(self, population, func, bounds):
        new_population = np.copy(population)
        for i in range(self.population_size):
            if self.current_evals >= self.budget:
                break
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            # Adaptive mutation scaling based on current progress
            adaptive_f = self.f_min + (self.f_max - self.f_min) * (1 - self.current_evals / self.budget)
            mutant = population[a] + adaptive_f * (population[b] - population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = func(trial)
            self.current_evals += 1
            if trial_fitness < func(population[i]):
                new_population[i] = trial
        return new_population

    def local_search(self, individual, func, bounds):
        adaptive_step_size = 0.1 * (bounds.ub - bounds.lb) * (1 - self.current_evals / self.budget)
        best = np.copy(individual)
        best_fitness = func(best)
        self.current_evals += 1
        for _ in range(5):
            if self.current_evals >= self.budget:
                break
            candidate = best + adaptive_step_size * np.random.uniform(-1, 1, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            self.current_evals += 1
            if candidate_fitness < best_fitness:
                best, best_fitness = candidate, candidate_fitness
        return best

    def __call__(self, func):
        bounds = func.bounds
        population = self.chaotic_initialization(bounds)
        self.current_evals = 0

        # Elite preservation
        best_solution = population[0]
        best_fitness = func(best_solution)
        self.current_evals += 1

        while self.current_evals < self.budget:
            population = self.adaptive_differential_evolution(population, func, bounds)
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                candidate = self.local_search(population[i], func, bounds)
                candidate_fitness = func(candidate)
                self.current_evals += 1
                if candidate_fitness < best_fitness:
                    best_solution, best_fitness = candidate, candidate_fitness

        return best_solution