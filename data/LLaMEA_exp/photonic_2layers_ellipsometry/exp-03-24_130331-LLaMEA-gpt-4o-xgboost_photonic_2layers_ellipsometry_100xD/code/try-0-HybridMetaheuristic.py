import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.current_evals = 0

    def differential_evolution(self, population, func, bounds):
        new_population = np.copy(population)
        for i in range(self.population_size):
            if self.current_evals >= self.budget:
                break
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + self.f * (population[b] - population[c])
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
        step_size = 0.1 * (bounds.ub - bounds.lb)
        best = np.copy(individual)
        best_fitness = func(best)
        self.current_evals += 1
        for _ in range(5):  # Local search attempts
            if self.current_evals >= self.budget:
                break
            candidate = best + step_size * np.random.uniform(-1, 1, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            self.current_evals += 1
            if candidate_fitness < best_fitness:
                best, best_fitness = candidate, candidate_fitness
        return best

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.current_evals = 0
        best_solution = population[0]
        best_fitness = func(best_solution)
        self.current_evals += 1

        while self.current_evals < self.budget:
            population = self.differential_evolution(population, func, bounds)
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                candidate = self.local_search(population[i], func, bounds)
                candidate_fitness = func(candidate)
                self.current_evals += 1
                if candidate_fitness < best_fitness:
                    best_solution, best_fitness = candidate, candidate_fitness

        return best_solution