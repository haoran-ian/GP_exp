import numpy as np
from scipy.stats import levy

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.num_subpopulations = 3
        self.subpopulation_size = self.population_size // self.num_subpopulations
        self.f_base = 0.8
        self.cr_base = 0.9
        self.current_evals = 0

    def adaptive_differential_evolution(self, population, func, bounds):
        new_population = np.copy(population)
        for i in range(self.subpopulation_size):
            if self.current_evals >= self.budget:
                break
            indices = list(range(self.subpopulation_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            f = self.f_base * (1 - self.current_evals / self.budget)
            mutant = population[a] + f * (population[b] - population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            dynamic_cr = self.cr_base * (1 - self.current_evals / self.budget)
            cross_points = np.random.rand(self.dim) < dynamic_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = func(trial)
            self.current_evals += 1
            if trial_fitness < func(population[i]):
                new_population[i] = trial
        return new_population

    def levy_flight_local_search(self, individual, func, bounds):
        step_size = 0.1 * (bounds.ub - bounds.lb) * levy.rvs(size=self.dim)
        best = np.copy(individual)
        best_fitness = func(best)
        self.current_evals += 1
        for _ in range(10):  # Increased attempts for more intensification
            if self.current_evals >= self.budget:
                break
            candidate = best + step_size * np.random.normal(0, 1, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            self.current_evals += 1
            if candidate_fitness < best_fitness:
                best, best_fitness = candidate, candidate_fitness
        return best

    def __call__(self, func):
        bounds = func.bounds
        total_population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.current_evals = 0
        best_solution = total_population[0]
        best_fitness = func(best_solution)
        self.current_evals += 1

        while self.current_evals < self.budget:
            for sp in range(self.num_subpopulations):
                start_idx = sp * self.subpopulation_size
                end_idx = start_idx + self.subpopulation_size
                subpopulation = total_population[start_idx:end_idx]
                subpopulation = self.adaptive_differential_evolution(subpopulation, func, bounds)
                for i in range(self.subpopulation_size):
                    if self.current_evals >= self.budget:
                        break
                    candidate = self.levy_flight_local_search(subpopulation[i], func, bounds)
                    candidate_fitness = func(candidate)
                    self.current_evals += 1
                    if candidate_fitness < best_fitness:
                        best_solution, best_fitness = candidate, candidate_fitness
                total_population[start_idx:end_idx] = subpopulation

        return best_solution