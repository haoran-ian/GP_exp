import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.num_subpopulations = 3
        self.subpopulation_size = self.population_size // self.num_subpopulations
        self.f = 0.8
        self.cr = 0.9
        self.current_evals = 0

    def differential_evolution(self, population, func, bounds):
        new_population = np.copy(population)
        for i in range(self.subpopulation_size):
            if self.current_evals >= self.budget:
                break
            indices = list(range(self.subpopulation_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + self.f * (population[b] - population[c])
            f_dynamic = self.f * (1 - (self.current_evals / self.budget))
            mutant = population[a] + f_dynamic * (population[b] - population[c])  # Line modified for adaptive mutation factor
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            dynamic_cr = self.cr * np.cos(np.pi/2 * (self.current_evals / self.budget))  # Line modified for cosine-based crossover rate
            cross_points = np.random.rand(self.dim) < dynamic_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = func(trial)
            self.current_evals += 1
            if trial_fitness < func(population[i]):
                new_population[i] = trial
        return new_population

    def local_search(self, individual, func, bounds):
        adaptive_step_size = 0.1 * (bounds.ub - bounds.lb) * (1 - np.sqrt(self.current_evals / self.budget))  # Modified line for step size
        best = np.copy(individual)
        best_fitness = func(best)
        self.current_evals += 1
        for _ in range(7):
            if self.current_evals >= self.budget:
                break
            candidate = best + adaptive_step_size * np.random.normal(0, 1, self.dim)
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
            adaptive_subpop_size = int(self.subpopulation_size * (1 + 0.15 * np.sin(self.current_evals / self.budget)))  # Modified line for adaptive subpopulation size
            for sp in range(self.num_subpopulations):
                start_idx = sp * adaptive_subpop_size
                end_idx = start_idx + adaptive_subpop_size
                subpopulation = total_population[start_idx:end_idx]
                subpopulation = self.differential_evolution(subpopulation, func, bounds)
                for i in range(adaptive_subpop_size):
                    if self.current_evals >= self.budget:
                        break
                    candidate = self.local_search(subpopulation[i], func, bounds)
                    candidate_fitness = func(candidate)
                    self.current_evals += 1
                    if candidate_fitness < best_fitness:
                        best_solution, best_fitness = candidate, candidate_fitness
                total_population[start_idx:end_idx] = subpopulation

        return best_solution