import numpy as np

class AdvancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.num_subpopulations = 3
        self.subpopulation_size = self.population_size // self.num_subpopulations
        self.f = 0.8
        self.cr = 0.9
        self.current_evals = 0
        self.memory = np.zeros((self.population_size, self.dim))  # Memory for adaptive behavior

    def differential_evolution(self, population, func, bounds):
        new_population = np.copy(population)
        for i in range(self.subpopulation_size):
            if self.current_evals >= self.budget:
                break
            a, b, c = np.random.choice(list(set(range(self.subpopulation_size)) - {i}), 3, replace=False)
            adaptive_f = self.f * (1 - (self.current_evals / self.budget))  # Adaptive scaling factor
            mutant = population[a] + adaptive_f * (population[b] - population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            dynamic_cr = self.cr + 0.1 * np.sin(2 * np.pi * self.current_evals / self.budget)  # Dynamic crossover rate
            cross_points = np.random.rand(self.dim) < dynamic_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = func(trial)
            self.current_evals += 1
            if trial_fitness < func(population[i]):
                new_population[i] = trial
                self.memory[i] = trial  # Store in memory for neighborhood-based local search
        return new_population

    def local_search(self, individual, func, bounds):
        neighborhood_radius = 0.05 * (1 - self.current_evals / self.budget) * (bounds.ub - bounds.lb)
        best = np.copy(individual)
        best_fitness = func(best)
        self.current_evals += 1
        for _ in range(10):  # Increased attempts with neighborhood search
            if self.current_evals >= self.budget:
                break
            candidates = best + neighborhood_radius * np.random.uniform(-1, 1, self.dim)
            candidates = np.clip(candidates, bounds.lb, bounds.ub)
            candidate_fitness = func(candidates)
            self.current_evals += 1
            if candidate_fitness < best_fitness:
                best, best_fitness = candidates, candidate_fitness
        return best

    def __call__(self, func):
        bounds = func.bounds
        total_population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.current_evals = 0
        best_solution = total_population[0]
        best_fitness = func(best_solution)
        self.current_evals += 1

        while self.current_evals < self.budget:
            adaptive_subpop_size = int(self.subpopulation_size * (1 + 0.2 * (self.current_evals / self.budget)))  # Enhanced adaptation
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