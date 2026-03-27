import numpy as np

class EnhancedAdaptiveMultiPhaseMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.num_subpopulations = 3
        self.subpopulation_size = self.population_size // self.num_subpopulations
        self.f = 0.8
        self.cr = 0.9
        self.current_evals = 0
        self.restart_threshold = 0.2  # Restart if no improvement in 20% of budget

    def differential_evolution(self, population, func, bounds):
        new_population = np.copy(population)
        for i in range(self.subpopulation_size):
            if self.current_evals >= self.budget:
                break
            indices = list(range(self.subpopulation_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + self.f * (population[b] - population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            dynamic_cr = self.cr * np.exp(-self.current_evals / self.budget)
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
        adaptive_step_size = 0.1 * (bounds.ub - bounds.lb) * (1 - self.current_evals / self.budget)
        best = np.copy(individual)
        best_fitness = func(best)
        self.current_evals += 1
        for _ in range(15):
            if self.current_evals >= self.budget:
                break
            candidate = best + adaptive_step_size * np.random.normal(0, 1, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            self.current_evals += 1
            if candidate_fitness < best_fitness:
                best, best_fitness = candidate, candidate_fitness
        return best

    def adaptive_local_search(self, population, func, bounds):
        for i in range(len(population)):
            if self.current_evals >= self.budget:
                break
            population[i] = self.local_search(population[i], func, bounds)
        return population

    def gaussian_mutation(self, population, bounds):
        sigma = 0.1 * (bounds.ub - bounds.lb) * (1 - self.current_evals / self.budget)  # Dynamic sigma scaling
        for i in range(len(population)):
            if self.current_evals >= self.budget:
                break
            mutant = population[i] + np.random.normal(0, sigma, self.dim)
            population[i] = np.clip(mutant, bounds.lb, bounds.ub)
        return population

    def __call__(self, func):
        bounds = func.bounds
        total_population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.current_evals = 0
        best_solution = total_population[0]
        best_fitness = func(best_solution)
        self.current_evals += 1
        last_best_fitness = best_fitness

        while self.current_evals < self.budget:
            adaptive_subpop_size = int(self.subpopulation_size * (1 + 0.1 * (self.current_evals / self.budget)))
            for sp in range(self.num_subpopulations):
                start_idx = sp * adaptive_subpop_size
                end_idx = start_idx + adaptive_subpop_size
                subpopulation = total_population[start_idx:end_idx]
                subpopulation = self.differential_evolution(subpopulation, func, bounds)
                subpopulation = self.adaptive_local_search(subpopulation, func, bounds)
                subpopulation = self.gaussian_mutation(subpopulation, bounds)
                for i in range(adaptive_subpop_size):
                    if self.current_evals >= self.budget:
                        break
                    candidate = subpopulation[i]
                    candidate_fitness = func(candidate)
                    self.current_evals += 1
                    if candidate_fitness < best_fitness:
                        best_solution, best_fitness = candidate, candidate_fitness
                total_population[start_idx:end_idx] = subpopulation

            # Restart mechanism
            if (self.current_evals / self.budget) > self.restart_threshold and abs(best_fitness - last_best_fitness) < 1e-8:
                total_population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
                last_best_fitness = best_fitness
                self.current_evals += self.population_size  # Account for restart evaluations

        return best_solution