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

    def opposition_based_learning(self, population, bounds):
        opp_population = bounds.lb + bounds.ub - population
        return np.clip(opp_population, bounds.lb, bounds.ub)

    def differential_evolution(self, population, func, bounds, opp_population):
        new_population = np.copy(population)
        for i in range(self.subpopulation_size):
            if self.current_evals >= self.budget:
                break
            indices = list(range(self.subpopulation_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + self.f * (population[b] - population[c])
            mutant = np.clip(mutant, bounds.lb, bounds.ub)
            dynamic_cr = self.cr * (1 - self.current_evals / self.budget)
            cross_points = np.random.rand(self.dim) < dynamic_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = func(trial)
            self.current_evals += 1
            if trial_fitness < func(population[i]):
                new_population[i] = trial
            elif func(opp_population[i]) < func(new_population[i]):
                new_population[i] = opp_population[i]
        return new_population

    def local_search(self, individual, func, bounds):
        adaptive_step_size = 0.1 * (bounds.ub - bounds.lb) * (1 - self.current_evals / self.budget)
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
        opp_population = self.opposition_based_learning(total_population, bounds)
        self.current_evals = 0
        best_solution = total_population[0]
        best_fitness = func(best_solution)
        self.current_evals += 1

        while self.current_evals < self.budget:
            for sp in range(self.num_subpopulations):
                start_idx = sp * self.subpopulation_size
                end_idx = start_idx + self.subpopulation_size
                subpopulation = total_population[start_idx:end_idx]
                opp_subpopulation = opp_population[start_idx:end_idx]
                subpopulation = self.differential_evolution(subpopulation, func, bounds, opp_subpopulation)
                for i in range(self.subpopulation_size):
                    if self.current_evals >= self.budget:
                        break
                    candidate = self.local_search(subpopulation[i], func, bounds)
                    candidate_fitness = func(candidate)
                    self.current_evals += 1
                    if candidate_fitness < best_fitness:
                        best_solution, best_fitness = candidate, candidate_fitness
                total_population[start_idx:end_idx] = subpopulation
                opp_population[start_idx:end_idx] = self.opposition_based_learning(subpopulation, bounds)

        return best_solution