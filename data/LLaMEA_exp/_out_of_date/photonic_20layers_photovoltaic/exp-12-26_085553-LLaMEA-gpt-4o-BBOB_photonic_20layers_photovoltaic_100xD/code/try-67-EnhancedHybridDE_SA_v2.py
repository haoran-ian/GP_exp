import numpy as np

class EnhancedHybridDE_SA_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_subpopulations = 3
        self.subpop_size = 5 * self.dim
        self.total_pop_size = self.num_subpopulations * self.subpop_size
        self.mutation_base = 0.5
        self.crossover_base = 0.9

    def initialize_population(self, lb, ub):
        self.populations = [np.random.uniform(lb, ub, (self.subpop_size, self.dim)) for _ in range(self.num_subpopulations)]
        self.fitness = [np.full(self.subpop_size, np.inf) for _ in range(self.num_subpopulations)]

    def evaluate_population(self, func):
        for sp in range(self.num_subpopulations):
            for i in range(self.subpop_size):
                if np.isinf(self.fitness[sp][i]):
                    self.fitness[sp][i] = func(self.populations[sp][i])

    def mutate(self, sp, target_idx, scale_factor, adaptive_factor):
        candidates = list(range(self.subpop_size))
        candidates.remove(target_idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = (self.populations[sp][a] + scale_factor * (self.populations[sp][b] - self.populations[sp][c]) 
                  + adaptive_factor * (self.populations[sp][a] - self.populations[sp][target_idx]))
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant, crossover_prob):
        cross_points = np.random.rand(self.dim) < crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def select(self, sp, target_idx, trial, trial_fitness):
        if trial_fitness < self.fitness[sp][target_idx]:
            self.populations[sp][target_idx] = trial
            self.fitness[sp][target_idx] = trial_fitness

    def adaptive_simulated_annealing(self, current_best, current_best_fitness):
        for _ in range(5):
            candidate = current_best + np.random.normal(0, 0.1 * self.temperature, self.dim)
            candidate = np.clip(candidate, self.lb, self.ub)
            candidate_fitness = self.func(candidate)
            delta_fitness = current_best_fitness - candidate_fitness
            if delta_fitness > 0 or np.exp(delta_fitness / self.temperature) > np.random.rand():
                current_best, current_best_fitness = candidate, candidate_fitness
        return current_best, current_best_fitness

    def __call__(self, func):
        self.func = func
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub)

        evaluations = 0
        best_solution = None
        best_fitness = np.inf

        self.temperature = 1.0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.total_pop_size

            for sp in range(self.num_subpopulations):
                for i in range(self.subpop_size):
                    scale_factor = self.mutation_base + 0.5 * np.cos(np.pi * evaluations / self.budget)
                    adaptive_factor = np.tanh(0.1 * (best_fitness - self.fitness[sp][i]))  
                    crossover_prob = self.crossover_base + np.random.uniform(-0.1, 0.1)
                    mutant = self.mutate(sp, i, scale_factor, adaptive_factor)
                    trial = self.crossover(self.populations[sp][i], mutant, crossover_prob)
                    trial_fitness = func(trial)
                    evaluations += 1
                    self.select(sp, i, trial, trial_fitness)

            current_best_idx, current_best_fitness = None, np.inf
            for sp in range(self.num_subpopulations):
                sp_best_idx = np.argmin(self.fitness[sp])
                sp_best, sp_best_fitness = self.populations[sp][sp_best_idx], self.fitness[sp][sp_best_idx]
                if sp_best_fitness < current_best_fitness:
                    current_best_idx, current_best_fitness = sp_best_idx, sp_best_fitness
                    current_best = sp_best

            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_best_fitness

            best_solution, best_fitness = self.adaptive_simulated_annealing(best_solution, best_fitness)

            self.temperature *= 0.92 + 0.08 * np.tanh(0.1 * evaluations / self.budget)
            dynamic_size_factor = 1 - evaluations / self.budget
            self.subpop_size = max(5, int(5 * self.dim * dynamic_size_factor))

            if evaluations >= self.budget:
                break

        return best_solution, best_fitness