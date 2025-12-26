import numpy as np

class RefinedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.mutation_rate = 0.8
        self.crossover_rate = 0.9

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.initial_pop_size, self.dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)

    def evaluate_population(self, func):
        for i in range(len(self.population)):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx, scale_factor, adaptive_factor):
        candidates = list(range(len(self.population)))
        candidates.remove(target_idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + scale_factor * (self.population[b] - self.population[c]) + adaptive_factor * (self.population[a] - self.population[target_idx])
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant, crossover_prob):
        cross_points = np.random.rand(self.dim) < crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target_idx, trial, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness

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
        pop_size_decay = np.linspace(self.initial_pop_size, 5, self.budget)

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += len(self.population)

            for i in range(len(self.population)):
                scale_factor = 0.5 + 0.5 * np.cos(np.pi * evaluations / self.budget)
                adaptive_factor = np.tanh(0.1 * (best_fitness - self.fitness[i]))
                crossover_prob = 0.7 + 0.25 * (1 - evaluations / self.budget)
                mutant = self.mutate(i, scale_factor, adaptive_factor)
                trial = self.crossover(self.population[i], mutant, crossover_prob)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness)

            current_best_idx = np.argmin(self.fitness)
            current_best, current_best_fitness = self.population[current_best_idx], self.fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_best_fitness

            best_solution, best_fitness = self.adaptive_simulated_annealing(best_solution, best_fitness)

            self.temperature *= 0.92 * (1 - evaluations / self.budget) + 0.08 * np.tanh(0.1 * evaluations / self.budget)
            current_pop_size = int(pop_size_decay[evaluations])
            if len(self.population) > current_pop_size:
                to_reduce = len(self.population) - current_pop_size
                worst_indices = np.argsort(self.fitness)[-to_reduce:]
                self.population = np.delete(self.population, worst_indices, axis=0)
                self.fitness = np.delete(self.fitness, worst_indices)

            if evaluations >= self.budget:
                break

        return best_solution, best_fitness