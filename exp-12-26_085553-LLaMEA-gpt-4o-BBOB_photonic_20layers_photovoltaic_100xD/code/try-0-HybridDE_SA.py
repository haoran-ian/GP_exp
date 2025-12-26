import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * self.dim
        self.population = None
        self.fitness = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx):
        candidates = list(range(self.pop_size))
        candidates.remove(target_idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + 0.8 * (self.population[b] - self.population[c])
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < 0.9
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target_idx, trial, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness

    def simulated_annealing(self, current_best, current_best_fitness):
        for _ in range(5):  # Inner loop iterations for SA
            candidate = current_best + np.random.normal(0, 0.1, self.dim)
            candidate = np.clip(candidate, self.lb, self.ub)
            candidate_fitness = self.func(candidate)
            if candidate_fitness < current_best_fitness or \
               np.exp((current_best_fitness - candidate_fitness) / self.temperature) > np.random.rand():
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
            evaluations += self.pop_size

            for i in range(self.pop_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness)

            current_best_idx = np.argmin(self.fitness)
            current_best, current_best_fitness = self.population[current_best_idx], self.fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_best_fitness

            best_solution, best_fitness = self.simulated_annealing(best_solution, best_fitness)

            self.temperature *= 0.95  # Cooling schedule for SA

            if evaluations >= self.budget:
                break

        return best_solution, best_fitness