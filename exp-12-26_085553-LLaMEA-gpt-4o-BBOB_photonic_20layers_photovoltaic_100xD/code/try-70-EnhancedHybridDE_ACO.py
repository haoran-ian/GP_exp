import numpy as np

class EnhancedHybridDE_ACO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.mutation_rate = 0.8
        self.crossover_rate = 0.9
        self.pheromone = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.pheromone = np.ones(self.dim)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx, scale_factor, adaptive_factor):
        candidates = list(range(self.pop_size))
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

    def ant_colony_update(self, current_best):
        pheromone_factor = np.exp(-self.fitness / np.min(self.fitness))
        self.pheromone = 0.9 * self.pheromone + 0.1 * pheromone_factor

        for i in range(self.pop_size):
            if np.random.rand() < self.pheromone[i % self.dim]:
                candidate = current_best + np.random.normal(0, 0.1 * self.pheromone[i % self.dim], self.dim)
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = self.func(candidate)
                if candidate_fitness < self.fitness[i]:
                    self.population[i] = candidate
                    self.fitness[i] = candidate_fitness

    def __call__(self, func):
        self.func = func
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub)

        evaluations = 0
        best_solution = None
        best_fitness = np.inf

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.pop_size

            for i in range(self.pop_size):
                scale_factor = 0.5 + 0.5 * np.cos(np.pi * evaluations / self.budget)
                adaptive_factor = np.tanh(0.1 * (best_fitness - self.fitness[i]))
                crossover_prob = np.random.uniform(0.8, 1.0)
                mutant = self.mutate(i, scale_factor, adaptive_factor)
                trial = self.crossover(self.population[i], mutant, crossover_prob)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness)

            current_best_idx = np.argmin(self.fitness)
            current_best, current_best_fitness = self.population[current_best_idx], self.fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_best_fitness

            self.ant_colony_update(best_solution)

            if evaluations >= self.budget:
                break

        return best_solution, best_fitness