import numpy as np

class RefinedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.max_pop_size = 20 * self.dim
        self.min_pop_size = 5 * self.dim
        self.mutation_rate = 0.8
        self.crossover_rate = 0.9

    def initialize_population(self, lb, ub, pop_size):
        self.population = np.random.uniform(lb, ub, (pop_size, self.dim))
        self.fitness = np.full(pop_size, np.inf)

    def evaluate_population(self, func):
        for i in range(len(self.population)):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx, scale_factor):
        candidates = list(range(len(self.population)))
        candidates.remove(target_idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + scale_factor * (self.population[b] - self.population[c])
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

    def adapt_population_size(self):
        if self.fitness.std() < 1e-3:
            new_size = max(self.min_pop_size, int(len(self.population) * 0.9))
        else:
            new_size = min(self.max_pop_size, int(len(self.population) * 1.1))
        if new_size != len(self.population):
            selected_indices = np.argsort(self.fitness)[:new_size]
            self.population = self.population[selected_indices]
            self.fitness = self.fitness[selected_indices]

    def __call__(self, func):
        self.func = func
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub, self.initial_pop_size)

        evaluations = 0
        best_solution = None
        best_fitness = np.inf

        self.temperature = 1.0

        while evaluations < self.budget:

            self.evaluate_population(func)
            evaluations += len(self.population)

            for i in range(len(self.population)):
                scale_factor = np.random.uniform(0.5, 1.0)
                crossover_prob = np.random.uniform(0.8, 1.0)
                mutant = self.mutate(i, scale_factor)
                trial = self.crossover(self.population[i], mutant, crossover_prob)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness)

            current_best_idx = np.argmin(self.fitness)
            current_best, current_best_fitness = self.population[current_best_idx], self.fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_best_fitness

            best_solution, best_fitness = self.adaptive_simulated_annealing(best_solution, best_fitness)

            self.temperature *= 0.92

            self.adapt_population_size()

            if evaluations >= self.budget:
                break

        return best_solution, best_fitness