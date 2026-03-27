import numpy as np

class ChaosPerturbedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.mutation_rate = 0.8
        self.crossover_rate = 0.9
        self.chaos_param = 0.75

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx, scale_factor):
        candidates = list(range(self.pop_size))
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

    def chaos_perturbation(self):
        self.chaos_param *= 3.57 * (1 - self.chaos_param)
        return self.chaos_param

    def diversity_control(self):
        diversity = np.std(self.population, axis=0).mean()
        scale_factor = 0.5 + (1.5 * (1 - diversity))
        crossover_prob = 0.8 + (0.2 * diversity)
        return scale_factor, crossover_prob

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
                scale_factor, crossover_prob = self.diversity_control()
                mutant = self.mutate(i, scale_factor)
                chaotic_factor = self.chaos_perturbation()
                trial = self.crossover(self.population[i], mutant * chaotic_factor, crossover_prob)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness)

            current_best_idx = np.argmin(self.fitness)
            current_best, current_best_fitness = self.population[current_best_idx], self.fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_best_fitness

            best_solution, best_fitness = self.adaptive_simulated_annealing(best_solution, best_fitness)

            self.temperature *= 0.92

            if evaluations >= self.budget:
                break

        return best_solution, best_fitness