import numpy as np

class AdvancedAdaptiveDifferentialMigrationLE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 + 2 * dim
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best = None
        self.learning_rate = 0.01
        self.eval_count = 0

    def _dynamic_parameters(self, fitness):
        fitness_variance = np.var(fitness)
        F = self.learning_rate * np.random.rand() + 0.5 * (1 + 0.1 * fitness_variance)
        CR = self.learning_rate * np.random.rand() + 0.8
        return F, CR

    def _mutate(self, indices, F):
        a, b, c = self.population[indices]
        mutant = a + F * (b - c) + 0.1 * (self.best - a)
        mutant = np.clip(mutant, self.lb, self.ub)
        return mutant

    def _crossover(self, target, mutant, CR):
        crossover = np.random.rand(self.dim) < CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def _selection(self, fitness, trial, trial_fitness, i):
        if trial_fitness < fitness[i]:
            self.population[i] = trial
            fitness[i] = trial_fitness
            if trial_fitness < fitness[np.argmin(fitness)]:
                self.best = trial

    def _evaluate_population(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.eval_count += self.population_size
        self.best = self.population[np.argmin(fitness)]
        return fitness

    def __call__(self, func):
        fitness = self._evaluate_population(func)

        while self.eval_count < self.budget:
            self.learning_rate = 0.05 * (1 - self.eval_count / self.budget) + 0.01
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                F, CR = self._dynamic_parameters(fitness)
                indices = [idx for idx in range(self.population_size) if idx != i]
                mutant = self._mutate(np.random.choice(indices, 3, replace=False), F)
                trial = self._crossover(self.population[i], mutant, CR)

                trial_fitness = func(trial)
                self.eval_count += 1
                self._selection(fitness, trial, trial_fitness, i)

            if self.eval_count < self.budget:
                self._dynamic_population_replacement(func, fitness)

            if self.eval_count % (self.budget // 10) == 0:  # Periodic reinitialization condition
                reinit_count = self.population_size // 5
                self.population[:reinit_count] = np.random.uniform(self.lb, self.ub, (reinit_count, self.dim))

        return self.best

    def _dynamic_population_replacement(self, func, fitness):
        dynamic_sigma = 0.1 + 0.5 * (1 - self.eval_count / self.budget)
        new_population = self.best + np.random.normal(0, dynamic_sigma, (self.population_size, self.dim))
        dynamic_perturbation = 0.7 * (1 - self.eval_count / self.budget)
        perturbed_population = new_population + dynamic_perturbation * np.random.uniform(-1, 1, new_population.shape)
        perturbed_population = np.clip(perturbed_population, self.lb, self.ub)
        new_fitness = np.apply_along_axis(func, 1, perturbed_population)
        self.eval_count += self.population_size

        combined_population = np.vstack((self.population, perturbed_population))
        combined_fitness = np.hstack((fitness, new_fitness))
        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.population = combined_population[best_indices]
        fitness[:] = combined_fitness[best_indices]
        self.best = self.population[np.argmin(fitness)]