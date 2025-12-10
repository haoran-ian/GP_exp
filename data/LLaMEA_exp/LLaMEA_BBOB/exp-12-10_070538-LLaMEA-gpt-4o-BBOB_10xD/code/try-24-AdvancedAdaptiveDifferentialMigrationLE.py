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
        self.learning_rate = 0.01  # Learning rate for parameter adaptation
        self.eval_count = 0

    def _dynamic_parameters(self):
        F = self.learning_rate * np.random.rand() + 0.5
        CR = self.learning_rate * np.random.rand() + 0.8
        return F, CR

    def _mutate(self, indices, F):
        a, b, c = self.population[indices]
        mutant = a + F * (b - c)
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
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                F, CR = self._dynamic_parameters()
                indices = [idx for idx in range(self.population_size) if idx != i]
                mutant = self._mutate(np.random.choice(indices, 3, replace=False), F)
                trial = self._crossover(self.population[i], mutant, CR)

                trial_fitness = func(trial)
                self.eval_count += 1
                self._selection(fitness, trial, trial_fitness, i)

            if self.eval_count < self.budget:
                self._learning_and_exploration(func, fitness)

        return self.best

    def _learning_and_exploration(self, func, fitness):
        exploration_sigma = 0.2  # Increased exploration
        new_population = self.best + np.random.normal(0, exploration_sigma, (self.population_size, self.dim))
        new_population = np.clip(new_population, self.lb, self.ub)
        new_fitness = np.apply_along_axis(func, 1, new_population)
        self.eval_count += self.population_size

        combined_population = np.vstack((self.population, new_population))
        combined_fitness = np.hstack((fitness, new_fitness))
        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.population = combined_population[best_indices]
        fitness = combined_fitness[best_indices]
        self.best = self.population[np.argmin(fitness)]