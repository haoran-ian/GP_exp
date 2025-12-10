import numpy as np

class AdaptiveDifferentialEvolutionChaotic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 + 2 * dim
        self.population = self._chaotic_initialization()
        self.best = None
        self.learning_rate = 0.01
        self.eval_count = 0
        self.local_search_frequency = 5  # Perform local search every few generations

    def _chaotic_initialization(self):
        # Use a logistic map for chaotic initialization
        x = np.random.rand(self.population_size, self.dim)
        for _ in range(100):  # Iterate to get chaotic sequence
            x = 4 * x * (1 - x)
        return self.lb + (self.ub - self.lb) * x

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

        generation = 0
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

            # Periodic local search
            if generation % self.local_search_frequency == 0 and self.eval_count < self.budget:
                self._local_search(func, fitness)

            generation += 1

        return self.best

    def _local_search(self, func, fitness):
        # Perform a local search around the best known solution
        search_radius = 0.1
        new_solutions = self.best + np.random.uniform(-search_radius, search_radius, (self.population_size, self.dim))
        new_solutions = np.clip(new_solutions, self.lb, self.ub)
        new_fitness = np.apply_along_axis(func, 1, new_solutions)
        self.eval_count += self.population_size

        combined_population = np.vstack((self.population, new_solutions))
        combined_fitness = np.hstack((fitness, new_fitness))
        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.population = combined_population[best_indices]
        fitness[:] = combined_fitness[best_indices]
        self.best = self.population[np.argmin(fitness)]