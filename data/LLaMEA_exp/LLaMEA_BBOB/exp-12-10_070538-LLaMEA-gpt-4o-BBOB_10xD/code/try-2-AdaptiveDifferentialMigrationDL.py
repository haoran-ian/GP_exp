import numpy as np

class AdaptiveDifferentialMigrationDL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 + 2 * dim
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best = None
        self.learning_rate = 0.02  # Modified learning rate for parameter adaptation

    def __call__(self, func):
        eval_count = 0
        fitness = np.apply_along_axis(func, 1, self.population)
        eval_count += self.population_size
        self.best = self.population[np.argmin(fitness)]

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Dynamic adjustment of F and CR
                self.F = self.learning_rate * np.random.rand() + 0.5
                self.CR = self.learning_rate * np.random.rand() + 0.9  # Slight adjustment here

                # Select three random indices different from i
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Differential mutation
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness

                # Update the best solution
                if trial_fitness < func(self.best):
                    self.best = trial

            # Learning phase to exploit best solutions
            if eval_count < self.budget:
                # Generate a new population around the current best
                new_population_size = self.population_size + 1  # Dynamic population size
                new_population = self.best + np.random.normal(0, 0.1, (new_population_size, self.dim))
                new_population = np.clip(new_population, self.lb, self.ub)
                new_fitness = np.apply_along_axis(func, 1, new_population)
                eval_count += new_population_size

                # Combine old and new population
                combined_population = np.vstack((self.population, new_population))
                combined_fitness = np.hstack((fitness, new_fitness))

                # Select the best individuals for the next generation
                best_indices = np.argsort(combined_fitness)[:self.population_size]
                self.population = combined_population[best_indices]
                fitness = combined_fitness[best_indices]
                self.best = self.population[np.argmin(fitness)]

        return self.best