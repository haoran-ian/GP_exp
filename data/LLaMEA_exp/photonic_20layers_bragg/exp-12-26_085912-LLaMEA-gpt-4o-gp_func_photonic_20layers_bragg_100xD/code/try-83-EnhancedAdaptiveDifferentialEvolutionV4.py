import numpy as np

class EnhancedAdaptiveDifferentialEvolutionV4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.best_global_position = None
        self.best_global_value = float('inf')
        self.current_evals = 0
        self.base_crossover_rate = 0.7
        self.base_mutation_f = 0.5
        self.learning_rate = 0.1
        self.historical_crossover = []
        self.historical_mutation_f = []

    def __call__(self, func):
        population = np.random.uniform(
            low=func.bounds.lb,
            high=func.bounds.ub,
            size=(self.population_size, self.dim)
        )
        fitness_values = np.full(self.population_size, float('inf'))

        while self.current_evals < self.budget:
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                fitness_value = func(population[i])
                self.current_evals += 1

                if fitness_value < fitness_values[i]:
                    fitness_values[i] = fitness_value
                    if fitness_value < self.best_global_value:
                        self.best_global_value = fitness_value
                        self.best_global_position = population[i]

            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = a + self.base_mutation_f * (b - c)
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.base_crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                trial_fitness = func(trial_vector)
                self.current_evals += 1

                if trial_fitness < fitness_values[i]:
                    fitness_values[i] = trial_fitness
                    population[i] = trial_vector
                    if trial_fitness < self.best_global_value:
                        self.best_global_value = trial_fitness
                        self.best_global_position = trial_vector

            # Update historical performance
            self.historical_crossover.append(self.base_crossover_rate)
            self.historical_mutation_f.append(self.base_mutation_f)

            # Adaptive adjustment based on historical performance
            if len(self.historical_crossover) > 5:
                performance_improvement = np.diff(self.historical_crossover[-5:])
                if np.mean(performance_improvement) > 0:
                    self.base_crossover_rate += self.learning_rate * np.mean(performance_improvement)
                else:
                    self.base_crossover_rate -= self.learning_rate * np.mean(performance_improvement)

                performance_improvement = np.diff(self.historical_mutation_f[-5:])
                if np.mean(performance_improvement) > 0:
                    self.base_mutation_f += self.learning_rate * np.mean(performance_improvement)
                else:
                    self.base_mutation_f -= self.learning_rate * np.mean(performance_improvement)

            # Dynamic population resizing
            new_population_size = int(self.initial_population_size * (1 - self.current_evals / self.budget))
            if new_population_size < 5:
                new_population_size = 5
            if new_population_size < self.population_size:
                indices_to_keep = np.argsort(fitness_values)[:new_population_size]
                population = population[indices_to_keep]
                fitness_values = fitness_values[indices_to_keep]
                self.population_size = new_population_size

        return self.best_global_position