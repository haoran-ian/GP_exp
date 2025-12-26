import numpy as np
import math

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.best_global_position = None
        self.best_global_value = float('inf')
        self.current_evals = 0
        self.crossover_rate = 0.9
        self.mutation_f = 0.8
        self.adaptive_factor = 0.05
        self.elitism_rate = 0.1
        self.logistic_map_r = 4.0
        self.logistic_map_x = 0.5

    def chaotic_map(self):
        self.logistic_map_x = self.logistic_map_r * self.logistic_map_x * (1 - self.logistic_map_x)
        return self.logistic_map_x

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
                mutant_vector = a + self.mutation_f * (b - c)
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                trial_fitness = func(trial_vector)
                self.current_evals += 1

                if trial_fitness < fitness_values[i]:
                    fitness_values[i] = trial_fitness
                    population[i] = trial_vector
                    if trial_fitness < self.best_global_value:
                        self.best_global_value = trial_fitness
                        self.best_global_position = trial_vector

            # Adaptive adjustment using chaotic maps
            self.crossover_rate = 0.7 + 0.2 * self.chaotic_map()
            self.mutation_f = 0.5 + 0.3 * self.chaotic_map()

            # Dynamic population resizing
            new_population_size = int(self.initial_population_size * (1 - self.current_evals / self.budget))
            if new_population_size < 5:
                new_population_size = 5
            if new_population_size < self.population_size:
                indices_to_keep = np.argsort(fitness_values)[:new_population_size]
                population = population[indices_to_keep]
                fitness_values = fitness_values[indices_to_keep]
                self.population_size = new_population_size

            # Elitism mechanism
            num_elites = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness_values)[:num_elites]
            elites = population[elite_indices]

            # Replace some non-elite individuals with elites
            non_elite_indices = np.setdiff1d(range(self.population_size), elite_indices)
            np.random.shuffle(non_elite_indices)
            replace_indices = non_elite_indices[:num_elites]
            population[replace_indices] = elites

        return self.best_global_position