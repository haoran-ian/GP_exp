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
        self.crossover_rate = 0.9
        self.base_mutation_f = 0.8
        self.adaptive_factor = 0.05
        self.last_improvement = 0
        self.exploration_phase = True
        self.diversity_threshold = 0.01
        self.stagnation_limit = 0.1 * self.budget

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
                        self.last_improvement = self.current_evals

            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = a + self.base_mutation_f * (b - c)
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
                        self.last_improvement = self.current_evals

            # Adaptive adjustment of crossover rate and mutation factor
            fitness_std = np.std(fitness_values)
            if fitness_std < self.diversity_threshold and self.current_evals - self.last_improvement > self.stagnation_limit:
                self.crossover_rate = 0.9
                self.base_mutation_f = 0.8 + np.random.rand() * 0.2
                # Restart mechanism by reinitializing half of the population
                indices_to_restart = np.random.choice(self.population_size, self.population_size // 2, replace=False)
                population[indices_to_restart] = np.random.uniform(
                    low=func.bounds.lb,
                    high=func.bounds.ub,
                    size=(len(indices_to_restart), self.dim)
                )
                self.last_improvement = self.current_evals
            else:
                self.crossover_rate = 0.9 - self.adaptive_factor * (self.current_evals / self.budget)
                self.base_mutation_f = 0.8 + self.adaptive_factor * ((self.current_evals - self.last_improvement) / self.budget)

            # Switch between exploration and exploitation phases
            if self.current_evals > self.budget * 0.5:
                self.exploration_phase = False

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