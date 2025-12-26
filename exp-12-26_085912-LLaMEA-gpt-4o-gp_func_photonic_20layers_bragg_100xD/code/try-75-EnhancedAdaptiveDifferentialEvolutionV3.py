import numpy as np

class EnhancedAdaptiveDifferentialEvolutionV3:
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
        self.last_improvement = 0
        self.success_params = []
        self.failure_params = []

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
                        self.success_params.append((self.crossover_rate, self.mutation_f))
                    else:
                        self.failure_params.append((self.crossover_rate, self.mutation_f))

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
                        self.last_improvement = self.current_evals
                        self.success_params.append((self.crossover_rate, self.mutation_f))
                    else:
                        self.failure_params.append((self.crossover_rate, self.mutation_f))
            
            # Self-adaptive learning strategy for control parameters
            if self.success_params:
                avg_success_rate = np.mean([cr for cr, _ in self.success_params])
                avg_success_mutation = np.mean([mf for _, mf in self.success_params])
                self.crossover_rate = 0.7 * self.crossover_rate + 0.3 * avg_success_rate
                self.mutation_f = 0.7 * self.mutation_f + 0.3 * avg_success_mutation
            
            if self.failure_params:
                avg_failure_rate = np.mean([cr for cr, _ in self.failure_params])
                avg_failure_mutation = np.mean([mf for _, mf in self.failure_params])
                self.crossover_rate = 0.9 * self.crossover_rate + 0.1 * avg_failure_rate
                self.mutation_f = 0.9 * self.mutation_f + 0.1 * avg_failure_mutation
            
            self.success_params.clear()
            self.failure_params.clear()
            
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