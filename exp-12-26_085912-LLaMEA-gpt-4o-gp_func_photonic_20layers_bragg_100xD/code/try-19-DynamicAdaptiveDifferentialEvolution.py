import numpy as np

class DynamicAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.best_global_position = None
        self.best_global_value = float('inf')
        self.current_evals = 0
        self.crossover_rate = 0.9
        self.mutation_f = 0.8
        self.adaptive_factor = 0.05

    def __call__(self, func):
        dynamic_population_size = self.initial_population_size
        population = np.random.uniform(
            low=func.bounds.lb,
            high=func.bounds.ub,
            size=(dynamic_population_size, self.dim)
        )
        fitness_values = np.full(dynamic_population_size, float('inf'))

        while self.current_evals < self.budget:
            for i in range(dynamic_population_size):
                if self.current_evals >= self.budget:
                    break
                fitness_value = func(population[i])
                self.current_evals += 1

                if fitness_value < fitness_values[i]:
                    fitness_values[i] = fitness_value

                if fitness_value < self.best_global_value:
                    self.best_global_value = fitness_value
                    self.best_global_position = population[i]

            for i in range(dynamic_population_size):
                if self.current_evals >= self.budget:
                    break
                
                idxs = [idx for idx in range(dynamic_population_size) if idx != i]
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
            
            # Adaptive adjustment of crossover rate and mutation factor
            self.crossover_rate += self.adaptive_factor * (1 - (self.current_evals / self.budget))
            self.mutation_f += self.adaptive_factor * (1 - (self.current_evals / self.budget))

            # Dynamic population size adjustment
            if (self.current_evals / self.budget) > 0.5 and dynamic_population_size > 10:
                dynamic_population_size = int(dynamic_population_size * 0.9)
                population = population[:dynamic_population_size]
                fitness_values = fitness_values[:dynamic_population_size]
        
        return self.best_global_position