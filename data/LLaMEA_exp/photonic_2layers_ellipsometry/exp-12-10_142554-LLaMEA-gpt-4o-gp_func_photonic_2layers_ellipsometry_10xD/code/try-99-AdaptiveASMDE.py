import numpy as np

class AdaptiveASMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(4, 10 * dim // 2)
        self.base_mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.success_mutation_factors = []
        self.learning_rate_scaler = 1.0
        self.population_size = self.initial_population_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                # Adaptive mutation factor with learning rate scaling
                F = self.base_mutation_factor * self.learning_rate_scaler
                if self.success_mutation_factors:
                    avg_success = np.mean(self.success_mutation_factors)
                    F = avg_success * np.random.uniform(0.9, 1.1)
                mutant = np.clip(a + F * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                    self.success_mutation_factors.append(F)
                    # Scale learning rate based on improvement
                    self.learning_rate_scaler *= 1.05
                else:
                    new_population.append(pop[i])
                    new_fitness.append(fitness[i])
                    # Decrease learning rate scaler if no improvement
                    self.learning_rate_scaler *= 0.95

                if eval_count >= self.budget:
                    break

            pop = np.array(new_population)
            fitness = np.array(new_fitness)

            # Adjust population size based on diversity
            diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
            if diversity > 0.1:
                self.population_size = min(self.initial_population_size, int(self.population_size * 1.1))
            else:
                self.population_size = max(4, int(self.population_size * 0.9))

            # Re-initialize the population if necessary
            if len(pop) < self.population_size:
                additional_pop = np.random.uniform(lb, ub, (self.population_size - len(pop), self.dim))
                pop = np.vstack((pop, additional_pop))
                additional_fitness = np.array([func(ind) for ind in additional_pop])
                fitness = np.concatenate((fitness, additional_fitness))
                eval_count += len(additional_pop)

            # Dynamic adjustment of mutation factor and crossover rate
            self.base_mutation_factor = 0.3 + 0.7 * diversity
            self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]