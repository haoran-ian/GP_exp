import numpy as np

class EnhancedHybridDELocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.base_mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.mutation_decay = 0.99

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        mutation_factors = np.full(self.population_size, self.base_mutation_factor)

        while evals < self.budget:
            trial_population = np.empty_like(population)
            fitness_variance = np.var(fitness)
            max_fitness_diff = np.max(fitness) - np.min(fitness)
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                weight = (fitness[indices[0]] - fitness[i]) / (1e-9 + max_fitness_diff)
                mutation_factor = mutation_factors[i] * self.mutation_decay
                mutant = np.clip(a + weight * mutation_factor * (b - c), lb, ub)

                self.crossover_probability = 0.9 - (fitness_variance / (1e-9 + np.max(fitness_variance))) * 0.5
                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    trial_population[i] = trial
                    fitness[i] = trial_fitness
                    mutation_factors[i] = min(1.0, mutation_factors[i] + self.adaptation_rate * 1.1)
                else:
                    trial_population[i] = population[i]
                    mutation_factors[i] = max(0.1, mutation_factors[i] * 0.9)

                if evals < self.budget:
                    perturbation_scale = self.local_search_perturbation + 0.01 * fitness_variance
                    local_trial = trial + perturbation_scale * np.random.normal(size=self.dim)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[i]:
                        trial_population[i] = local_trial
                        fitness[i] = local_fitness

            population[:] = trial_population

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]