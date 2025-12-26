import numpy as np

class EnhancedHybridDELocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.elitism_rate = 0.1
        self.dynamic_population_shrink = 0.95

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = population_size

        while evals < self.budget:
            trial_population = np.empty_like(population)
            fitness_variance = np.var(fitness)
            num_elites = max(1, int(self.elitism_rate * population_size))
            elites = population[np.argsort(fitness)[:num_elites]]

            for i in range(num_elites, population_size):
                if evals >= self.budget:
                    break

                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    trial_population[i] = trial
                    fitness[i] = trial_fitness
                    self.mutation_factor = min(1.0, self.mutation_factor + self.adaptation_rate * 1.1)
                else:
                    trial_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.adaptation_rate)

                if evals < self.budget:
                    local_trial = trial + (self.local_search_perturbation + 0.01 * fitness_variance) * np.random.normal(size=self.dim)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[i]:
                        trial_population[i] = local_trial
                        fitness[i] = local_fitness

            trial_population[:num_elites] = elites
            fitness[:num_elites] = [func(ind) for ind in elites]
            population = trial_population
            population_size = int(population_size * self.dynamic_population_shrink)
            population = population[:population_size]
            fitness = fitness[:population_size]

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]