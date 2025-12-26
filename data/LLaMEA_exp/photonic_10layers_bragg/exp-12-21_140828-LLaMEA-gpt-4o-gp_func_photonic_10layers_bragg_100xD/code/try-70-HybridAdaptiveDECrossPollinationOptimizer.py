import numpy as np

class HybridAdaptiveDECrossPollinationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.elitism_rate = 0.1  # Elitism to retain top performers
        self.feedback_rate = 0.05  # New parameter for dynamic feedback

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = population_size

        while evals < self.budget:
            trial_population = np.empty_like(population)
            fitness_variance = np.var(fitness)
            max_fitness_diff = np.max(fitness) - np.min(fitness)
            elitism_count = int(self.elitism_rate * population_size)

            for i in range(population_size):
                if evals >= self.budget:
                    break

                if i < elitism_count:
                    trial_population[i] = population[np.argsort(fitness)[:elitism_count][i]]
                    continue

                if np.random.rand() < 0.5:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                else:
                    indices = np.random.permutation(population_size)[:5]
                    a, b, c, d, e = population[indices]
                    a = a + self.mutation_factor * (b - c) + self.mutation_factor * (d - e)

                weight = (fitness[indices[0]] - fitness[i]) / (1e-9 + max_fitness_diff)
                mutant = np.clip(a + weight * self.mutation_factor * (b - c), lb, ub)

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
                    self.mutation_factor = min(1.0, self.mutation_factor + self.adaptation_rate * 1.1)
                else:
                    trial_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.adaptation_rate)

                if evals < self.budget:
                    perturbation = self.local_search_perturbation + 0.01 * (1 - fitness_variance)
                    local_trial = trial + perturbation * np.random.normal(size=self.dim)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[i]:
                        trial_population[i] = local_trial
                        fitness[i] = local_fitness

                if np.random.rand() < self.feedback_rate:
                    peer = population[np.random.randint(0, population_size)]
                    cross_pollen = trial + self.feedback_rate * (peer - trial)
                    cross_pollen = np.clip(cross_pollen, lb, ub)
                    cross_fitness = func(cross_pollen)
                    evals += 1
                    if cross_fitness < fitness[i]:
                        trial_population[i] = cross_pollen
                        fitness[i] = cross_fitness

            population[:] = trial_population

            if evals < self.budget:
                population_size = max(10, int(self.initial_population_size * (1 - evals / self.budget)))
                population = population[:population_size]
                fitness = fitness[:population_size]
                trial_population = trial_population[:population_size]

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]