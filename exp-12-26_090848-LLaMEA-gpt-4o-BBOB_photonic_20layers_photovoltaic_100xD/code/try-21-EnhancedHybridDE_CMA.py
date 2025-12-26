import numpy as np

class EnhancedHybridDE_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        min_pop_size = 4 + int(3 * np.log(self.dim))
        max_pop_size = 10 * min_pop_size
        population_size = min_pop_size
        de_scale = 0.5
        crossover_rate = 0.9
        cma_sigma = 0.5
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        previous_best_fitness = np.min(fitness)

        while evaluations < self.budget:
            adap_state = (self.budget - evaluations) / self.budget
            de_scale = 0.1 + 0.4 * adap_state
            crossover_rate = 0.6 + 0.3 * (1 - adap_state)
            cma_sigma = 0.3 + 0.2 * adap_state

            num_immigrants = max(1, int(population_size * 0.1))
            immigrants = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_immigrants, self.dim))
            pop_with_immigrants = np.vstack((population, immigrants))
            fitness_with_immigrants = np.concatenate([fitness, np.array([func(ind) for ind in immigrants])])
            evaluations += num_immigrants

            for i in range(population_size):
                indices = np.random.choice(len(pop_with_immigrants), 3, replace=False)
                a, b, c = pop_with_immigrants[indices]
                mutant = np.clip(a + de_scale * (b - c), bounds[:, 0], bounds[:, 1])
                crossover = np.random.rand(self.dim) < crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            current_best_fitness = np.min(fitness)
            fitness_variance = np.var(fitness)
            fitness_improvement = previous_best_fitness - current_best_fitness
            previous_best_fitness = current_best_fitness

            if fitness_variance < 1e-5 and fitness_improvement < 1e-6:
                best_indices = np.argsort(fitness)[:population_size//2]
                cma_population = population[best_indices]
                cma_mean = np.mean(cma_population, axis=0)
                cma_cov = np.cov(cma_population.T)
                cma_samples = np.random.multivariate_normal(cma_mean, cma_sigma**2 * cma_cov, population_size//2)

                for sample in cma_samples:
                    sample = np.clip(sample, bounds[:, 0], bounds[:, 1])
                    sample_fitness = func(sample)
                    evaluations += 1

                    if sample_fitness < np.max(fitness):
                        worst_index = np.argmax(fitness)
                        population[worst_index] = sample
                        fitness[worst_index] = sample_fitness

                    if evaluations >= self.budget:
                        break

            if fitness_improvement > 1e-3:
                population_size = min(max_pop_size, int(population_size * 1.2))
                new_population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size - len(population), self.dim))
                population = np.vstack((population, new_population))
                fitness = np.concatenate((fitness, np.array([func(ind) for ind in new_population])))
                evaluations += len(new_population)

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]