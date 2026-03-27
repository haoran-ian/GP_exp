import numpy as np

class AdvancedHybridDE_CMA_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_population_size = 4 + int(3 * np.log(self.dim))
        population_size = initial_population_size
        de_scale = 0.5
        crossover_rate = 0.9
        cma_population_size = population_size // 2
        cma_sigma = 0.5
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        previous_best_fitness = np.min(fitness)

        while evaluations < self.budget:
            adap_state = (self.budget - evaluations) / self.budget
            de_scale = 0.1 + 0.4 * adap_state  # Adaptive DE scaling
            crossover_rate = 0.6 + 0.3 * (1 - adap_state)  # Adaptive crossover rate
            cma_sigma = 0.3 + 0.2 * adap_state  # Adaptive CMA sigma

            # Dynamic population size adjustment
            if evaluations > self.budget * 0.5:
                population_size = max(2, int(initial_population_size * adap_state))

            # Random immigrants to introduce diversity
            num_immigrants = max(1, int(population_size * 0.1))
            immigrants = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_immigrants, self.dim))
            pop_with_immigrants = np.vstack((population, immigrants))
            fitness_with_immigrants = np.concatenate([fitness, np.array([func(ind) for ind in immigrants])])
            evaluations += num_immigrants

            # Differential Evolution Step
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

            # Environmental selection to maintain diversity
            combined_population = np.vstack((population, pop_with_immigrants))
            combined_fitness = np.concatenate((fitness, fitness_with_immigrants))
            best_indices = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # CMA-ES component
            current_best_fitness = np.min(fitness)
            fitness_variance = np.var(fitness)
            if current_best_fitness < previous_best_fitness:
                previous_best_fitness = current_best_fitness
                if fitness_variance < 1e-5:
                    best_indices = np.argsort(fitness)[:cma_population_size]
                    cma_population = population[best_indices]
                    cma_mean = np.mean(cma_population, axis=0)
                    cma_cov = np.cov(cma_population.T)
                    cma_samples = np.random.multivariate_normal(cma_mean, cma_sigma**2 * cma_cov, cma_population_size)

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

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]