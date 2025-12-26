import numpy as np
from scipy.spatial.distance import pdist, squareform

class AdaptiveHybridDE_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 4 + int(3 * np.log(self.dim))
        de_scale_min, de_scale_max = 0.1, 0.9
        crossover_rate_min, crossover_rate_max = 0.1, 0.9
        cma_population_size = population_size // 2
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        previous_best_fitness = np.min(fitness)
        restart_threshold = 1e-6

        while evaluations < self.budget:
            adap_state = (self.budget - evaluations) / self.budget
            de_scale = de_scale_min + (de_scale_max - de_scale_min) * adap_state
            crossover_rate = crossover_rate_min + (crossover_rate_max - crossover_rate_min) * (1 - adap_state)

            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
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
            if current_best_fitness < previous_best_fitness:
                previous_best_fitness = current_best_fitness

                if fitness_variance < 1e-5:
                    best_indices = np.argsort(fitness)[:cma_population_size]
                    cma_population = population[best_indices]
                    cma_mean = np.mean(cma_population, axis=0)
                    cma_cov = np.cov(cma_population.T) + 1e-8 * np.eye(self.dim)
                    cma_samples = np.random.multivariate_normal(cma_mean, 0.3**2 * cma_cov, cma_population_size)

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

            if fitness_variance < restart_threshold:
                population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                evaluations += population_size

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]