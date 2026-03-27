import numpy as np
from scipy.spatial.distance import pdist, squareform

class RefinedHybridDE_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 4 + int(3 * np.log(self.dim))
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        temp_initial = 1.0
        temp_min = 0.01
        cooling_rate = 0.99

        de_scale = 0.5
        crossover_rate = 0.9
        cma_population_size = population_size // 2
        cma_sigma = 0.5

        while evaluations < self.budget:
            temperature = max(temp_min, temp_initial * (cooling_rate ** (evaluations / self.budget)))

            if np.random.rand() < 0.5:
                de_scale = 0.1 + 0.4 * temperature
                crossover_rate = 0.6 + 0.3 * (1 - temperature)
            else:
                cma_sigma = 0.3 + 0.2 * temperature

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

            distances = squareform(pdist(population))
            diversity_preserved = np.zeros(population_size, dtype=bool)
            for i in range(population_size):
                if not diversity_preserved[i]:
                    niching_indices = np.where(distances[i] < 0.1)[0]
                    best_in_niche = niching_indices[np.argmin(fitness[niching_indices])]
                    diversity_preserved[niching_indices] = True
                    diversity_preserved[best_in_niche] = False

            population = population[~diversity_preserved]
            fitness = fitness[~diversity_preserved]
            missing_count = population_size - len(population)
            new_individuals = np.random.uniform(bounds[:, 0], bounds[:, 1], (missing_count, self.dim))
            population = np.vstack((population, new_individuals))
            fitness = np.concatenate((fitness, [func(ind) for ind in new_individuals]))
            evaluations += missing_count

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]