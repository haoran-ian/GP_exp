import numpy as np

class EnhancedHybridDE_CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 4 + int(3 * np.log(self.dim))
        de_scale = 0.5
        crossover_rate = 0.9
        cma_population_size = population_size // 2
        cma_sigma = 0.5
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            # Adaptive Differential Evolution Step
            fitness_std = np.std(fitness)
            scale_factor = 0.5 + 0.5 * fitness_std / max(fitness)
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + scale_factor * (b - c), bounds[:, 0], bounds[:, 1])
                crossover = np.random.rand(self.dim) < crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Select best individuals for CMA-ES with adaptive covariance
            best_indices = np.argsort(fitness)[:cma_population_size]
            cma_population = population[best_indices]
            cma_mean = np.mean(cma_population, axis=0)
            cma_cov = np.cov(cma_population.T) + 1e-5 * np.eye(self.dim)  # Avoid singular matrix
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