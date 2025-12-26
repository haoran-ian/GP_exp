import numpy as np
from scipy.stats import qmc

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
        previous_best_fitness = np.min(fitness)
        fitness_history = [previous_best_fitness]
        sobol_engine = qmc.Sobol(d=self.dim, scramble=True)

        while evaluations < self.budget:
            adap_state = (self.budget - evaluations) / self.budget
            de_scale = max(0.1, np.mean(fitness_history[-5:]) / np.std(fitness_history[-5:] + 1e-9)) * 0.5
            crossover_rate = 0.6 + 0.3 * (1 - adap_state)
            cma_sigma = 0.3 + 0.2 * adap_state

            num_immigrants = max(1, int(population_size * 0.1))
            sobol_points = sobol_engine.random(num_immigrants)
            immigrants = np.array([bounds[:, 0] + point * (bounds[:, 1] - bounds[:, 0]) for point in sobol_points])
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
            fitness_history.append(current_best_fitness)
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