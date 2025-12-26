import numpy as np
from scipy.spatial.distance import pdist, squareform

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

        success_de = 0
        success_cma = 0
        adap_counter = 0

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

            success_de = 0
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
                    success_de += 1

            current_best_fitness = np.min(fitness)
            fitness_variance = np.var(fitness)
            if current_best_fitness < previous_best_fitness:
                previous_best_fitness = current_best_fitness

                if fitness_variance < 1e-5 or success_de / population_size < 0.1:
                    best_indices = np.argsort(fitness)[:cma_population_size]
                    cma_population = population[best_indices]
                    cma_mean = np.mean(cma_population, axis=0)
                    cma_cov = np.cov(cma_population.T)
                    cma_samples = np.random.multivariate_normal(cma_mean, cma_sigma**2 * cma_cov, cma_population_size)
                    success_cma = 0

                    for sample in cma_samples:
                        sample = np.clip(sample, bounds[:, 0], bounds[:, 1])
                        sample_fitness = func(sample)
                        evaluations += 1

                        if sample_fitness < np.max(fitness):
                            worst_index = np.argmax(fitness)
                            population[worst_index] = sample
                            fitness[worst_index] = sample_fitness
                            success_cma += 1

                        if evaluations >= self.budget:
                            break

            # Dynamic strategy adjustment based on success rates
            if success_de > success_cma:
                cma_sigma *= 0.9
            else:
                de_scale *= 0.9

            # Enhanced exploration through characterizing niches
            if adap_counter % 10 == 0:
                distances = squareform(pdist(population))
                diversity_preserved = np.zeros(population_size, dtype=bool)
                local_exploration_size = max(1, population_size // 5)
                
                for i in range(population_size):
                    if not diversity_preserved[i]:
                        niching_indices = np.where(distances[i] < 0.1)[0]
                        best_in_niche = niching_indices[np.argmin(fitness[niching_indices])]
                        diversity_preserved[niching_indices] = True
                        diversity_preserved[best_in_niche] = False
                        
                        # Local exploitation within niche
                        local_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], (local_exploration_size, self.dim))
                        local_fitness = np.array([func(candidate) for candidate in local_candidates])
                        evaluations += local_exploration_size
                        
                        local_best_index = np.argmin(local_fitness)
                        if local_fitness[local_best_index] < fitness[best_in_niche]:
                            population[best_in_niche] = local_candidates[local_best_index]
                            fitness[best_in_niche] = local_fitness[local_best_index]

                population = population[~diversity_preserved]
                fitness = fitness[~diversity_preserved]
                missing_count = population_size - len(population)
                new_individuals = np.random.uniform(bounds[:, 0], bounds[:, 1], (missing_count, self.dim))
                population = np.vstack((population, new_individuals))
                fitness = np.concatenate((fitness, [func(ind) for ind in new_individuals]))
                evaluations += missing_count

            adap_counter += 1

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]