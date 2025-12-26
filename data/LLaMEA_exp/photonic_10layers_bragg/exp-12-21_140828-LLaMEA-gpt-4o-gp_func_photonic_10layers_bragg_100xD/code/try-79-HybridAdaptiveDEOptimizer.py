import numpy as np

class HybridAdaptiveDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.elitism_rate = 0.1

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

            # Retain top performers using elitism
            elite_indices = np.argsort(fitness)[:elitism_count]
            trial_population[:elitism_count] = population[elite_indices]

            for i in range(elitism_count, population_size):
                if evals >= self.budget:
                    break

                # Multi-strategy DE mutation
                if np.random.rand() < 0.5:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                else:
                    indices = np.random.permutation(population_size)[:5]
                    a, b, c, d, e = population[indices]
                    a = a + self.mutation_factor * (b - c) + self.mutation_factor * (d - e)

                weight = (fitness[indices[0]] - fitness[i]) / (1e-9 + max_fitness_diff)
                mutant = np.clip(a + weight * self.mutation_factor * (b - c), lb, ub)

                # Adaptive crossover
                self.crossover_probability = 0.9 - (fitness_variance / (1e-9 + np.max(fitness_variance))) * 0.5
                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                evals += 1

                # Selection and adaptation
                if trial_fitness < fitness[i]:
                    trial_population[i] = trial
                    fitness[i] = trial_fitness
                    self.mutation_factor = min(1.0, self.mutation_factor + self.adaptation_rate * 1.1)
                else:
                    trial_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.adaptation_rate)

                # Local and global hybrid search
                if evals < self.budget:
                    perturbation = self.local_search_perturbation + 0.01 * (1 - fitness_variance)
                    global_search = (np.random.rand() > 0.5)
                    if global_search:
                        random_point = np.random.uniform(lb, ub, self.dim)
                        trial_population[i] = random_point
                        trial_fitness = func(random_point)
                        evals += 1
                        if trial_fitness < fitness[i]:
                            fitness[i] = trial_fitness
                    else:
                        local_trial = trial + perturbation * np.random.normal(size=self.dim)
                        local_trial = np.clip(local_trial, lb, ub)
                        local_fitness = func(local_trial)
                        evals += 1
                        if local_fitness < fitness[i]:
                            trial_population[i] = local_trial
                            fitness[i] = local_fitness

            population[:] = trial_population

            # Adaptive Population Resizing
            if evals < self.budget:
                population_size = max(10, int(self.initial_population_size * (1 - evals / self.budget)))
                population = population[:population_size]
                fitness = fitness[:population_size]
                trial_population = trial_population[:population_size]

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]