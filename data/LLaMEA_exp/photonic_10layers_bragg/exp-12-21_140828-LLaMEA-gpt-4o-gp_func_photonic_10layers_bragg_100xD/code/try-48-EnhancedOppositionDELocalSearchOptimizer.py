import numpy as np

class EnhancedOppositionDELocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.restart_threshold = 0.001

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
            for i in range(population_size):
                if evals >= self.budget:
                    break

                # Fitness-weighted Differential Evolution Mutation
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                weight = (fitness[indices[0]] - fitness[i]) / (1e-9 + max_fitness_diff)
                mutant = np.clip(a + weight * self.mutation_factor * (b - c), lb, ub)

                # Adaptive Crossover with dynamic adjustment
                self.crossover_probability = 0.9 - (fitness_variance / (1e-9 + np.max(fitness_variance))) * 0.5
                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                evals += 1

                # Selection with adaptation
                if trial_fitness < fitness[i]:
                    trial_population[i] = trial
                    fitness[i] = trial_fitness
                    self.mutation_factor = min(1.0, self.mutation_factor + self.adaptation_rate * 1.1)
                else:
                    trial_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.adaptation_rate)

                # Adaptive Stochastic Local Search
                if evals < self.budget:
                    perturbation = self.local_search_perturbation + 0.01 * (1 - fitness_variance)
                    local_trial = trial + perturbation * np.random.normal(size=self.dim)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[i]:
                        trial_population[i] = local_trial
                        fitness[i] = local_fitness

            population[:] = trial_population

            # Dynamic Population Resizing
            if evals < self.budget:
                population_size = max(10, int(self.initial_population_size * (1 - evals / self.budget)))
                population = population[:population_size]
                fitness = fitness[:population_size]
                trial_population = trial_population[:population_size]

            # Opposition-based Learning
            if evals < self.budget:
                opposition_population = lb + ub - population
                opposition_fitness = np.array([func(ind) for ind in opposition_population])
                evals += population_size
                improved_indices = opposition_fitness < fitness
                population[improved_indices] = opposition_population[improved_indices]
                fitness[improved_indices] = opposition_fitness[improved_indices]

            # Adaptive Restart Mechanism
            if fitness_variance < self.restart_threshold:
                population = np.random.uniform(lb, ub, (population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                evals += population_size

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]