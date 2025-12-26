import numpy as np

class RefinedEnhancedDiffStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * dim  # Slightly increased to improve diversity
        self.mutation_factor = 0.6
        self.crossover_probability = 0.85
        self.local_search_perturbation = 0.04  # Reduced to focus on global search initially
        self.adaptation_rate = 0.15  # Adjusted for smoother mutation factor adaptation
        self.elitism_rate = 0.15  # Increased to ensure better preservation of top solutions
        self.dynamic_fitness_weight = 0.5  # Additional mechanism to balance exploration-exploitation

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
                
                # Use elitism to retain top performers
                if i < elitism_count:
                    trial_population[i] = population[np.argsort(fitness)[:elitism_count][i]]
                    continue

                # Enhanced Multi-strategy Differential Evolution Mutation
                mutation_strategy = np.random.rand()
                if mutation_strategy < 0.3:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = a + self.mutation_factor * (b - c)
                elif mutation_strategy < 0.6:
                    indices = np.random.permutation(population_size)[:5]
                    a, b, c, d, e = population[indices]
                    mutant = a + self.mutation_factor * (b - c + d - e)
                else:
                    indices = np.random.permutation(population_size)[:5]
                    a, b, c, d, e = population[indices]
                    mutant = (a + b + c - d - e) / 3

                mutant = np.clip(mutant, lb, ub)
                
                # Adaptive Crossover with dynamic adjustment
                adaptive_cp = 0.8 + self.dynamic_fitness_weight * (fitness_variance / (1e-9 + np.max(fitness_variance)))
                crossover = np.random.rand(self.dim) < adaptive_cp
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
                    self.mutation_factor = min(1.0, self.mutation_factor + self.adaptation_rate * 1.2)
                else:
                    trial_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.adaptation_rate)

                # Adaptive Stochastic Local Search
                if evals < self.budget:
                    perturbation = self.local_search_perturbation * (1 + np.log1p(fitness_variance))
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

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]