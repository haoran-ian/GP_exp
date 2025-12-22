import numpy as np

class EnhancedDynamicDELocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.inertia_weight = 0.9

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            trial_population = np.empty_like(population)
            fitness_variance = np.var(fitness)
            max_fitness_diff = np.max(fitness) - np.min(fitness)
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Dynamic inertia-weighted Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                weight = (fitness[indices[0]] - fitness[i]) / (1e-9 + max_fitness_diff)
                self.inertia_weight = 0.5 + 0.4 * np.random.rand()
                mutant = np.clip(a + self.inertia_weight * weight * self.mutation_factor * (b - c), lb, ub)

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

                # Gradient-based Stochastic Local Search
                if evals < self.budget:
                    grad_perturbation = np.random.normal(size=self.dim)
                    grad_trial = trial - self.local_search_perturbation * grad_perturbation
                    grad_trial = np.clip(grad_trial, lb, ub)
                    grad_fitness = func(grad_trial)
                    evals += 1
                    if grad_fitness < fitness[i]:
                        trial_population[i] = grad_trial
                        fitness[i] = grad_fitness

            population[:] = trial_population

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]