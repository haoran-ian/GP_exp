import numpy as np

class EnhancedHybridDELocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.convergence_speedup = 0.1

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            trial_population = np.empty_like(population)
            fitness_variance = np.var(fitness)
            diversity = np.mean(np.std(population, axis=0))
            
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Dynamic Scaling Mutation Factor
                adjusted_mutation_factor = self.mutation_factor * (1 + diversity)
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + adjusted_mutation_factor * (b - c), lb, ub)

                # Adaptive Crossover with dynamic adjustment
                self.crossover_probability = 0.9 - (fitness_variance / (1e-9 + np.max(fitness_variance))) * 0.5
                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                evals += 1

                # Selection with adaptation and convergence speedup
                if trial_fitness < fitness[i]:
                    trial_population[i] = trial
                    fitness[i] = trial_fitness
                    self.mutation_factor = min(1.0, self.mutation_factor + self.adaptation_rate * 1.1)
                else:
                    trial_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.adaptation_rate)
                    
                if evals > self.budget / 2:  # Apply convergence speedup in the second half of evaluations
                    self.local_search_perturbation *= self.convergence_speedup

                # Stochastic Local Search
                if evals < self.budget:
                    local_trial = trial + (self.local_search_perturbation + 0.01 * fitness_variance) * np.random.normal(size=self.dim)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[i]:
                        trial_population[i] = local_trial
                        fitness[i] = local_fitness

            population[:] = trial_population

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]