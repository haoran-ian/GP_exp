import numpy as np

class MultiScaleAdaptiveDiffEvoOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(5 * dim, 50)
        self.mutation_rate = 0.1
        self.crossover_prob = 0.7
        self.evaluation_chunk = 100
        self.mutation_rate_adaptation_factor = 0.05
        self.crossover_rate_adaptation_factor = 0.03
        self.best_mutation = None
        self.success_memory_length = 30
        self.success_memory = []
        self.population_size = self.initial_population_size
        self.min_population_size = dim
        self.max_population_size = 10 * dim
        self.adaptive_learning_rate = 1.0
        self.diversity_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        def evaluate_and_select(pop, fit):
            offspring_fitness = np.array([func(ind) for ind in pop])
            combined_population = np.vstack((population, pop))
            combined_fitness = np.hstack((fitness, offspring_fitness))
            best_idx = np.argsort(combined_fitness)[:self.population_size]
            return combined_population[best_idx], combined_fitness[best_idx], offspring_fitness

        while evaluations < self.budget:
            parents_idx = np.argsort(fitness)[:self.population_size // 2]
            parents = population[parents_idx]

            offspring = []
            for _ in range(self.population_size // 2):
                idxs = np.random.choice(len(parents), 3, replace=False)
                x0, x1, x2 = parents[idxs[0]], parents[idxs[1]], parents[idxs[2]]
                diversity_factor = np.std(population) / self.dim
                adapt_scaling = 1 + self.adaptive_learning_rate * diversity_factor
                mutant_vector = np.clip(x0 + self.mutation_rate * adapt_scaling * (x1 - x2), lb, ub)
                if self.best_mutation is not None:
                    mutant_vector = (mutant_vector + self.best_mutation + np.mean(population, axis=0)) / 3.0
                child = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, x0)
                offspring.append(child)

            offspring = np.array(offspring)
            evaluations += len(offspring)
            population, fitness, offspring_fitness = evaluate_and_select(offspring, fitness)

            if evaluations % self.evaluation_chunk == 0:
                successful_trials = np.sum(offspring_fitness < fitness[:len(offspring_fitness)])
                success_rate = successful_trials / len(offspring_fitness)
                self.success_memory.append(success_rate)
                if len(self.success_memory) > self.success_memory_length:
                    self.success_memory.pop(0)
                average_success_rate = np.mean(self.success_memory)

                if len(self.success_memory) > 1:
                    trend = self.success_memory[-1] - self.success_memory[-2]
                    self.adaptive_learning_rate = max(0.1, self.adaptive_learning_rate + trend)

                self.mutation_rate = max(0.01, self.mutation_rate + self.adaptive_learning_rate * self.mutation_rate_adaptation_factor * (average_success_rate - 0.5))
                self.crossover_prob = min(1.0, max(0.3, self.crossover_prob + self.adaptive_learning_rate * self.crossover_rate_adaptation_factor * (average_success_rate - 0.5)))

                diversity = np.std(population)
                if average_success_rate > 0.5 and diversity < self.diversity_threshold:
                    self.population_size = max(self.min_population_size, self.population_size - int(diversity * self.population_size + 1))
                elif average_success_rate < 0.3:
                    self.population_size = min(self.max_population_size, self.population_size + int((0.3 - average_success_rate) * self.population_size + 1))

            best_offspring_idx = np.argmin(offspring_fitness)
            if self.best_mutation is None or offspring_fitness[best_offspring_idx] < func(self.best_mutation):
                self.best_mutation = offspring[best_offspring_idx]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]