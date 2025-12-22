import numpy as np

class EnhancedDynamicControlDiffEvoOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 5 * dim
        self.initial_mutation_rate = 0.1
        self.initial_crossover_prob = 0.7
        self.evaluation_chunk = 100
        self.success_memory_length = 30
        self.success_memory = []
        self.min_population_size = dim
        self.max_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_rate = self.initial_mutation_rate
        self.crossover_prob = self.initial_crossover_prob

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
                mutant_vector = np.clip(x0 + self.mutation_rate * diversity_factor * (x1 - x2), lb, ub)
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

                # Adjust mutation and crossover rates dynamically
                convergence_speed = np.abs(fitness.min() - fitness.mean())
                self.mutation_rate = max(0.01, self.mutation_rate * (1 + (0.5 - average_success_rate) * 0.1))
                self.crossover_prob = min(1.0, max(0.3, self.crossover_prob * (1 + (0.5 - convergence_speed) * 0.1)))

                # Adapt population size based on success rate and diversity
                diversity = np.std(population)
                if average_success_rate > 0.5 and diversity < 0.1:
                    self.population_size = max(self.min_population_size, self.population_size - 1)
                elif average_success_rate < 0.3:
                    self.population_size = min(self.max_population_size, self.population_size + 1)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]