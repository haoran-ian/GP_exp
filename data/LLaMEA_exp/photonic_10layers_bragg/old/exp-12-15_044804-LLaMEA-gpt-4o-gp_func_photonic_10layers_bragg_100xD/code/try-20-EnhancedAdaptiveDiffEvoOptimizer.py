import numpy as np

class EnhancedAdaptiveDiffEvoOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 * dim
        self.initial_mutation_rate = 0.1
        self.initial_crossover_prob = 0.7
        self.evaluation_chunk = 100
        self.mutation_rate_adaptation_factor = 0.05
        self.crossover_rate_adaptation_factor = 0.03
        self.best_mutation = None  # New attribute to store the best mutation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        mutation_rate = self.initial_mutation_rate
        crossover_prob = self.initial_crossover_prob
        improvement_trend = []

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
                mutant_vector = np.clip(x0 + mutation_rate * diversity_factor * (x1 - x2), lb, ub)
                if self.best_mutation is not None:  # Use the best mutation if available
                    mutant_vector = (0.7 * mutant_vector + 0.3 * self.best_mutation)  # Weighted combination
                child = np.where(np.random.rand(self.dim) < crossover_prob, mutant_vector, x0)
                offspring.append(child)

            offspring = np.array(offspring)
            evaluations += len(offspring)
            population, fitness, offspring_fitness = evaluate_and_select(offspring, fitness)

            if evaluations % self.evaluation_chunk == 0:
                recent_improvement = np.mean([np.min(offspring_fitness) < np.min(fitness)])
                improvement_trend.append(recent_improvement)
                if len(improvement_trend) > 1:
                    recent_trend = improvement_trend[-1] - improvement_trend[-2]
                    mutation_rate = max(0.01, mutation_rate + self.mutation_rate_adaptation_factor * recent_trend)
                    crossover_prob = min(1.0, max(0.3, crossover_prob + self.crossover_rate_adaptation_factor * recent_trend))

            best_offspring_idx = np.argmin(offspring_fitness)
            if self.best_mutation is None or offspring_fitness[best_offspring_idx] < func(self.best_mutation):
                self.best_mutation = offspring[best_offspring_idx]  # Update the best mutation

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]