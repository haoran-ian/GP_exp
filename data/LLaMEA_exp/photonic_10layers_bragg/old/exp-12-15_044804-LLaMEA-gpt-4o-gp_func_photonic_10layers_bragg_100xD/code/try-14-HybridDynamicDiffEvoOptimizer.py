import numpy as np

class HybridDynamicDiffEvoOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 * dim
        self.initial_mutation_rate = 0.1
        self.mutation_decay_factor = 0.995
        self.adaptive_crossover_prob = 0.7
        self.crossover_decay_factor = 0.98
        self.evaluation_chunk = 100

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        mutation_rate = self.initial_mutation_rate
        
        def dynamic_adaptation(evals):
            if evals % self.evaluation_chunk == 0:
                self.adaptive_crossover_prob *= self.crossover_decay_factor
            return evals % (self.evaluation_chunk * 2) == 0

        while evaluations < self.budget:
            parents_idx = np.argsort(fitness)[:self.population_size // 2]
            parents = population[parents_idx]

            offspring = []
            for _ in range(self.population_size // 2):
                idxs = np.random.choice(len(parents), 3, replace=False)
                x0, x1, x2 = parents[idxs[0]], parents[idxs[1]], parents[idxs[2]]
                mutant_vector = np.clip(x0 + mutation_rate * (x1 - x2), lb, ub)
                child = np.where(np.random.rand(self.dim) < self.adaptive_crossover_prob, mutant_vector, x0)
                offspring.append(child)

            offspring = np.array(offspring)
            mutation_rate *= self.mutation_decay_factor

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            combined_population = np.vstack((population, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))
            diversity_score = np.std(combined_population, axis=0)
            selection_metric = combined_fitness - diversity_score.sum() * 0.01  # Line changed for diversity promotion
            best_idx = np.argsort(selection_metric)[:self.population_size]
            population = combined_population[best_idx]
            fitness = combined_fitness[best_idx]

            if dynamic_adaptation(evaluations):
                print(f"Dynamic adaptation at evaluation {evaluations}. Crossover probability: {self.adaptive_crossover_prob:.4f}")

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]