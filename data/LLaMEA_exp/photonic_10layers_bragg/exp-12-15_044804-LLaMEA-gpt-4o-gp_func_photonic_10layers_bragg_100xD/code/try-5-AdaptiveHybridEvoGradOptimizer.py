import numpy as np

class AdaptiveHybridEvoGradOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 * dim
        self.initial_mutation_rate = 0.1
        self.gradient_alpha = 0.01
        self.mutation_decay_factor = 0.99  # New: decay factor for adaptive mutation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        mutation_rate = self.initial_mutation_rate
        
        while evaluations < self.budget:
            # Selection
            parents_idx = np.argsort(fitness)[:self.population_size // 2]
            parents = population[parents_idx]

            # Crossover
            offspring = []
            for _ in range(self.population_size // 2):
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                child = np.where(np.random.rand(self.dim) < 0.5, p1, p2)
                offspring.append(child)

            # Mutation with decay
            offspring = np.array(offspring)
            mutations = np.random.normal(0, mutation_rate, offspring.shape)
            offspring += mutations
            offspring = np.clip(offspring, lb, ub)

            # Evaluate offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)
            mutation_rate *= self.mutation_decay_factor  # New: decay mutation rate

            # Gradient refinement with dynamic adjustment
            for i in range(len(offspring)):
                grad = self._estimate_gradient(func, offspring[i], lb, ub) if np.random.rand() < 0.5 else np.zeros(self.dim)  # Change: selective gradient estimation
                alpha_adjusted = self.gradient_alpha / (1.0 + np.linalg.norm(grad))  # New: dynamic adjustment
                offspring[i] -= alpha_adjusted * grad
                offspring[i] = np.clip(offspring[i], lb, ub)
                offspring_fitness[i] = func(offspring[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break

            # Combine and select the next generation
            combined_population = np.vstack((parents, offspring))
            combined_fitness = np.hstack((fitness[parents_idx], offspring_fitness))
            best_idx = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_idx]
            fitness = combined_fitness[best_idx]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def _estimate_gradient(self, func, x, lb, ub, epsilon=1e-5):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = np.copy(x)
            x_minus = np.copy(x)
            x_plus[i] = min(x[i] + epsilon, ub[i])
            x_minus[i] = max(x[i] - epsilon, lb[i])
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        return grad