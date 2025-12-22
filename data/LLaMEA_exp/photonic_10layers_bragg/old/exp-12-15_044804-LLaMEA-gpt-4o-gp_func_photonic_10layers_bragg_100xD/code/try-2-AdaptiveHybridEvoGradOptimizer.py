import numpy as np

class AdaptiveHybridEvoGradOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 5 * dim)
        self.initial_mutation_rate = 0.1
        self.gradient_alpha = 0.01
        self.mutation_decay_factor = 0.99
        self.learning_rate_decay = 0.995  # New: decay for learning rate
        self.dynamic_population_increase = 0.1  # New: dynamic population adjustment rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        mutation_rate = self.initial_mutation_rate
        gradient_alpha = self.gradient_alpha

        while evaluations < self.budget:
            # Selection
            selected_indices = np.argsort(fitness)[:self.population_size // 2]
            parents = population[selected_indices]

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

            # Adjust mutation rate and learning rate
            mutation_rate *= self.mutation_decay_factor
            gradient_alpha *= self.learning_rate_decay

            # Gradient refinement with dynamic learning rate
            for i in range(len(offspring)):
                grad = self._estimate_gradient(func, offspring[i], lb, ub)
                offspring[i] -= gradient_alpha * grad
                offspring[i] = np.clip(offspring[i], lb, ub)
                offspring_fitness[i] = func(offspring[i])
                evaluations += 1
                if evaluations >= self.budget:
                    break

            # Combine and select the next generation
            combined_population = np.vstack((parents, offspring))
            combined_fitness = np.hstack((fitness[selected_indices], offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # Dynamic population size increase
            if evaluations < self.budget // 2:
                self.population_size = int(self.population_size * (1 + self.dynamic_population_increase))
                self.population_size = min(self.population_size, self.budget - evaluations)

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