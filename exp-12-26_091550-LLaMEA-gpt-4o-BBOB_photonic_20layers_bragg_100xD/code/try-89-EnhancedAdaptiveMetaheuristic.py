import numpy as np

class EnhancedAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 2 * dim)
        self.learning_rate = 0.1
        self.strategy_pool = [self.random_search, self.gradient_search, self.crossover_mutation]
        self.strategy_probabilities = np.ones(len(self.strategy_pool)) / len(self.strategy_pool)
        self.diversity_threshold = 0.1  # Threshold for maintaining population diversity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            strategy_idx = np.random.choice(len(self.strategy_pool), p=self.strategy_probabilities)
            new_population = self.strategy_pool[strategy_idx](population, func, lb, ub)
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += len(new_population)

            # Update best solution
            new_best_idx = np.argmin(new_fitness)
            if new_fitness[new_best_idx] < best_fitness:
                best_fitness = new_fitness[new_best_idx]
                best_solution = new_population[new_best_idx].copy()

            # Update strategy probabilities
            success = new_fitness < fitness
            self.strategy_probabilities[strategy_idx] = max(0.1, 
                self.strategy_probabilities[strategy_idx] + self.learning_rate * success.mean())
            self.strategy_probabilities /= self.strategy_probabilities.sum()

            # Diversity preservation
            if self.measure_diversity(population) < self.diversity_threshold:
                population = self.introduce_diversity(population, lb, ub)

            # Move to the next generation
            population, fitness = new_population, new_fitness

        return best_solution

    def random_search(self, population, func, lb, ub):
        return np.random.uniform(lb, ub, population.shape)

    def gradient_search(self, population, func, lb, ub):
        perturbation = np.random.randn(*population.shape) * 0.1
        candidates = population + perturbation
        np.clip(candidates, lb, ub, out=candidates)
        return candidates

    def crossover_mutation(self, population, func, lb, ub):
        offspring = []
        for _ in range(self.population_size):
            parents = population[np.random.choice(len(population), 2, replace=False)]
            cross_point = np.random.randint(1, self.dim - 1)
            child = np.concatenate((parents[0][:cross_point], parents[1][cross_point:]))
            mutation = np.random.randn(self.dim) * (0.05 + 0.1 * np.random.rand())  # Adaptive mutation
            child += mutation
            np.clip(child, lb, ub, out=child)
            offspring.append(child)
        return np.array(offspring)

    def measure_diversity(self, population):
        pairwise_distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
        avg_distance = np.mean(pairwise_distances)
        return avg_distance

    def introduce_diversity(self, population, lb, ub):
        num_new_individuals = self.population_size // 2
        new_individuals = np.random.uniform(lb, ub, (num_new_individuals, self.dim))
        return np.vstack((population, new_individuals))