import numpy as np

class EnhancedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_c1 = 2.0  # initial cognitive parameter
        self.initial_c2 = 2.0  # initial social parameter
        self.w = 0.9  # inertia weight
        self.mutation_rate = 0.1
        self.cross_over_rate = 0.7
        self.local_search_rate = 0.2

    def adaptive_parameters(self, evaluations):
        """Adapt parameters based on the number of evaluations."""
        progress = evaluations / self.budget
        c1 = self.initial_c1 * (1 - progress) + 1.5 * progress
        c2 = self.initial_c2 * progress + 1.5 * (1 - progress)
        w = 0.9 - 0.5 * progress
        return c1, c2, w

    def local_search(self, individual, func, lb, ub):
        """Refine individual using a simple local search strategy."""
        for _ in range(int(self.local_search_rate * self.dim)):
            candidate = individual + np.random.uniform(-0.1, 0.1, self.dim)
            candidate = np.clip(candidate, lb, ub)
            if func(candidate) < func(individual):
                individual = candidate
        return individual

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            # Adapt parameters
            c1, c2, w = self.adaptive_parameters(evaluations)

            # Update velocities and population positions (PSO Step)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best - population) +
                          c2 * r2 * (global_best - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Evaluate the population
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Update personal best and global best
            improved = scores < personal_best_scores
            personal_best[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            if min(scores) < global_best_score:
                global_best = population[np.argmin(scores)]
                global_best_score = min(scores)

            # Apply Genetic Operators (GA Step)
            # Crossover
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.cross_over_rate:
                    cross_point = np.random.randint(1, self.dim)
                    parent1, parent2 = population[i], population[i + 1]
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                    population[i], population[i + 1] = child1, child2

            # Mutation
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

            # Local Search
            for i in range(self.population_size):
                population[i] = self.local_search(population[i], func, lb, ub)

        return global_best, global_best_score