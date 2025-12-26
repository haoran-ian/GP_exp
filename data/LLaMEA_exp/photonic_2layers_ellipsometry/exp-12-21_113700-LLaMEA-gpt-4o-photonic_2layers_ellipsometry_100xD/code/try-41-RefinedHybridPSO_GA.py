import numpy as np

class RefinedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_rate_initial = 0.2
        self.mutation_rate_final = 0.05
        self.cross_over_rate = 0.7
        self.elitism_rate = 0.1  # Elitism rate to retain the top performers
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size
        t = 0

        while evaluations < self.budget:
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            mutation_rate = self.mutation_rate_initial - (self.mutation_rate_initial - self.mutation_rate_final) * (evaluations / self.budget)

            # Dynamic neighborhood topology
            neighborhood_size = max(2, int(self.population_size * 0.3))
            for i in range(self.population_size):
                neighbors_indices = np.random.choice(self.population_size, neighborhood_size, replace=False)
                local_best = personal_best[neighbors_indices[np.argmin(personal_best_scores[neighbors_indices])]]
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] + 
                                 self.c1_initial * r1 * (personal_best[i] - population[i]) + 
                                 self.c2_initial * r2 * (local_best - population[i]))
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], lb, ub)

            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best[improved] = population[improved]
            personal_best_scores[improved] = scores[improved]
            
            if min(scores) < global_best_score:
                global_best = population[np.argmin(scores)]
                global_best_score = min(scores)

            # Apply Genetic Operators
            elite_size = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(personal_best_scores)[:elite_size]
            elite_individuals = population[elite_indices]
            
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.cross_over_rate:
                    cross_point = np.random.randint(1, self.dim)
                    parent1, parent2 = population[i], population[i + 1]
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                    population[i], population[i + 1] = child1, child2

            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate
            mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
            population = np.where(mutation_mask, mutation_values, population)

            # Retain elite individuals
            population[:elite_size] = elite_individuals

            t += 1

        return global_best, global_best_score