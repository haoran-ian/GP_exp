import numpy as np

class E_AHSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.mutation_prob = 0.1
        self.crossover_prob = 0.7
        self.evaluations = 0

    def adaptive_inertia_weight(self):
        w = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * (self.evaluations / self.budget))
        return w

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        while self.evaluations < self.budget:
            scores = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            # Update personal and global bests
            better_positions = scores < personal_best_scores
            personal_best_positions[better_positions] = population[better_positions]
            personal_best_scores[better_positions] = scores[better_positions]
            
            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_position = personal_best_positions[min_score_idx]
                global_best_score = personal_best_scores[min_score_idx]

            # Update velocities and positions (PSO dynamics with adaptive inertia)
            w = self.adaptive_inertia_weight()
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = population + velocities
            population = np.clip(population, lb, ub)

            # Genetic Algorithm Crossover and Differential Evolution-inspired Mutation
            offspring = []
            for _ in range(self.population_size // 2):
                if np.random.rand() < self.crossover_prob:
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    cutpoint = np.random.randint(1, self.dim - 1)
                    child1 = np.concatenate((population[parents[0], :cutpoint],
                                             population[parents[1], cutpoint:]))
                    child2 = np.concatenate((population[parents[1], :cutpoint],
                                             population[parents[0], cutpoint:]))
                    offspring.extend([child1, child2])
                else:
                    offspring.extend(population[np.random.choice(self.population_size, 2)])
            
            # Differential Evolution-inspired Mutation
            F = 0.8
            mutation_population = []
            for individual in offspring:
                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = a + F * (b - c)
                    mutant = np.clip(mutant, lb, ub)
                    mutation_population.append(mutant)
                else:
                    mutation_population.append(individual)
            offspring = np.array(mutation_population)

            # Replace worst half of the population with offspring
            worst_idx = np.argsort(scores)[-self.population_size//2:]
            population[worst_idx] = offspring[:len(worst_idx)]

        return global_best_position, global_best_score