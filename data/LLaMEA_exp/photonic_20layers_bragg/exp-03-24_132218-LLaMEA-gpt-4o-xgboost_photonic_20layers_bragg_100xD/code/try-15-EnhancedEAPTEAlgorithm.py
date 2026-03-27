import numpy as np

class EnhancedEAPTEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        initial_pop_size = 10 + 2 * self.dim
        max_pop_size = initial_pop_size * 2
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += initial_pop_size

        elite_archive = []
        adaptive_mutation_rate = 0.1

        def local_search(x, adapt_strength):
            mutation_var = adapt_strength * np.random.rand()
            candidate = x + np.random.randn(self.dim) * mutation_var
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def differential_crossover(a, b, c):
            F = 0.5 + np.random.rand() * 0.5
            candidate = a + F * (b - c)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        while self.eval_count < self.budget:
            adapt_strength = 0.5 + 0.5 * (1 - (self.eval_count / self.budget))

            for i in range(len(population)):
                if self.eval_count >= self.budget:
                    break

                if np.random.rand() < adaptive_mutation_rate:
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    if elite_archive and np.random.rand() < 0.3:
                        elite_index = np.random.choice(len(elite_archive))
                        new_solution = local_search(elite_archive[elite_index], adapt_strength)
                    elif len(population) >= 3:
                        idxs = np.random.choice(len(population), 3, replace=False)
                        a, b, c = population[idxs]
                        new_solution = differential_crossover(a, b, c)
                    else:
                        new_solution = local_search(population[i], adapt_strength)

                new_value = func(new_solution)
                self.eval_count += 1

                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                    elite_archive.append(new_solution)

                    if len(elite_archive) > max_pop_size:
                        elite_archive.pop(0)

                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value

            if len(population) < max_pop_size and np.random.rand() < 0.1:
                extra = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size // 2, self.dim))
                pop_fitness = np.array([func(x) for x in extra])
                self.eval_count += len(extra)
                population = np.vstack((population, extra))
                fitness = np.hstack((fitness, pop_fitness))

            # Adjust mutation rate based on population diversity
            diversity = np.std(fitness)
            adaptive_mutation_rate = 0.1 + 0.4 * (1 - diversity / (diversity + 0.1))

        return best_solution, best_value