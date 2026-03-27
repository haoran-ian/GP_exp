import numpy as np

class AMSEAlgorithm:
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

        memory = []
        pheromones = np.zeros((max_pop_size, self.dim))

        def update_pheromones():
            decay_rate = 0.1
            pheromones[:] *= (1 - decay_rate)
            for i, fit in enumerate(fitness):
                contribution = (best_value - fit) / best_value
                pheromones[i] += contribution

        def guided_mutation(x, idx):
            noise = np.random.randn(self.dim) * (0.1 + 0.9 * pheromones[idx]) * (1 - (self.eval_count / self.budget))
            candidate = x + noise
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        def adaptive_crossover(a, b, c):
            F = 0.3 + 0.7 * np.random.rand() * (1 - (self.eval_count / self.budget))
            candidate = a + F * (b - c)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            return candidate

        while self.eval_count < self.budget:
            update_pheromones()
            adapt_strength = 0.5 + 0.5 * (1 - (self.eval_count / self.budget))

            for i in range(len(population)):
                if self.eval_count >= self.budget:
                    break

                if np.random.rand() < 0.4:
                    new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
                else:
                    if memory and np.random.rand() < 0.3:
                        mem_index = np.random.choice(len(memory))
                        new_solution = guided_mutation(memory[mem_index], mem_index)
                    elif len(population) >= 3:
                        idxs = np.random.choice(len(population), 3, replace=False)
                        a, b, c = population[idxs]
                        new_solution = adaptive_crossover(a, b, c)
                    else:
                        new_solution = guided_mutation(population[i], i)

                new_value = func(new_solution)
                self.eval_count += 1

                if new_value < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_value
                    memory.append(new_solution)

                    if len(memory) > max_pop_size:
                        memory.pop(np.random.choice(len(memory)))  # Adaptive memory decay

                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value

            if len(population) < max_pop_size and np.random.rand() < 0.2:
                additional = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_pop_size // 2, self.dim))
                pop_fitness = np.array([func(x) for x in additional])
                self.eval_count += len(additional)
                population = np.vstack((population, additional))
                fitness = np.hstack((fitness, pop_fitness))

        return best_solution, best_value