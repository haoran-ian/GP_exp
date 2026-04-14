import numpy as np

class EnhancedGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.memory_size = 5
        self.memory = []
        self.evaluations = 0
        self.history = []

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self._initialize_population(bounds)
        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, population)
            self.evaluations += len(fitness)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx]
                self._update_memory(best_solution)

            # Store historical performance
            self.history.append((self.evaluations, best_fitness))

            selected = self._selection(population, fitness)
            offspring = self._crossover(selected, bounds)
            population = self._mutation(offspring, bounds)

            # Dynamic parameter adjustment based on historical progress
            self._adjust_parameters()

        return best_solution

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))

    def _selection(self, population, fitness):
        selected_idx = np.argsort(fitness)[:self.population_size // 2]
        return population[selected_idx]

    def _crossover(self, selected, bounds):
        offspring = []
        for i in range(self.population_size // 2):
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
                cross_point = np.random.randint(1, self.dim - 1)
                child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                offspring.append(child)
            else:
                offspring.append(selected[i])
        return np.clip(offspring, bounds[:, 0], bounds[:, 1])

    def _mutation(self, offspring, bounds):
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 0.1, self.dim)
                offspring[i] = offspring[i] + mutation_vector
                offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
        return offspring

    def _update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            self.memory[np.random.randint(0, self.memory_size)] = solution

    def _adjust_parameters(self):
        if len(self.history) < 2:
            return
        last_eval, last_fitness = self.history[-1]
        prev_eval, prev_fitness = self.history[-2]
        improvement = (prev_fitness - last_fitness) / (last_fitness + 1e-10)
        
        if improvement < 0.01:
            self.mutation_rate = min(self.mutation_rate * 1.1, 0.5)
            self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)
        else:
            self.mutation_rate = max(self.mutation_rate * 0.9, 0.05)
            self.crossover_rate = min(self.crossover_rate * 1.1, 0.9)