import numpy as np
import scipy.spatial.distance as dist

class AdaptiveMemeticGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.niche_radius = 0.1
        self.memory_size = 5
        self.memory = []
        self.evaluations = 0

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

            selected = self._niche_selection(population, fitness)
            offspring = self._crossover(selected, bounds)
            population = self._mutation(offspring, bounds)

            best_solution = self._local_search(best_solution, func, bounds)

            self._dynamic_niching(population)
            self._adaptive_parameter_adjustment(fitness)

        return best_solution

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))

    def _niche_selection(self, population, fitness):
        selected_idx = np.random.choice(np.argsort(fitness)[:self.population_size // 2], size=self.population_size // 2, replace=True)
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
                mutation_vector = np.random.normal(0, 0.1 * np.std(offspring, axis=0), self.dim)
                offspring[i] = offspring[i] + mutation_vector
                offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
        return offspring

    def _update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            improvement = np.min([np.linalg.norm(solution - mem) for mem in self.memory])
            if improvement > self.niche_radius:
                self.memory[np.random.randint(0, self.memory_size)] = solution

    def _dynamic_niching(self, population):
        dist_matrix = dist.squareform(dist.pdist(population))
        niche_count = np.sum(dist_matrix < self.niche_radius, axis=0)
        self.niche_radius = np.median(niche_count) / self.population_size

    def _adaptive_parameter_adjustment(self, fitness):
        improvement = (np.min(fitness) - np.mean(fitness)) / np.std(fitness)
        if improvement < 0.05:
            self.mutation_rate = min(self.mutation_rate * 1.1, 0.5)
            self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)
        else:
            self.mutation_rate = max(self.mutation_rate * 0.9, 0.05)
            self.crossover_rate = min(self.crossover_rate * 1.1, 0.9)

    def _local_search(self, solution, func, bounds):
        step_size = 0.005
        for _ in range(10):
            candidate = solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            if func(candidate) < func(solution):
                solution = candidate
        return solution