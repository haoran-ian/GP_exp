import numpy as np

class DynamicRadiusEliteOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(100, budget // 10)
        self.num_phases = 3
        self.phase_lengths = [self.budget // self.num_phases] * self.num_phases
        self.inertia_weight = 0.7
        self.elite_archive = []
        self.initial_radius = 0.1
        self.mutation_rate = 0.2

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        def initialize_population():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def adaptive_exploitation_step(best_individual, radius):
            search_radius = radius * (self.evaluations / self.budget)  # Dynamic reduction of radius
            neighbors = best_individual + search_radius * np.random.normal(0, 1, (self.population_size, self.dim))
            neighbors = np.clip(neighbors, lb, ub)
            return neighbors

        def exploration_step():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        def mutate(individual):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 0.1, self.dim)
                individual = np.clip(individual + mutation_vector, lb, ub)
            return individual

        population = initialize_population()
        fitness = evaluate_population(population)

        for phase in range(self.num_phases):
            if self.evaluations >= self.budget:
                break

            if phase == 0:
                population = exploration_step()
            elif phase == 1:
                for _ in range(self.phase_lengths[phase]):
                    if self.evaluations >= self.budget:
                        break
                    parents = population[np.argsort(fitness)[:2]]
                    offspring = crossover(parents[0], parents[1])
                    offspring = mutate(offspring)
                    offspring_fitness = func(offspring)
                    self.evaluations += 1
                    if offspring_fitness < max(fitness):
                        replace_idx = np.argmax(fitness)
                        population[replace_idx] = offspring
                        fitness[replace_idx] = offspring_fitness
            else:
                best_idx = np.argmin(fitness)
                best_individual = population[best_idx]
                population = adaptive_exploitation_step(best_individual, self.initial_radius)

            fitness = evaluate_population(population)
            if len(self.elite_archive) < 5:
                self.elite_archive.append(population[np.argmin(fitness)])
            else:
                if func(population[np.argmin(fitness)]) < func(max(self.elite_archive, key=func)):
                    self.elite_archive.remove(max(self.elite_archive, key=func))
                    self.elite_archive.append(population[np.argmin(fitness)])

        best_idx = np.argmin(fitness)
        return population[best_idx] if not self.elite_archive else min(self.elite_archive, key=func)