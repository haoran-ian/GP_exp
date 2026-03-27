import numpy as np

class EnhancedAdaptiveParallelOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(100, budget // 10)
        self.num_phases = 3
        self.phase_lengths = [self.budget // self.num_phases] * self.num_phases
        self.inertia_weight = 0.7
        self.elite_archive = []
        self.mutation_scale = 0.1

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        
        def initialize_population():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def adaptive_exploitation(best_individual, velocity):
            velocity = self.inertia_weight * velocity + np.random.normal(0, self.mutation_scale, (self.population_size, self.dim))
            neighbors = best_individual + velocity
            neighbors = np.clip(neighbors, lb, ub)
            return neighbors, velocity

        def adaptive_exploration():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        # Phase 1: Exploration
        population = initialize_population()
        fitness = evaluate_population(population)
        velocity = np.zeros((self.population_size, self.dim))

        for phase in range(self.num_phases):
            if self.evaluations >= self.budget:
                break

            if phase == 0:
                population = adaptive_exploration()
            elif phase == 1:
                for _ in range(self.phase_lengths[phase]):
                    if self.evaluations >= self.budget:
                        break
                    parents = population[np.argsort(fitness)[:2]]
                    offspring = crossover(parents[0], parents[1])
                    offspring = np.clip(offspring, lb, ub)
                    offspring_fitness = func(offspring)
                    self.evaluations += 1
                    if offspring_fitness < max(fitness):
                        replace_idx = np.argmax(fitness)
                        population[replace_idx] = offspring
                        fitness[replace_idx] = offspring_fitness
            else:
                best_idx = np.argmin(fitness)
                best_individual = population[best_idx]
                population, velocity = adaptive_exploitation(best_individual, velocity)

            fitness = evaluate_population(population)
            if len(self.elite_archive) == 0 or min(fitness) < min(map(func, self.elite_archive)):
                self.elite_archive.append(population[np.argmin(fitness)])

        best_idx = np.argmin(fitness)
        return population[best_idx] if not self.elite_archive else min(self.elite_archive, key=func)