import numpy as np

class DynamicEliteDiversifier:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(100, budget // 10)
        self.num_phases = 3
        self.inertia_weight = 0.7
        self.elite_archive = []
        self.diversification_rate = 0.1
        self.velocity_layers = 3

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        def initialize_population():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def adaptive_exploitation_step(best_individual, velocities):
            stochastic_factor = np.random.uniform(0.25, 0.75)
            for layer in range(self.velocity_layers):
                learning_factor = 0.5 + np.random.rand() * 0.5  # Adjusted learning factor range
                velocities[layer] = (self.inertia_weight * velocities[layer] +
                                     learning_factor * stochastic_factor * np.random.normal(0, 0.1, (self.population_size, self.dim)))
            velocity_combined = np.mean(velocities, axis=0)
            neighbors = best_individual + velocity_combined * (np.random.rand(self.population_size, self.dim))
            neighbors = np.clip(neighbors, lb, ub)
            return neighbors, velocities

        def exploration_step():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        def diversify_population(population):
            num_to_diversify = int(self.diversification_rate * self.population_size)
            new_individuals = np.random.uniform(lb, ub, (num_to_diversify, self.dim))
            population[:num_to_diversify] = new_individuals
            return population

        population = initialize_population()
        fitness = evaluate_population(population)

        velocities = [np.zeros((self.population_size, self.dim)) for _ in range(self.velocity_layers)]

        for phase in range(self.num_phases):
            if self.evaluations >= self.budget:
                break

            if phase == 0:
                population = exploration_step()
            elif phase == 1:
                for _ in range(self.budget // 3):
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
                population, velocities = adaptive_exploitation_step(best_individual, velocities)

            if phase == 2 and self.evaluations < self.budget // 2:
                population = diversify_population(population)
            
            fitness = evaluate_population(population)
            if len(self.elite_archive) < 5:
                self.elite_archive.append(population[np.argmin(fitness)])
            else:
                if func(population[np.argmin(fitness)]) < func(max(self.elite_archive, key=func)):
                    self.elite_archive.remove(max(self.elite_archive, key=func))
                    self.elite_archive.append(population[np.argmin(fitness)])

        best_idx = np.argmin(fitness)
        return population[best_idx] if not self.elite_archive else min(self.elite_archive, key=func)