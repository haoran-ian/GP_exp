import numpy as np

class EnhancedAdaptiveInertiaOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(100, budget // 10)
        self.num_phases = 3
        self.phase_lengths = [self.budget // self.num_phases] * self.num_phases
        self.inertia_weight = 0.7  # Adaptive inertia weight
        self.elite_archive = []

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        
        def initialize_population():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def exploitation_step(best_individual, velocity):
            # Directional mutation with adaptive inertia
            velocity = self.inertia_weight * velocity + np.random.normal(0, 0.1, (self.population_size, self.dim))
            neighbors = best_individual + velocity * (np.random.rand(self.population_size, self.dim))
            neighbors = np.clip(neighbors, lb, ub)
            return neighbors, velocity

        def exploration_step():
            # Maintain diversity by random initialization
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim) * 0.5  # Dynamic crossover strategy
            return alpha * parent1 + (1 - alpha) * parent2

        # Phase 1: Exploration with random population
        population = initialize_population()
        fitness = evaluate_population(population)

        velocity = np.zeros((self.population_size, self.dim))  # Initialize velocity for inertia

        for phase in range(self.num_phases):
            if self.evaluations >= self.budget:
                break

            if phase == 0:
                # Global exploration phase
                population = exploration_step()
            elif phase == 1:
                # Intermediate phase with crossover and diversity preservation
                for _ in range(int(self.phase_lengths[phase] * 1.1)):  # Adaptive phase length
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
                # Local exploitation phase with directional mutation
                best_idx = np.argmin(fitness)
                best_individual = population[best_idx]
                population, velocity = exploitation_step(best_individual, velocity)

            fitness = evaluate_population(population)
            self.elite_archive.append(population[np.argmin(fitness)])  # Archive the best individual

        best_idx = np.argmin(fitness)
        return population[best_idx] if not self.elite_archive else min(self.elite_archive, key=func)