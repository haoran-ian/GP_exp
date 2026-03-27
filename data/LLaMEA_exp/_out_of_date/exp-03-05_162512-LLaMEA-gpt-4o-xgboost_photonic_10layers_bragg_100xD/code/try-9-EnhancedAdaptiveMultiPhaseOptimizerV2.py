import numpy as np

class EnhancedAdaptiveMultiPhaseOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(100, budget // 10)
        self.num_phases = 3
        self.phase_lengths = [self.budget // self.num_phases] * self.num_phases
        self.elite_fraction = 0.1  # Fraction of the population retained as elite members

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        
        def initialize_population():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def exploitation_step(best_individual):
            # Directional mutation with adaptive step size
            scale_factor = max(0.1, 0.5 * (1 - self.evaluations / self.budget))
            direction = np.random.normal(0, scale_factor, (self.population_size, self.dim))
            neighbors = best_individual + direction
            neighbors = np.clip(neighbors, lb, ub)
            return neighbors

        def exploration_step():
            # Maintain diversity with random initialization
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        # Phase 1: Exploration with random population
        population = initialize_population()
        fitness = evaluate_population(population)

        for phase in range(self.num_phases):
            if self.evaluations >= self.budget:
                break

            if phase == 0:
                # Global exploration phase
                population = exploration_step()
            elif phase == 1:
                # Intermediate phase with crossover and diversity preservation
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
                # Local exploitation phase with directional mutation
                best_idx = np.argmin(fitness)
                best_individual = population[best_idx]
                population = exploitation_step(best_individual)

            # Elitist strategy: retain best individuals across generations
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Re-evaluate updated population
            fitness = evaluate_population(population)

            # Reinstate elite individuals
            population[:elite_size] = elite_population
            fitness[:elite_size] = elite_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]