import numpy as np

class EnhancedDynamicAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = min(100, budget // 10)
        self.population_size = self.initial_population_size
        self.num_phases = 3
        self.phase_lengths = [self.budget // self.num_phases] * self.num_phases

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        
        def initialize_population():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def adaptive_mutation(individual, best_individual, step_size):
            direction = np.random.normal(0, step_size, self.dim)
            return np.clip(individual + direction * (best_individual - individual), lb, ub)

        def exploration_step():
            # Maintain diversity by random initialization
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        # Dynamic population resizing based on available budget
        def dynamic_resize(factor):
            new_size = max(int(self.initial_population_size * factor), 2)
            indices = np.argsort(fitness)[:new_size]
            return population[indices], fitness[indices]

        population = initialize_population()
        fitness = evaluate_population(population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        
        for phase in range(self.num_phases):
            if self.evaluations >= self.budget:
                break

            step_size = 0.1 / (phase + 1)  # Adaptive step size

            if phase == 0:
                population = exploration_step()
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
                for i in range(self.population_size):
                    population[i] = adaptive_mutation(population[i], best_individual, step_size)

            fitness = evaluate_population(population)
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            # Dynamically resize the population for the next phase based on performance
            if phase < self.num_phases - 1:
                population, fitness = dynamic_resize(factor=0.8 + 0.1 * phase)

        return best_individual