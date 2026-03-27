import numpy as np

class EnhancedAdaptiveInertiaOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(100, budget // 10)
        self.num_swarms = 5  # Introduced multiple swarms for parallelization
        self.swarm_sizes = [self.population_size // self.num_swarms] * self.num_swarms
        self.inertia_weight = 0.7  # Adaptive inertia weight
        self.mutation_rate = 0.05  # Adaptive mutation rate
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

        def adaptive_exploitation_step(population, fitness):
            # Multi-swarm directional mutation with adaptive inertia and mutation
            best_indices = np.argpartition(fitness, self.num_swarms)[:self.num_swarms]
            for i in range(self.num_swarms):
                swarm_best = population[best_indices[i]]
                velocity = self.inertia_weight * np.random.normal(0, 0.1, (self.swarm_sizes[i], self.dim))
                mutation = self.mutation_rate * np.random.normal(0, 0.2, (self.swarm_sizes[i], self.dim))
                swarm = swarm_best + velocity + mutation
                swarm = np.clip(swarm, lb, ub)
                population[i*self.swarm_sizes[i]:(i+1)*self.swarm_sizes[i]] = swarm
            return population

        def exploration_step():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        # Phase 1: Exploration with random population
        population = initialize_population()
        fitness = evaluate_population(population)
        velocity = np.zeros((self.population_size, self.dim))  # Initialize velocity for inertia

        for phase in range(3):
            if self.evaluations >= self.budget:
                break

            if phase == 0:
                # Global exploration phase
                population = exploration_step()
            elif phase == 1:
                # Intermediate phase with crossover and diversity preservation
                for _ in range(self.population_size):
                    if self.evaluations >= self.budget:
                        break
                    parents = np.random.choice(population.shape[0], 2, replace=False)
                    offspring = crossover(population[parents[0]], population[parents[1]])
                    offspring = np.clip(offspring, lb, ub)
                    offspring_fitness = func(offspring)
                    self.evaluations += 1
                    if offspring_fitness < max(fitness):
                        replace_idx = np.argmax(fitness)
                        population[replace_idx] = offspring
                        fitness[replace_idx] = offspring_fitness
            else:
                # Local exploitation phase with adaptive steps
                population = adaptive_exploitation_step(population, fitness)

            fitness = evaluate_population(population)
            best_idx = np.argmin(fitness)
            self.elite_archive.append(population[best_idx])  # Archive the best individual

        best_idx = np.argmin(fitness)
        return population[best_idx] if not self.elite_archive else min(self.elite_archive, key=func)