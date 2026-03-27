import numpy as np

class EnhancedHybridChaoticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.evaluations = 0
        self.elite_fraction = 0.1
        self.chaos_parameter = 0.7
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.1
        self.CR_max = 0.9

    def initialize_population(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        return population[idx[0]] if fitness[idx[0]] < fitness[idx[1]] else population[idx[1]]

    def mutate(self, individual, bounds, mutation_rate=0.1, diversity_factor=0):
        dynamic_mutation_rate = mutation_rate * (1 - self.evaluations / self.budget) ** 2 * (1 + diversity_factor)
        if np.random.rand() < dynamic_mutation_rate:
            mutation_vector = np.random.normal(0, 1, size=self.dim)
            new_individual = individual + mutation_vector
            return np.clip(new_individual, bounds.lb, bounds.ub)
        return individual

    def crossover(self, parent1, parent2, diversity_factor=0):
        beta = np.random.rand() * (1 + diversity_factor)
        return beta * parent1 + (1 - beta) * parent2

    def differential_evolution_strategy(self, target, donor, bounds, F, CR):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < CR:
                trial[i] = target[i] + F * (donor[i] - target[i])
        return np.clip(trial, bounds.lb, bounds.ub)

    def chaotic_map(self, x):
        return self.chaos_parameter * x * (1 - x)

    def adaptive_inertia(self):
        return self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (self.evaluations / self.budget)

    def levy_flight(self, L=1.5):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return 0.01 * step

    def swarm_update(self, population, global_best, bounds):
        for i in range(len(population)):
            inertia_weight = self.adaptive_inertia()
            personal_best = population[i]
            chaotic_factor = self.chaotic_map(np.random.rand())
            velocity = chaotic_factor * (personal_best - population[i]) + (1 - chaotic_factor) * (global_best - population[i])
            velocity *= inertia_weight
            population[i] += velocity
            population[i] = np.clip(population[i], bounds.lb, bounds.ub)
        return population

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        best_individual = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            new_population = []
            current_elite_fraction = self.elite_fraction * (1 - (self.evaluations / self.budget)**2)
            elites = int(current_elite_fraction * self.population_size)
            sorted_indices = np.argsort(fitness)
            elites_indices = sorted_indices[:elites]
            elites_population = [population[i] for i in elites_indices]

            # Calculate diversity factor
            diversity_factor = np.std(fitness) / (np.mean(fitness) + 1e-9)

            for _ in range(self.population_size - elites):
                parent1 = self.select_parents(population, fitness)
                parent2 = self.select_parents(population, fitness)
                donor = self.select_parents(population, fitness)
                
                # Self-adaptive parameter control
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                
                offspring = self.crossover(parent1, parent2, diversity_factor)
                offspring = self.differential_evolution_strategy(offspring, donor, bounds, F, CR)
                offspring = self.mutate(offspring, bounds, diversity_factor=diversity_factor)
                
                # Chaotic Levy Flight
                if np.random.rand() < 0.1:
                    offspring += self.levy_flight()
                
                new_population.append(offspring)

            new_population.extend(elites_population)
            new_fitness = self.evaluate_population(new_population, func)
            self.evaluations += self.population_size

            combined_population = population + new_population
            combined_fitness = fitness + new_fitness

            selected_indices = np.argsort(combined_fitness)[:self.population_size]
            population = [combined_population[i] for i in selected_indices]
            fitness = [combined_fitness[i] for i in selected_indices]

            current_best = min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmin(fitness)]

            population = self.swarm_update(population, best_individual, bounds)

        return best_individual, best_fitness