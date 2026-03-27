import numpy as np

class EnhancedDifferentialEvolutionWithLevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.evaluations = 0
        self.elite_fraction = 0.1
        self.chaos_parameter = 0.7
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4

    def initialize_population(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        return population[idx[0]] if fitness[idx[0]] < fitness[idx[1]] else population[idx[1]]

    def levy_flight(self, individual, bounds, alpha=0.01):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / beta)
        new_individual = individual + alpha * step
        return np.clip(new_individual, bounds.lb, bounds.ub)

    def adaptive_crossover(self, parent1, parent2, fitness1, fitness2):
        if fitness1 < fitness2:
            return np.random.uniform(0.4, 0.9) * parent1 + (1 - np.random.uniform(0.4, 0.9)) * parent2
        else:
            return np.random.uniform(0.1, 0.6) * parent1 + (1 - np.random.uniform(0.1, 0.6)) * parent2

    def differential_evolution_strategy(self, target, donor, bounds, F=0.8, CR=0.9):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < CR:
                trial[i] = target[i] + F * (donor[i] - target[i])
        return np.clip(trial, bounds.lb, bounds.ub)

    def chaotic_map(self, x):
        return self.chaos_parameter * x * (1 - x)

    def adaptive_inertia(self):
        return self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (self.evaluations / self.budget)

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
            current_elite_fraction = self.elite_fraction * (1 - self.evaluations / self.budget)
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
                offspring = self.adaptive_crossover(parent1, parent2, func(parent1), func(parent2))
                offspring = self.differential_evolution_strategy(offspring, donor, bounds)
                offspring = self.levy_flight(offspring, bounds)
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