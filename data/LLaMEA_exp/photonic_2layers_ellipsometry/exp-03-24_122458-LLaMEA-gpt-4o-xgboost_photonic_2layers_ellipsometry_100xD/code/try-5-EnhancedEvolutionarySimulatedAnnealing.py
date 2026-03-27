import numpy as np

class EnhancedEvolutionarySimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.temp_initial = 1.0
        self.temp_final = 0.01
        self.evaluations = 0

    def initialize_population(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        return population[idx[0]] if fitness[idx[0]] < fitness[idx[1]] else population[idx[1]]

    def mutate(self, individual, bounds, mutation_rate=0.1):
        dynamic_mutation_rate = mutation_rate * (1 - self.evaluations / self.budget)
        if np.random.rand() < dynamic_mutation_rate:
            mutation_vector = np.random.normal(0, 0.5, size=self.dim)
            new_individual = individual + mutation_vector
            return np.clip(new_individual, bounds.lb, bounds.ub)
        return individual

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        return alpha * parent1 + (1 - alpha) * parent2

    def adaptive_perturbation(self, candidate, func, bounds, temp):
        perturbation_scale = max(0.01, temp)
        perturbed = candidate + np.random.normal(0, perturbation_scale, size=self.dim)
        perturbed = np.clip(perturbed, bounds.lb, bounds.ub)
        candidate_eval = func(candidate)
        perturbed_eval = func(perturbed)
        self.evaluations += 2
        if perturbed_eval < candidate_eval:
            return perturbed
        else:
            prob = np.exp((candidate_eval - perturbed_eval) / temp)
            return perturbed if np.random.rand() < prob else candidate

    def hierarchical_ranking_selection(self, population, fitness, new_fitness):
        combined_population = population + new_population
        combined_fitness = fitness + new_fitness
        ranking = np.argsort(combined_fitness)
        return [combined_population[i] for i in ranking[:self.population_size]], [combined_fitness[i] for i in ranking[:self.population_size]]

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        best_individual = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select_parents(population, fitness)
                parent2 = self.select_parents(population, fitness)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring, bounds)
                new_population.append(offspring)
            
            new_fitness = self.evaluate_population(new_population, func)
            self.evaluations += self.population_size

            for i in range(self.population_size):
                temperature = self.temp_initial * ((self.temp_final / self.temp_initial) ** (self.evaluations / self.budget))
                new_population[i] = self.adaptive_perturbation(new_population[i], func, bounds, temperature)
                new_fitness[i] = func(new_population[i])

            population, fitness = self.hierarchical_ranking_selection(population, fitness, new_fitness)

            current_best = min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmin(fitness)]

        return best_individual, best_fitness