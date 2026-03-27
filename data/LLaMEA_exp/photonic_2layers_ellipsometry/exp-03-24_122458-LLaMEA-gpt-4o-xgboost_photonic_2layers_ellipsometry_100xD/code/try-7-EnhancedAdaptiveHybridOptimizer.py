import numpy as np

class EnhancedAdaptiveHybridOptimizer:
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
        idx = np.random.choice(np.arange(len(population)), size=3, replace=False)
        return min(idx, key=lambda i: fitness[i])

    def mutate(self, individual, bounds, mutation_rate=0.1):
        dynamic_mutation_rate = mutation_rate * (1 - (self.evaluations / self.budget) ** 2)  # Refined dynamic mutation rate
        if np.random.rand() < dynamic_mutation_rate:
            mutation_vector = np.random.normal(0, 0.5, size=self.dim)
            new_individual = individual + mutation_vector
            return np.clip(new_individual, bounds.lb, bounds.ub)
        return individual

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        return alpha * parent1 + (1 - alpha) * parent2

    def simulated_annealing(self, candidate, func, bounds, temp):
        perturbed = self.mutate(candidate, bounds, mutation_rate=0.5)
        candidate_eval = func(candidate)
        perturbed_eval = func(perturbed)
        self.evaluations += 2
        if perturbed_eval < candidate_eval:
            return perturbed
        else:
            prob = np.exp((candidate_eval - perturbed_eval) / temp)
            return perturbed if np.random.rand() < prob else candidate

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        best_individual = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        self.evaluations += self.population_size

        elite_fraction = 0.2  # Preserve more elite individuals
        num_elites = max(1, int(elite_fraction * self.population_size))

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                parent1_idx = self.select_parents(population, fitness)
                parent2_idx = self.select_parents(population, fitness)
                offspring = self.crossover(population[parent1_idx], population[parent2_idx])
                offspring = self.mutate(offspring, bounds)
                new_population.append(offspring)
            
            new_fitness = self.evaluate_population(new_population, func)
            self.evaluations += self.population_size

            for i in range(self.population_size):
                temperature = self.temp_initial * ((self.temp_final / self.temp_initial) ** (self.evaluations / self.budget))
                new_population[i] = self.simulated_annealing(new_population[i], func, bounds, temperature)
                new_fitness[i] = func(new_population[i])

            combined_population = population + new_population
            combined_fitness = fitness + new_fitness

            selected_indices = np.argsort(combined_fitness)[:self.population_size - num_elites]
            elites_indices = np.argsort(fitness)[:num_elites]
            population = [combined_population[i] for i in selected_indices] + [population[i] for i in elites_indices]
            fitness = [combined_fitness[i] for i in selected_indices] + [fitness[i] for i in elites_indices]

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx]

        return best_individual, best_fitness