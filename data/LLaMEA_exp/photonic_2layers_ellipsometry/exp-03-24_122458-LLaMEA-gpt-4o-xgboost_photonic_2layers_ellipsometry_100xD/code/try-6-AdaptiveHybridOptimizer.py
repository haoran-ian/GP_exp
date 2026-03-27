import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.temp_initial = 1.0
        self.temp_final = 0.01
        self.evaluations = 0
        self.base_population_size = 10 * dim

    def initialize_population(self, bounds, size):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(size)]

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        return population[idx[0]] if fitness[idx[0]] < fitness[idx[1]] else population[idx[1]]

    def mutate(self, individual, bounds, mutation_rate=0.1):
        dynamic_mutation_rate = mutation_rate * (1 - self.evaluations / self.budget)
        mutation_scale = 0.5 * (1 + np.sin(self.evaluations / self.budget * np.pi))
        if np.random.rand() < dynamic_mutation_rate:
            mutation_vector = np.random.normal(0, mutation_scale, size=self.dim)
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

    def adjust_population_size(self):
        return max(self.base_population_size // (1 + self.evaluations // (self.budget // 4)), 1)

    def __call__(self, func):
        bounds = func.bounds
        population_size = self.base_population_size
        population = self.initialize_population(bounds, population_size)
        fitness = self.evaluate_population(population, func)
        best_individual = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        self.evaluations += population_size

        while self.evaluations < self.budget:
            population_size = self.adjust_population_size()
            new_population = []
            for _ in range(population_size):
                parent1 = self.select_parents(population, fitness)
                parent2 = self.select_parents(population, fitness)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring, bounds)
                new_population.append(offspring)
            
            new_fitness = self.evaluate_population(new_population, func)
            self.evaluations += population_size

            for i in range(population_size):
                temperature = self.temp_initial * ((self.temp_final / self.temp_initial) ** (self.evaluations / self.budget))
                new_population[i] = self.simulated_annealing(new_population[i], func, bounds, temperature)
                new_fitness[i] = func(new_population[i])

            combined_population = population + new_population
            combined_fitness = fitness + new_fitness

            selected_indices = np.argsort(combined_fitness)[:population_size]
            population = [combined_population[i] for i in selected_indices]
            fitness = [combined_fitness[i] for i in selected_indices]

            current_best = min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmin(fitness)]

        return best_individual, best_fitness