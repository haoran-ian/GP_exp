import numpy as np

class AdaptiveHybridDESA2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.successful_mutations = []
        self.successful_crossovers = []
        self.temperature = 1.0

    def differential_evolution(self, population, fitness, func):
        new_population = np.copy(population)
        diversity = np.mean(np.std(population, axis=0))
        adaptive_mutation_factor = self.mutation_factor * (1 + diversity) * (1 + np.mean(self.successful_mutations))
        adaptive_crossover_rate = self.crossover_rate * (1 - diversity) * (1 + np.mean(self.successful_crossovers))
        
        inertia_weight = 0.9 - (0.9 - 0.4) * (fitness - min(fitness)) / (max(fitness) - min(fitness) + 1e-7)

        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = (1 - inertia_weight[i]) * population[i] + inertia_weight[i] * (population[a] + adaptive_mutation_factor * (population[b] - population[c]))
            trial_vector = np.copy(population[i])
            crossover_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial_vector[crossover_points] = mutant_vector[crossover_points]
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[i]:
                new_population[i] = trial_vector
                fitness[i] = trial_fitness
                self.successful_mutations.append(1)
                self.successful_crossovers.append(1)
            else:
                self.successful_mutations.append(0)
                self.successful_crossovers.append(0)
        return new_population, fitness

    def greedy_local_search(self, candidate, candidate_fitness, func, bounds):
        step_size = (bounds.ub - bounds.lb) * 0.01
        for _ in range(5):  # Reduced iterations for speed
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            new_candidate = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
            new_fitness = func(new_candidate)
            if new_fitness < candidate_fitness:
                candidate, candidate_fitness = new_candidate, new_fitness
                break  # Greedy acceptance
        return candidate, candidate_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            population, fitness = self.differential_evolution(population, fitness, func)
            evaluations += self.population_size

            best_idx = np.argmin(fitness)
            best_candidate, best_fitness = population[best_idx], fitness[best_idx]
            new_candidate, new_fitness = self.greedy_local_search(best_candidate, best_fitness, func, bounds)

            if new_fitness < best_fitness:
                population[best_idx] = new_candidate
                fitness[best_idx] = new_fitness

            evaluations += 1
            self.temperature *= 0.99

            if evaluations + 5 <= self.budget:
                best_candidate, best_fitness = self.greedy_local_search(best_candidate, best_fitness, func, bounds)
                if best_fitness < fitness[best_idx]:
                    population[best_idx] = best_candidate
                    fitness[best_idx] = best_fitness
                evaluations += 5

        return population[np.argmin(fitness)]