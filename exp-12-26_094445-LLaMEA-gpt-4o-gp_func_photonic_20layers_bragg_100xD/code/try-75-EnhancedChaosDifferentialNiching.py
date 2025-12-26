import numpy as np

class EnhancedChaosDifferentialNiching:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.min_population_size = 5
        self.initial_temperature = 1.0
        self.cooling_rate = 0.95
        self.temperature_threshold = 0.1
        self.niche_radius = 0.1

    def adaptive_scaling(self, current_fitness, best_fitness):
        return min(0.8, max(0.4, (best_fitness - current_fitness) / max(abs(best_fitness), 1e-10) + 0.4))

    def chaotic_map(self, x):
        return 4.0 * x * (1.0 - x)

    def niche_selection(self, population, fitness):
        niche_pop = []
        niche_fit = []
        for idx, ind in enumerate(population):
            is_unique = True
            for niche in niche_pop:
                if np.linalg.norm(ind - niche) < self.niche_radius:
                    is_unique = False
                    break
            if is_unique:
                niche_pop.append(ind)
                niche_fit.append(fitness[idx])
        return np.array(niche_pop), np.array(niche_fit)

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evals = self.population_size
        temperature = self.initial_temperature
        chaos_value = np.random.rand()

        while evals < self.budget:
            current_population_size = len(population)
            for i in range(current_population_size):
                idxs = [idx for idx in range(current_population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = self.adaptive_scaling(fitness[i], best_fitness)
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < 0.9
                offspring = np.where(cross_points, mutant, population[i])

                offspring_fitness = func(offspring)
                evals += 1

                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness

                    if offspring_fitness < best_fitness or np.exp((best_fitness - offspring_fitness) / temperature) > np.random.rand():
                        best_solution = offspring
                        best_fitness = offspring_fitness

                temperature *= self.cooling_rate
                if temperature < self.temperature_threshold:
                    temperature = self.temperature_threshold

                if evals >= self.budget:
                    break

            population, fitness = self.niche_selection(population, fitness)
            current_population_size = len(population)

            # Apply chaotic perturbation to enhance exploration
            chaos_value = self.chaotic_map(chaos_value)
            perturbation = chaos_value * (upper_bound - lower_bound) * 0.1
            population += np.random.uniform(-1, 1, population.shape) * perturbation
            population = np.clip(population, lower_bound, upper_bound)

        return best_solution