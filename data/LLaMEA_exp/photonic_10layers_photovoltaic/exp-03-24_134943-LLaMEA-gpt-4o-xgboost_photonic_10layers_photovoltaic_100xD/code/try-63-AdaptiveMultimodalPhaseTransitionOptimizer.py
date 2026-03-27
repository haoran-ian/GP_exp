import numpy as np

class AdaptiveMultimodalPhaseTransitionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.base_population_size = 30
        self.inner_population_size = 15
        self.populations = 3
        self.phase_detection_threshold = 1e-3
        self.exploration_factors = np.linspace(0.9, 1.1, self.populations)
        self.exploitation_factors = np.linspace(0.2, 0.3, self.populations)
        self.dynamic_ratios = np.linspace(0.05, 0.15, self.populations)
        self.learning_rates = np.linspace(0.01, 0.1, self.populations)

    def initialize_populations(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, (self.base_population_size, self.dim)) for _ in range(self.populations)]

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(population)
        return fitness

    def detect_phase_transition(self, population):
        distances = np.linalg.norm(np.diff(population, axis=0), axis=1)
        return np.mean(distances) > self.phase_detection_threshold

    def exploit(self, best_solution, bounds, exploitation_factor, learning_rate):
        noise = learning_rate * np.random.uniform(-0.5, 0.5, (self.inner_population_size, self.dim))
        return np.clip(best_solution + exploitation_factor * noise, bounds.lb, bounds.ub)

    def explore(self, bounds, exploration_factor, learning_rate):
        base_exploration = np.random.uniform(bounds.lb, bounds.ub, (self.inner_population_size, self.dim))
        additional_exploration = learning_rate * np.random.normal(0, exploration_factor, (self.inner_population_size, self.dim))
        return base_exploration + additional_exploration

    def dynamically_adjust_population(self, population_size, dynamic_ratio):
        return max(int(population_size * (1 + dynamic_ratio * (self.evaluations / self.budget))), 1)

    def refine_local_search_space(self, best_solution, bounds, radius=0.1):
        lower_bound = np.maximum(bounds.lb, best_solution - radius * (bounds.ub - bounds.lb))
        upper_bound = np.minimum(bounds.ub, best_solution + radius * (bounds.ub - bounds.lb))
        return np.random.uniform(lower_bound, upper_bound, (self.inner_population_size, self.dim))

    def hybrid_search(self, best_solution, bounds, exploration_factor, exploitation_factor, learning_rate):
        exploration_candidates = self.explore(bounds, exploration_factor, learning_rate)
        exploitation_candidates = self.exploit(best_solution, bounds, exploitation_factor, learning_rate)
        combined_candidates = np.vstack((exploration_candidates, exploitation_candidates))
        return combined_candidates

    def context_aware_transition(self, populations, fitnesses):
        for i in range(self.populations):
            if self.detect_phase_transition(populations[i]):
                exploitation_factor = self.exploitation_factors[i] * 1.5
                self.exploitation_factors[i] = min(exploitation_factor, 1.0)
                self.dynamic_ratios[i] = min(self.dynamic_ratios[i] * 1.1, 0.2)
            else:
                exploration_factor = self.exploration_factors[i] * 0.9
                self.exploration_factors[i] = max(exploration_factor, 0.5)
                self.dynamic_ratios[i] = max(self.dynamic_ratios[i] * 0.9, 0.01)

    def __call__(self, func):
        bounds = func.bounds
        populations = self.initialize_populations(bounds)
        fitnesses = [self.evaluate_population(population, func) for population in populations]
        population_sizes = [self.base_population_size for _ in range(self.populations)]

        while self.evaluations < self.budget:
            self.context_aware_transition(populations, fitnesses)
            new_populations, new_fitnesses = [], []

            for i in range(self.populations):
                population_sizes[i] = self.dynamically_adjust_population(population_sizes[i], self.dynamic_ratios[i])
                best_idx = np.argmin(fitnesses[i])
                best_solution = populations[i][best_idx]

                if np.random.rand() < 0.6:
                    new_population = self.hybrid_search(best_solution, bounds, self.exploration_factors[i], self.exploitation_factors[i], self.learning_rates[i])
                else:
                    new_population = self.refine_local_search_space(best_solution, bounds)

                new_fitness = self.evaluate_population(new_population, func)

                combined_population = np.vstack((populations[i], new_population))
                combined_fitness = np.concatenate((fitnesses[i], new_fitness))
                best_indices = np.argsort(combined_fitness)[:population_sizes[i]]
                new_populations.append(combined_population[best_indices])
                new_fitnesses.append(combined_fitness[best_indices])

            populations, fitnesses = new_populations, new_fitnesses

        best_idx = np.argmin([np.min(fitness) for fitness in fitnesses])
        return populations[best_idx][np.argmin(fitnesses[best_idx])]