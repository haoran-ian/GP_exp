import numpy as np

class AdaptiveDynamicSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.local_search_rate = 0.3  # Further increased local search rate
        self.memory_reset_interval = 0.1 * budget  # New parameter for memory reset
        self.best_global_position = None
        self.best_global_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        local_best_positions = population.copy()
        local_best_values = np.array([func(ind) for ind in population])
        self.best_global_value = np.min(local_best_values)
        self.best_global_position = population[np.argmin(local_best_values)].copy()
        evals = self.population_size

        chaos_parameter = np.random.rand()

        while evals < self.budget:
            chaos_parameter = 3.98 * chaos_parameter * (1 - chaos_parameter)  # Further tuned chaotic map parameter
            inertia_weight = self.final_inertia_weight + (self.initial_inertia_weight - self.final_inertia_weight) * chaos_parameter

            # Updated line: Dynamic adjustment of cognitive and social parameters based on chaos
            self.cognitive_param = 1.5 * chaos_parameter; self.social_param = 2.5 - self.cognitive_param

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (
                inertia_weight * velocities +
                self.cognitive_param * r1 * (local_best_positions - population) +
                self.social_param * r2 * (self.best_global_position - population)
            )
            population += velocities
            population = np.clip(population, lb, ub)

            values = np.array([func(ind) for ind in population])
            evals += self.population_size

            better_mask = values < local_best_values
            local_best_positions[better_mask] = population[better_mask]
            local_best_values[better_mask] = values[better_mask]

            if np.min(values) < self.best_global_value:
                self.best_global_value = np.min(values)
                self.best_global_position = population[np.argmin(values)].copy()

            # Strategic memory reset to escape local optima
            if evals % self.memory_reset_interval < self.population_size:
                random_indices = np.random.choice(self.population_size, size=int(0.1 * self.population_size), replace=False)
                population[random_indices] = np.random.uniform(lb, ub, (random_indices.size, self.dim))
                local_best_positions[random_indices] = population[random_indices]
                local_best_values[random_indices] = np.array([func(ind) for ind in population[random_indices]])
                evals += random_indices.size

            if np.random.rand() < self.local_search_rate:
                dynamic_radius = (ub - lb) * (1 - evals / self.budget) * 0.1  # Further adjusted dynamic radius factor
                perturbation = np.random.uniform(-1, 1, self.dim) * dynamic_radius
                candidate = self.best_global_position + perturbation
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evals += 1

                if candidate_value < self.best_global_value:
                    self.best_global_value = candidate_value
                    self.best_global_position = candidate

        return self.best_global_position