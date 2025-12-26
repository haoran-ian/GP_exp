import numpy as np

class EnhancedAdaptiveSwarmHybridization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.local_search_rate = 0.1
        self.levy_alpha = 1.5
        self.best_global_position = None
        self.best_global_value = np.inf

    def levy_flight(self, step_size):
        u = np.random.normal(0, 1, self.dim) * step_size
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1/self.levy_alpha)
        return step

    def opposition_based_learning(self, population, lb, ub):
        return lb + ub - population

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
            chaos_parameter = 3.9 * chaos_parameter * (1 - chaos_parameter)
            inertia_weight = self.final_inertia_weight + (self.initial_inertia_weight - self.final_inertia_weight) * chaos_parameter

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

            if np.random.rand() < self.local_search_rate:
                dynamic_radius = (ub - lb) * (1 - evals / self.budget) * 0.1
                candidate = self.best_global_position + np.random.uniform(-dynamic_radius, dynamic_radius, self.dim)
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evals += 1

                if candidate_value < self.best_global_value:
                    self.best_global_value = candidate_value
                    self.best_global_position = candidate

            # Lévy flight for enhanced exploration
            for i in range(self.population_size):
                step_size = 0.01 * (ub - lb)
                levy_step = self.levy_flight(step_size)
                new_position = population[i] + levy_step
                new_position = np.clip(new_position, lb, ub)
                new_value = func(new_position)
                evals += 1

                if new_value < local_best_values[i]:
                    local_best_positions[i] = new_position
                    local_best_values[i] = new_value

                    if new_value < self.best_global_value:
                        self.best_global_value = new_value
                        self.best_global_position = new_position

            # Opposition-based learning
            if np.random.rand() < 0.1:
                opposite_population = self.opposition_based_learning(population, lb, ub)
                opposite_values = np.array([func(ind) for ind in opposite_population])
                evals += self.population_size
                combined = np.concatenate((population, opposite_population))
                combined_values = np.concatenate((values, opposite_values))
                best_indices = np.argsort(combined_values)[:self.population_size]
                population = combined[best_indices]
                local_best_positions = population.copy()
                local_best_values = combined_values[best_indices]

        return self.best_global_position