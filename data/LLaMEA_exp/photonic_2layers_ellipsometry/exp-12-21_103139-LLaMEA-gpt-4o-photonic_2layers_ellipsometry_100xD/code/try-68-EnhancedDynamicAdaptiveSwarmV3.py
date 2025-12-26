import numpy as np

class EnhancedDynamicAdaptiveSwarmV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_param_base = 1.5
        self.social_param_base = 1.5
        self.local_search_rate = 0.2
        self.best_global_position = None
        self.best_global_value = np.inf
        self.epsilon = 1e-8

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
            chaos_parameter = 3.95 * chaos_parameter * (1 - chaos_parameter)
            inertia_weight = self.final_inertia_weight + (self.initial_inertia_weight - self.final_inertia_weight) * chaos_parameter

            # Dynamic balancing of cognitive and social parameters
            cognitive_param = self.cognitive_param_base * (1 + 0.5 * np.sin(evals * np.pi / self.budget))
            social_param = self.social_param_base * (1 - 0.5 * np.cos(evals * np.pi / self.budget))

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (
                inertia_weight * velocities +
                cognitive_param * r1 * (local_best_positions - population) +
                social_param * r2 * (self.best_global_position - population)
            )
            
            # Enhanced dynamic diversity preservation
            diversity = np.mean(np.std(population, axis=0))
            velocity_adjustment_factor = (1 + self.epsilon / (diversity + self.epsilon) ** 0.5)
            velocities *= velocity_adjustment_factor

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

            # Local search enhancement with adaptive perturbation
            if np.random.rand() < self.local_search_rate:
                dynamic_radius = (ub - lb) * (1 - evals / self.budget) * 0.1
                perturbation = np.random.uniform(-1, 1, self.dim) * dynamic_radius
                candidate = self.best_global_position + perturbation
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evals += 1

                if candidate_value < self.best_global_value:
                    self.best_global_value = candidate_value
                    self.best_global_position = candidate

            # Employ adaptive Lévy flights for escaping local minima
            if np.random.rand() < 0.15:
                levy_flight = np.random.standard_cauchy(size=self.dim) * (ub - lb) * 0.01 * velocity_adjustment_factor
                candidate = self.best_global_position + levy_flight
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evals += 1

                if candidate_value < self.best_global_value:
                    self.best_global_value = candidate_value
                    self.best_global_position = candidate

        return self.best_global_position