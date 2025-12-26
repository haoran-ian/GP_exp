import numpy as np

class RefinedAdaptiveChaosSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_param_base = 1.5
        self.social_param_base = 1.5
        self.local_search_rate = 0.3
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
            chaos_parameter = 3.95 * chaos_parameter * (1 - chaos_parameter)
            phase_ratio = evals / self.budget
            inertia_weight = (1 - phase_ratio) * self.initial_inertia_weight + phase_ratio * self.final_inertia_weight
            cognitive_param = self.cognitive_param_base * (1 + 0.5 * np.sin(phase_ratio * np.pi))
            social_param = self.social_param_base * (1 - 0.5 * np.cos(phase_ratio * np.pi))

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (
                inertia_weight * velocities +
                cognitive_param * r1 * (local_best_positions - population) +
                social_param * r2 * (self.best_global_position - population)
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
                dynamic_radius = (ub - lb) * (1 - phase_ratio) * 0.1
                perturbation = np.random.uniform(-dynamic_radius, dynamic_radius, self.dim)
                candidate = self.best_global_position + perturbation
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evals += 1

                if candidate_value < self.best_global_value:
                    self.best_global_value = candidate_value
                    self.best_global_position = candidate

            if evals < self.budget * 0.6 and np.random.rand() < 0.2:
                levy_flight = np.random.standard_cauchy(size=self.dim) * (ub - lb) * 0.005
                candidate = self.best_global_position + levy_flight
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evals += 1

                if candidate_value < self.best_global_value:
                    self.best_global_value = candidate_value
                    self.best_global_position = candidate

        return self.best_global_position