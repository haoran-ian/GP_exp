import numpy as np

class AdaptiveInertiaSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.diversity_preservation_weight = 0.1

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm_position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm_velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(swarm_position)
        personal_best_value = np.array([float('inf')] * self.population_size)
        global_best_value = float('inf')
        global_best_position = np.zeros(self.dim)

        for t in range(self.budget):
            for i in range(self.population_size):
                fitness_value = func(swarm_position[i])
                if fitness_value < personal_best_value[i]:
                    personal_best_value[i] = fitness_value
                    personal_best_position[i] = swarm_position[i]
                if fitness_value < global_best_value:
                    global_best_value = fitness_value
                    global_best_position = swarm_position[i]

            inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (t / self.budget)

            for i in range(self.population_size):
                neighbors = np.random.choice(self.population_size, 2, replace=False)
                local_best_position = personal_best_position[neighbors[np.argmin(personal_best_value[neighbors])]]

                inertia = inertia_weight * swarm_velocity[i]
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (personal_best_position[i] - swarm_position[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (local_best_position - swarm_position[i])
                diversity_component = self.diversity_preservation_weight * np.random.rand(self.dim) * (np.mean(swarm_position, axis=0) - swarm_position[i])

                swarm_velocity[i] = inertia + cognitive_component + social_component + diversity_component
                swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], lb, ub)

                # Minor random perturbation to maintain diversity
                swarm_position[i] += 0.01 * np.random.normal(0, 1, self.dim)

        return global_best_position, global_best_value