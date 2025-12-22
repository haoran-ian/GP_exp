import numpy as np

class EnhancedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.reinit_prob = 0.1

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm_position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm_velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(swarm_position)
        personal_best_value = np.array([float('inf')] * self.population_size)
        global_best_value = float('inf')
        global_best_position = np.zeros(self.dim)

        for iteration in range(self.budget):
            inertia_weight = self.inertia_weight_max - (
                (self.inertia_weight_max - self.inertia_weight_min) * (iteration / self.budget)
            )

            for i in range(self.population_size):
                fitness_value = func(swarm_position[i])
                if fitness_value < personal_best_value[i]:
                    personal_best_value[i] = fitness_value
                    personal_best_position[i] = swarm_position[i]
                if fitness_value < global_best_value:
                    global_best_value = fitness_value
                    global_best_position = swarm_position[i]

            for i in range(self.population_size):
                if np.random.rand() < self.reinit_prob:
                    swarm_position[i] = np.random.uniform(lb, ub, self.dim)
                    swarm_velocity[i] = np.random.uniform(-1, 1, self.dim)
                else:
                    inertia = inertia_weight * swarm_velocity[i]
                    cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (personal_best_position[i] - swarm_position[i])
                    social_component = self.social_weight * np.random.rand(self.dim) * (global_best_position - swarm_position[i])
                    swarm_velocity[i] = inertia + cognitive_component + social_component
                    swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], lb, ub)

        return global_best_position, global_best_value