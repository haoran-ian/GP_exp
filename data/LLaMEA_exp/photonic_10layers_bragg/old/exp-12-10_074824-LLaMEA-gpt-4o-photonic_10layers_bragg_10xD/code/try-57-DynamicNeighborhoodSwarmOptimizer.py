import numpy as np

class DynamicNeighborhoodSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.inertia_weight = 0.8  # Reduced initial inertia for better exploration
        self.inertia_damping = 0.99  # Slightly increased damping for gradual reduction
        self.cognitive_weight = 1.7  # Enhanced cognitive coefficient
        self.social_weight = 1.3  # Reduced social influence for diverse explorations
        self.local_convergence_weight = 0.5  # Reduce local convergence to avoid premature convergence
        self.initial_mutation_rate = 0.15  # Increase initial mutation for better diversity
        self.final_mutation_rate = 0.02  # Slightly increase final mutation for sustained exploration

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm_position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm_velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(swarm_position)
        personal_best_value = np.array([float('inf')] * self.population_size)
        global_best_value = float('inf')
        global_best_position = np.zeros(self.dim)

        for t in range(self.budget):
            entropy = np.mean(np.std(swarm_position, axis=0) / (ub - lb))
            adaptive_mutation_rate = self.initial_mutation_rate * (1 - t / self.budget) + self.final_mutation_rate * (t / self.budget) ** 1.2

            for i in range(self.population_size):
                fitness_value = func(swarm_position[i])
                if fitness_value < personal_best_value[i]:
                    personal_best_value[i] = fitness_value
                    personal_best_position[i] = swarm_position[i]
                if fitness_value < global_best_value:
                    global_best_value = fitness_value
                    global_best_position = swarm_position[i]

            neighborhood_size = max(2, int(self.population_size * (1 - t / self.budget)))
            for i in range(self.population_size):
                neighbors = np.random.choice(self.population_size, neighborhood_size, replace=False)
                local_best_value = float('inf')
                local_best_position = np.zeros(self.dim)
                for neighbor in neighbors:
                    if personal_best_value[neighbor] < local_best_value:
                        local_best_value = personal_best_value[neighbor]
                        local_best_position = personal_best_position[neighbor]

                inertia = (0.4 + (self.inertia_weight - 0.4) * (1 - t / self.budget)) * swarm_velocity[i]
                cognitive_weight = self.cognitive_weight * np.random.rand() * (1 - (1 - fitness_value/global_best_value) * t / self.budget)
                social_weight = self.social_weight * ((1 + (fitness_value/global_best_value)) * t / self.budget) * (1 + entropy)
                cognitive_component = (cognitive_weight + 0.5 * (1 - fitness_value / global_best_value)) * np.random.rand(self.dim) * (personal_best_position[i] - swarm_position[i])
                social_component = social_weight * np.random.rand(self.dim) * (local_best_position - swarm_position[i])
                local_convergence = (self.local_convergence_weight * (1 + t / self.budget)) * np.random.rand(self.dim) * (np.mean(swarm_position, axis=0) - swarm_position[i])
                swarm_velocity[i] = inertia + cognitive_component + social_component + local_convergence
                swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], lb, ub)

                swarm_position[i] += adaptive_mutation_rate * np.random.normal(0, 1, self.dim)
                personal_best_position[i] += 0.01 * np.random.rand(self.dim) * (global_best_position - personal_best_position[i])

            self.inertia_weight *= self.inertia_damping

        return global_best_position, global_best_value