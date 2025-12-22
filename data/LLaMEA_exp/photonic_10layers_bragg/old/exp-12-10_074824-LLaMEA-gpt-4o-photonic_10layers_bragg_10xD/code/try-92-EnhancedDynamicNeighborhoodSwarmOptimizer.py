import numpy as np

class EnhancedDynamicNeighborhoodSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.inertia_weight = 0.9
        self.inertia_damping = 0.995
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.local_convergence_weight = 0.6
        self.diversity_weight = 0.3
        self.initial_mutation_rate = 0.1
        self.final_mutation_rate = 0.01

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm_position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm_velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(swarm_position)
        personal_best_value = np.array([float('inf')] * self.population_size)
        global_best_value = float('inf')
        global_best_position = np.zeros(self.dim)

        for t in range(self.budget):
            diversity = np.linalg.norm(np.std(swarm_position, axis=0)) / np.linalg.norm(ub - lb)
            adaptive_mutation_rate = self.initial_mutation_rate * (1 - (t / self.budget) ** 1.1 + diversity) + self.final_mutation_rate * (t / self.budget) ** 1.2
            
            # Changed line
            adaptive_mutation_rate *= (1 - t / self.budget)  # Mutation decay function

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

                inertia = (0.4 + (self.inertia_weight - 0.4) * (1 - t / self.budget) ** 2) * swarm_velocity[i]
                cognitive_weight = self.cognitive_weight * np.random.rand() * (1 - (1 - fitness_value/global_best_value) * t / self.budget)
                social_weight = self.social_weight * ((1 + (fitness_value/global_best_value)) * t / self.budget) * (1 + diversity * self.diversity_weight)
                cognitive_component = (cognitive_weight + 0.5 * (1 - fitness_value / global_best_value)) * np.random.rand(self.dim) * (personal_best_position[i] - swarm_position[i])
                social_component = social_weight * np.random.rand(self.dim) * (local_best_position - swarm_position[i])
                local_convergence = (self.local_convergence_weight * (1 + t / self.budget)) * np.random.rand(self.dim) * (np.mean(swarm_position, axis=0) - swarm_position[i])
                swarm_velocity[i] = inertia + cognitive_component + social_component + local_convergence
                swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], lb, ub)

                swarm_position[i] += adaptive_mutation_rate * np.random.normal(0, 1, self.dim)
                personal_best_position[i] += 0.01 * np.random.rand(self.dim) * (global_best_position - personal_best_position[i])

            self.inertia_weight *= self.inertia_damping * (1 - t / self.budget)

        return global_best_position, global_best_value