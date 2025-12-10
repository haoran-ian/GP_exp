import numpy as np

class EnhancedAdaptiveLearningSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.inertia_weight = 0.9
        self.inertia_damping = 0.99
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.local_convergence_weight = 0.5
        self.mutation_rate = 0.1
        self.diversity_weight = 0.1
        self.leaders_fraction = 0.2

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

            # Dynamic leader selection based on the top fraction
            leader_indices = np.argsort(personal_best_value)[:int(self.leaders_fraction * self.population_size)]
            dynamic_leader_position = np.mean(personal_best_position[leader_indices], axis=0)

            for i in range(self.population_size):
                inertia = (0.4 + (self.inertia_weight - 0.4) * (1 - t / self.budget)) * swarm_velocity[i]
                cognitive_weight = self.cognitive_weight * (1 - t / self.budget)
                social_weight = self.social_weight * (t / self.budget)
                cognitive_component = cognitive_weight * np.random.rand(self.dim) * (personal_best_position[i] - swarm_position[i])
                social_component = social_weight * np.random.rand(self.dim) * (dynamic_leader_position - swarm_position[i])
                local_convergence = (self.local_convergence_weight * (1 + t / self.budget)) * np.random.rand(self.dim) * (np.mean(swarm_position, axis=0) - swarm_position[i])
                diversity_component = self.diversity_weight * np.random.rand(self.dim) * (np.std(swarm_position, axis=0))
                swarm_velocity[i] = inertia + cognitive_component + social_component + local_convergence + diversity_component
                swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], lb, ub)
                swarm_position[i] += (self.mutation_rate * (1 - t / self.budget)) * np.random.uniform(-1, 1, self.dim)

            self.inertia_weight *= self.inertia_damping

        return global_best_position, global_best_value