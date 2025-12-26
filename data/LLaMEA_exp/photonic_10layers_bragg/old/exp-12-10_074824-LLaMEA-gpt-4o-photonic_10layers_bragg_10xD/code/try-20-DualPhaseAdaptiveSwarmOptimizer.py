import numpy as np

class DualPhaseAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.inertia_weight = 0.9
        self.inertia_damping = 0.99
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.exploration_weight = 0.7
        self.exploitation_weight = 0.3
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
            entropy = np.mean(np.std(swarm_position, axis=0) / (ub - lb))
            adaptive_mutation_rate = self.initial_mutation_rate * (1 - t / self.budget) + self.final_mutation_rate * (t / self.budget)

            for i in range(self.population_size):
                fitness_value = func(swarm_position[i])
                if fitness_value < personal_best_value[i]:
                    personal_best_value[i] = fitness_value
                    personal_best_position[i] = swarm_position[i]
                if fitness_value < global_best_value:
                    global_best_value = fitness_value
                    global_best_position = swarm_position[i]

            phase_control = (1 - t / self.budget)
            exploration_phase = phase_control > 0.5

            for i in range(self.population_size):
                inertia = (0.4 + (self.inertia_weight - 0.4) * (1 - t / self.budget)) * swarm_velocity[i]
                if exploration_phase:
                    weighted_cognitive = self.cognitive_weight * self.exploration_weight * np.random.rand(self.dim)
                    weighted_social = self.social_weight * self.exploration_weight * np.random.rand(self.dim)
                else:
                    weighted_cognitive = self.cognitive_weight * self.exploitation_weight * np.random.rand(self.dim)
                    weighted_social = self.social_weight * self.exploitation_weight * np.random.rand(self.dim)

                cognitive_component = weighted_cognitive * (personal_best_position[i] - swarm_position[i])
                social_component = weighted_social * (global_best_position - swarm_position[i])
                local_convergence = (self.local_convergence_weight * (1 + t / self.budget)) * np.random.rand(self.dim) * (np.mean(swarm_position, axis=0) - swarm_position[i])
                
                swarm_velocity[i] = inertia + cognitive_component + social_component + local_convergence
                swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], lb, ub)

                swarm_position[i] += adaptive_mutation_rate * np.random.uniform(-1, 1, self.dim)
                personal_best_position[i] += 0.01 * np.random.rand(self.dim) * (global_best_position - personal_best_position[i]) 

            self.inertia_weight *= self.inertia_damping

        return global_best_position, global_best_value