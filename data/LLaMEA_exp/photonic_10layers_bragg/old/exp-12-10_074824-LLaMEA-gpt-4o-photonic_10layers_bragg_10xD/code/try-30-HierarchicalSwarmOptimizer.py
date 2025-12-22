import numpy as np

class HierarchicalSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.inertia_weight_start = 0.9
        self.inertia_weight_end = 0.4
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.local_convergence_weight = 0.5
        self.mutation_rate_start = 0.1
        self.mutation_rate_end = 0.01

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        swarm_position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm_velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(swarm_position)
        personal_best_value = np.array([float('inf')] * self.population_size)
        global_best_value = float('inf')
        global_best_position = np.zeros(self.dim)
        
        global_success_rate = 0.5  # Initial global success rate
        local_success_rates = np.ones(self.population_size) * 0.5  # Initial local success rates

        for t in range(self.budget):
            inertia_weight = self.inertia_weight_end + (self.inertia_weight_start - self.inertia_weight_end) * \
                             (1 - t / self.budget)
            mutation_rate = self.mutation_rate_end + (self.mutation_rate_start - self.mutation_rate_end) * \
                            (1 - t / self.budget)

            for i in range(self.population_size):
                fitness_value = func(swarm_position[i])
                if fitness_value < personal_best_value[i]:
                    personal_best_value[i] = fitness_value
                    personal_best_position[i] = swarm_position[i]
                    local_success_rates[i] = 0.9 * local_success_rates[i] + 0.1
                else:
                    local_success_rates[i] *= 0.9

                if fitness_value < global_best_value:
                    global_best_value = fitness_value
                    global_best_position = swarm_position[i]
                    global_success_rate = 0.9 * global_success_rate + 0.1
                else:
                    global_success_rate *= 0.9

            for i in range(self.population_size):
                inertia = inertia_weight * swarm_velocity[i]
                cognitive_component = (self.cognitive_weight * local_success_rates[i]) * np.random.rand(self.dim) * \
                                      (personal_best_position[i] - swarm_position[i])
                social_component = (self.social_weight * global_success_rate) * np.random.rand(self.dim) * \
                                   (global_best_position - swarm_position[i])
                local_convergence = self.local_convergence_weight * np.random.rand(self.dim) * \
                                    (np.mean(swarm_position, axis=0) - swarm_position[i])
                
                swarm_velocity[i] = inertia + cognitive_component + social_component + local_convergence
                swarm_position[i] = np.clip(swarm_position[i] + swarm_velocity[i], lb, ub)
                swarm_position[i] += mutation_rate * np.random.normal(0, 1, self.dim)

        return global_best_position, global_best_value