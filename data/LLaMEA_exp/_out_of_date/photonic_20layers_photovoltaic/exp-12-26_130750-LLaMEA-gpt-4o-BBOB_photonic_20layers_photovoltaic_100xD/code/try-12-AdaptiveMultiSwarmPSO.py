import numpy as np

class AdaptiveMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = np.clip(3, 1, dim // 2)  # number of swarms
        self.population_size = int(np.clip(5 * np.log10(dim), 10, 50))
        self.swarm_velocities = [np.random.rand(self.population_size, dim) * 0.1 for _ in range(self.num_swarms)]
        self.swarm_positions = [np.random.rand(self.population_size, dim) for _ in range(self.num_swarms)]
        self.swarm_best_positions = [np.copy(p) for p in self.swarm_positions]
        self.swarm_best_values = [np.full(self.population_size, np.inf) for _ in range(self.num_swarms)]
        self.global_best_position = None
        self.global_best_value = np.inf
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.mutation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        for positions in self.swarm_positions:
            positions *= (ub - lb)
        evaluations = 0
        
        while evaluations < self.budget:
            for s in range(self.num_swarms):
                for i in range(self.population_size):
                    value = func(self.swarm_positions[s][i])
                    evaluations += 1
                    if value < self.swarm_best_values[s][i]:
                        self.swarm_best_values[s][i] = value
                        self.swarm_best_positions[s][i] = self.swarm_positions[s][i]
                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best_position = self.swarm_positions[s][i]
                    if evaluations >= self.budget:
                        break

            if evaluations >= self.budget:
                break

            for s in range(self.num_swarms):
                r1, r2 = np.random.rand(2)
                for i in range(self.population_size):
                    self.swarm_velocities[s][i] = (
                        self.inertia_weight * self.swarm_velocities[s][i] +
                        self.cognitive_coeff * r1 * (self.swarm_best_positions[s][i] - self.swarm_positions[s][i]) +
                        self.social_coeff * r2 * (self.global_best_position - self.swarm_positions[s][i])
                    )
                    self.swarm_positions[s][i] += self.swarm_velocities[s][i]
                    self.swarm_positions[s][i] = np.clip(self.swarm_positions[s][i], lb, ub)

            self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)
            self.cognitive_coeff = 1.5 + (1.5 - 0.5) * (evaluations / self.budget)
            self.social_coeff = 0.5 + (2.0 - 0.5) * (1 - evaluations / self.budget)
            
            if np.random.rand() < self.mutation_rate:
                for s in range(self.num_swarms):
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    target, donor1, donor2 = self.swarm_positions[s][idxs]
                    mutant = target + 0.8 * (donor1 - donor2)
                    mutant = np.clip(mutant, lb, ub)
                    
                    if evaluations < self.budget:
                        mutant_value = func(mutant)
                        evaluations += 1
                        if mutant_value < self.swarm_best_values[s][idxs[0]]:
                            self.swarm_best_values[s][idxs[0]] = mutant_value
                            self.swarm_best_positions[s][idxs[0]] = mutant

            local_radius = 0.1 * (ub - lb)
            for _ in range(3):
                local_search_position = self.global_best_position + np.random.uniform(-local_radius, local_radius, self.dim)
                local_search_position = np.clip(local_search_position, lb, ub)
                local_search_value = func(local_search_position)
                evaluations += 1
                if local_search_value < self.global_best_value:
                    self.global_best_value = local_search_value
                    self.global_best_position = local_search_position

        return self.global_best_position, self.global_best_value