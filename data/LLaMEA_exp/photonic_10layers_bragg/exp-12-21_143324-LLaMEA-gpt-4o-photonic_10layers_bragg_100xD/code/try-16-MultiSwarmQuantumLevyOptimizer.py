import numpy as np

class MultiSwarmQuantumLevyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.swarms = [self.initialize_swarm() for _ in range(self.num_swarms)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        
    def initialize_swarm(self):
        return {
            'positions': None,
            'velocities': None,
            'best_positions': None,
            'best_scores': None,
            'inertia_weight': 0.9,
            'cognitive_const': 1.4,
            'social_const': 1.6,
        }
        
    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 
                  2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        center = (lb + ub) / 2
        quantum_range = (ub - lb) / 2
        for swarm in self.swarms:
            swarm['positions'] = np.random.uniform(center - quantum_range, center + quantum_range, (self.population_size, self.dim))
            swarm['velocities'] = np.random.uniform(-quantum_range, quantum_range, (self.population_size, self.dim))
            swarm['best_positions'] = np.copy(swarm['positions'])
            swarm['best_scores'] = np.full(self.population_size, float('inf'))
        self.global_best_position = np.random.uniform(lb, ub, self.dim)

    def update_positions_and_velocities(self, bounds, progress_factor):
        lb, ub = bounds.lb, bounds.ub
        for swarm in self.swarms:
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = swarm['cognitive_const'] * r1 * (swarm['best_positions'][i] - swarm['positions'][i])
                social_velocity = swarm['social_const'] * r2 * (self.global_best_position - swarm['positions'][i])
                levy_vel = self.levy_flight(self.dim)
                quantum_weight = 0.01 * (1 - progress_factor)  # More influence early on
                swarm['velocities'][i] = (swarm['inertia_weight'] * swarm['velocities'][i] +
                                          cognitive_velocity + social_velocity +
                                          quantum_weight * levy_vel)
                swarm['positions'][i] = np.clip(swarm['positions'][i] + swarm['velocities'][i], lb, ub)
            swarm['inertia_weight'] = max(0.4, swarm['inertia_weight'] - (0.45 / self.budget) * self.population_size)
            swarm['cognitive_const'] = 1.0 + (0.4 * (1 - progress_factor))
            swarm['social_const'] = 2.0 - (0.4 * (1 - progress_factor))

    def evaluate_and_update_best(self, func):
        for swarm in self.swarms:
            for i in range(self.population_size):
                score = func(swarm['positions'][i])
                if score < swarm['best_scores'][i]:
                    swarm['best_scores'][i] = score
                    swarm['best_positions'][i] = swarm['positions'][i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = swarm['positions'][i].copy()

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        evaluations = 0

        while evaluations < self.budget:
            progress_factor = evaluations / self.budget
            self.update_positions_and_velocities(bounds, progress_factor)
            self.evaluate_and_update_best(func)
            evaluations += self.population_size * self.num_swarms

        return self.global_best_position