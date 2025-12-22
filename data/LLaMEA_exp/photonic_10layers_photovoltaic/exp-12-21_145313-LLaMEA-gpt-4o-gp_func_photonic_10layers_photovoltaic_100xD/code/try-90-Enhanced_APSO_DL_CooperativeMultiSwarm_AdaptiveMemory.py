import numpy as np

class Enhanced_APSO_DL_CooperativeMultiSwarm_AdaptiveMemory:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // 10)
        self.num_swarms = 3
        self.swarms = [self.init_swarm() for _ in range(self.num_swarms)]
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0
        self.dynamic_subpop_threshold = budget // (10 * self.num_swarms)
        self.memory = {
            'global_best_positions': np.zeros((self.num_swarms, self.dim)), 
            'global_best_values': np.full(self.num_swarms, np.inf)
        }

    def init_swarm(self):
        return {
            'position': np.random.rand(self.num_particles, self.dim),
            'velocity': np.random.rand(self.num_particles, self.dim) * 0.1,
            'best_personal_position': np.random.rand(self.num_particles, self.dim),
            'best_personal_value': np.full(self.num_particles, np.inf),
            'inertia_weight': 0.9,
            'cognitive_coeff': 2.0,
            'social_coeff': 2.0,
            'mutation_prob': 0.1
        }

    def __call__(self, func):
        while self.evaluations < self.budget:
            for idx, swarm in enumerate(self.swarms):
                self.update_swarm(swarm, func, idx)
                if self.evaluations % self.dynamic_subpop_threshold == 0:
                    self.restructure_subpopulations(swarm)
                self.migrate_swarms()
        
        return self.best_global_position, self.best_global_value

    def update_swarm(self, swarm, func, swarm_idx):
        for i in range(self.num_particles):
            if self.evaluations < self.budget:
                value = func(swarm['position'][i])
                self.evaluations += 1

                if value < swarm['best_personal_value'][i]:
                    swarm['best_personal_value'][i] = value
                    swarm['best_personal_position'][i] = swarm['position'][i]
                
                if value < self.memory['global_best_values'][swarm_idx]:
                    self.memory['global_best_values'][swarm_idx] = value
                    self.memory['global_best_positions'][swarm_idx] = swarm['position'][i]

                if value < self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = swarm['position'][i]

        # Update dynamic coefficients
        swarm['inertia_weight'] = 0.4 + 0.5 * (1 - self.evaluations / self.budget)
        swarm['cognitive_coeff'] = 1.5 + np.random.rand() * 1.5
        swarm['social_coeff'] = 1.5 + np.random.rand() * 1.5
        swarm['mutation_prob'] = 0.05 + 0.45 * (1 - self.evaluations / self.budget)

        # Velocity and position update
        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        swarm['velocity'] = (swarm['inertia_weight'] * swarm['velocity'] +
                             swarm['cognitive_coeff'] * r1 * (swarm['best_personal_position'] - swarm['position']) +
                             swarm['social_coeff'] * r2 * (self.best_global_position - swarm['position']))

        for i in range(self.num_particles):
            if np.random.rand() < swarm['mutation_prob']:
                indices = np.random.choice(self.num_particles, 3, replace=False)
                F = np.random.rand() * (0.5 + 0.5 * (1 - self.evaluations / self.budget))
                mutant = swarm['position'][indices[0]] + F * (swarm['position'][indices[1]] - swarm['position'][indices[2]])
                swarm['position'][i] = np.clip(mutant, func.bounds.lb, func.bounds.ub)

        swarm['position'] += swarm['velocity']
        swarm['position'] = np.clip(swarm['position'], func.bounds.lb, func.bounds.ub)

    def restructure_subpopulations(self, swarm):
        sorted_indices = np.argsort(swarm['best_personal_value'])
        half_point = self.num_particles // 2
        top_half, bottom_half = sorted_indices[:half_point], sorted_indices[half_point:]
        swarm['position'][bottom_half], swarm['velocity'][bottom_half] = swarm['position'][top_half], swarm['velocity'][top_half] * 0.5

    def migrate_swarms(self):
        if self.evaluations % (self.budget // (5 * self.num_swarms)) == 0:
            # Exchange best positions among swarms
            for i in range(self.num_swarms):
                other_swarm = (i + 1) % self.num_swarms
                if self.memory['global_best_values'][other_swarm] < self.memory['global_best_values'][i]:
                    self.memory['global_best_positions'][i], self.memory['global_best_positions'][other_swarm] = \
                    self.memory['global_best_positions'][other_swarm], self.memory['global_best_positions'][i]
                    self.memory['global_best_values'][i], self.memory['global_best_values'][other_swarm] = \
                    self.memory['global_best_values'][other_swarm], self.memory['global_best_values'][i]