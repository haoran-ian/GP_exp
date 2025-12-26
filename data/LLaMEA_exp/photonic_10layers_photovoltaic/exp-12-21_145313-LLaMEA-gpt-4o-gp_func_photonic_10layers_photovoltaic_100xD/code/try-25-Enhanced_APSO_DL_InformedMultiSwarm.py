import numpy as np

class Enhanced_APSO_DL_InformedMultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // 10)
        self.num_swarms = 2
        self.swarms = [self.init_swarm() for _ in range(self.num_swarms)]
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0

    def init_swarm(self):
        return {
            'position': np.random.rand(self.num_particles, self.dim),
            'velocity': np.random.rand(self.num_particles, self.dim) * 0.1,
            'best_personal_position': np.random.rand(self.num_particles, self.dim),
            'best_personal_value': np.full(self.num_particles, np.inf),
            'inertia_weight': 0.9,
            'cognitive_coeff': 2.0,
            'social_coeff': 2.0,
            'mutation_prob': 0.1,
            'historical_best_position': None,
            'historical_best_value': np.inf
        }

    def __call__(self, func):
        while self.evaluations < self.budget:
            for swarm in self.swarms:
                self.update_swarm(swarm, func)
                self.informed_migrate_swarms()
        
        return self.best_global_position, self.best_global_value

    def update_swarm(self, swarm, func):
        for i in range(self.num_particles):
            if self.evaluations < self.budget:
                # Evaluate current position
                value = func(swarm['position'][i])
                self.evaluations += 1

                # Update personal and global bests
                if value < swarm['best_personal_value'][i]:
                    swarm['best_personal_value'][i] = value
                    swarm['best_personal_position'][i] = swarm['position'][i]
                if value < self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = swarm['position'][i]

                # Update historical best
                if value < swarm['historical_best_value']:
                    swarm['historical_best_value'] = value
                    swarm['historical_best_position'] = swarm['position'][i]

        # Dynamic adjustment of inertia weight and learning rates
        swarm['inertia_weight'] = 0.4 + 0.5 * (1 - self.evaluations / self.budget)
        swarm['cognitive_coeff'] = 1.5 + np.random.rand() * 1.5
        swarm['social_coeff'] = 1.5 + np.random.rand() * 1.5
        swarm['mutation_prob'] = 0.05 + 0.45 * (1 - self.evaluations / self.budget)

        # Update velocity and position
        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        swarm['velocity'] = (swarm['inertia_weight'] * swarm['velocity'] +
                             swarm['cognitive_coeff'] * r1 * (swarm['best_personal_position'] - swarm['position']) +
                             swarm['social_coeff'] * r2 * (self.best_global_position - swarm['position']))

        # Apply differential mutation
        for i in range(self.num_particles):
            if np.random.rand() < swarm['mutation_prob']:
                indices = np.random.choice(self.num_particles, 3, replace=False)
                F = np.random.rand()
                mutant = swarm['position'][indices[0]] + F * (swarm['position'][indices[1]] - swarm['position'][indices[2]])
                swarm['position'][i] = np.clip(mutant, func.bounds.lb, func.bounds.ub)

        swarm['position'] += swarm['velocity']
        swarm['position'] = np.clip(swarm['position'], func.bounds.lb, func.bounds.ub)

    def informed_migrate_swarms(self):
        # Migrate particles between swarms based on historical best positions
        if self.evaluations % (self.budget // (5 * self.num_swarms)) == 0:
            for i in range(self.num_swarms - 1):
                idx = np.random.choice(self.num_particles, self.num_particles // 2, replace=False)
                current_swarm_best = self.swarms[i]['historical_best_position']
                next_swarm_best = self.swarms[i+1]['historical_best_position']
                if current_swarm_best is not None and next_swarm_best is not None:
                    self.swarms[i]['position'][idx] = next_swarm_best
                    self.swarms[i+1]['position'][idx] = current_swarm_best