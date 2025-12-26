import numpy as np

class Enhanced_APSO_DL_AdaptiveMultiSwarm_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, budget // 8)
        self.num_swarms = 3
        self.swarms = [self.init_swarm() for _ in range(self.num_swarms)]
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0
        self.dynamic_subpop_threshold = budget // (12 * self.num_swarms)

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
            for swarm in self.swarms:
                self.update_swarm(swarm, func)
                if self.evaluations % self.dynamic_subpop_threshold == 0:
                    self.enhanced_restructure_subpopulations(swarm)
                self.randomized_migrate_swarms()
        
        return self.best_global_position, self.best_global_value

    def update_swarm(self, swarm, func):
        for i in range(self.num_particles):
            if self.evaluations < self.budget:
                value = func(swarm['position'][i])
                self.evaluations += 1

                if value < swarm['best_personal_value'][i]:
                    swarm['best_personal_value'][i] = value
                    swarm['best_personal_position'][i] = swarm['position'][i]
                if value < self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = swarm['position'][i]

        swarm['inertia_weight'] = 0.4 + 0.5 * (1 - self.evaluations / self.budget)
        swarm['cognitive_coeff'] = 1.5 + np.random.rand() * 1.5
        swarm['social_coeff'] = 1.8 + np.random.rand() * 1.2
        swarm['mutation_prob'] = 0.05 + 0.45 * (1 - self.evaluations / self.budget)

        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        swarm['velocity'] = (swarm['inertia_weight'] * swarm['velocity'] +
                             swarm['cognitive_coeff'] * r1 * (swarm['best_personal_position'] - swarm['position']) +
                             swarm['social_coeff'] * r2 * (self.best_global_position - swarm['position']))

        for i in range(self.num_particles):
            if np.random.rand() < swarm['mutation_prob']:
                indices = np.random.choice(self.num_particles, 4, replace=False)
                F = 0.5 + np.random.rand() * (1 - self.evaluations / self.budget)
                mutant = swarm['position'][indices[0]] + F * (swarm['position'][indices[1]] - swarm['position'][indices[2]]) + \
                         np.random.rand() * (swarm['position'][indices[3]] - swarm['position'][i])
                swarm['position'][i] = np.clip(mutant, func.bounds.lb, func.bounds.ub)

        swarm['position'] += swarm['velocity']
        swarm['position'] = np.clip(swarm['position'], func.bounds.lb, func.bounds.ub)

    def enhanced_restructure_subpopulations(self, swarm):
        sorted_indices = np.argsort(swarm['best_personal_value'])
        for i in range(self.num_particles // 2, self.num_particles):
            if np.random.rand() < 0.5:
                idx = sorted_indices[i]
                swarm['position'][idx] = swarm['position'][sorted_indices[np.random.randint(0, self.num_particles // 2)]]
                swarm['velocity'][idx] *= 0.3

    def randomized_migrate_swarms(self):
        if self.evaluations % (self.budget // (6 * self.num_swarms)) == 0:
            for i in range(self.num_swarms - 1):
                idx = np.random.choice(self.num_particles, self.num_particles // 3, replace=False)
                self.swarms[i]['position'][idx], self.swarms[i+1]['position'][idx] = \
                    self.swarms[i+1]['position'][idx], self.swarms[i]['position'][idx]