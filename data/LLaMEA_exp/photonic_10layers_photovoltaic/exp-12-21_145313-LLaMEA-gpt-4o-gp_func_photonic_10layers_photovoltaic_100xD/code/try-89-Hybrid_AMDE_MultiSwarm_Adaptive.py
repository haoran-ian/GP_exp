import numpy as np

class Hybrid_AMDE_MultiSwarm_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, budget // 10)
        self.num_swarms = 3
        self.swarms = [self.init_swarm() for _ in range(self.num_swarms)]
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0
        self.memory = []
        self.dynamic_subpop_threshold = budget // (10 * self.num_swarms)

    def init_swarm(self):
        return {
            'position': np.random.rand(self.num_particles, self.dim),
            'velocity': np.random.rand(self.num_particles, self.dim) * 0.1,
            'best_personal_position': np.random.rand(self.num_particles, self.dim),
            'best_personal_value': np.full(self.num_particles, np.inf),
            'memory': np.random.rand(self.num_particles // 2, self.dim),
            'mutation_prob': 0.1
        }

    def __call__(self, func):
        while self.evaluations < self.budget:
            for swarm in self.swarms:
                self.update_swarm(swarm, func)
                if self.evaluations % self.dynamic_subpop_threshold == 0:
                    self.restructure_subpopulations(swarm)
                self.migrate_swarms()
        
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

        swarm['mutation_prob'] = 0.05 + 0.45 * (1 - self.evaluations / self.budget)

        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        F = 0.5 + 0.5 * np.random.rand()
        for i in range(self.num_particles):
            if np.random.rand() < swarm['mutation_prob']:
                indices = np.random.choice(self.num_particles, 3, replace=False)
                mutant = swarm['position'][indices[0]] + F * (swarm['position'][indices[1]] - swarm['position'][indices[2]])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                trial = np.where(np.random.rand(self.dim) < 0.5, mutant, swarm['position'][i])
                if func(trial) < func(swarm['position'][i]):
                    swarm['position'][i] = trial

        swarm['velocity'] += r1 * (swarm['best_personal_position'] - swarm['position']) + r2 * (self.best_global_position - swarm['position'])
        swarm['position'] += swarm['velocity']
        swarm['position'] = np.clip(swarm['position'], func.bounds.lb, func.bounds.ub)

    def restructure_subpopulations(self, swarm):
        sorted_indices = np.argsort(swarm['best_personal_value'])
        half_point = self.num_particles // 2
        top_half, bottom_half = sorted_indices[:half_point], sorted_indices[half_point:]
        swarm['position'][bottom_half], swarm['velocity'][bottom_half] = swarm['position'][top_half], swarm['velocity'][top_half] * 0.5

    def migrate_swarms(self):
        if self.evaluations % (self.budget // (5 * self.num_swarms)) == 0:
            for i in range(self.num_swarms - 1):
                idx = np.random.choice(self.num_particles, self.num_particles // 2, replace=False)
                self.swarms[i]['position'][idx], self.swarms[i+1]['position'][idx] = \
                    self.swarms[i+1]['position'][idx], self.swarms[i]['position'][idx]