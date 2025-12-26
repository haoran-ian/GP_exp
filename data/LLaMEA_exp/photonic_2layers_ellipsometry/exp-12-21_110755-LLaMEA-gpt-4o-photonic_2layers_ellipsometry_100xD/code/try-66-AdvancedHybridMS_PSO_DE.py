import numpy as np

class AdvancedHybridMS_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.num_swarms = 3
        self.swarms = []
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR = 0.9
        self.global_best_value = np.inf
        self.global_best = None
        self.current_eval = 0

    def adaptive_inertia_weight(self, swarm_idx):
        return self.w_max - ((self.w_max - self.w_min) * (swarm_idx / self.num_swarms))

    def initialize_swarms(self, lb, ub):
        for _ in range(self.num_swarms):
            population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
            velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
            personal_best = population.copy()
            personal_best_values = np.array([np.inf] * self.pop_size)
            self.swarms.append({
                'population': population,
                'velocities': velocities,
                'personal_best': personal_best,
                'personal_best_values': personal_best_values,
                'global_best': None,
                'global_best_value': np.inf
            })

    def update_particles(self, lb, ub, swarm):
        w = self.adaptive_inertia_weight(self.swarms.index(swarm))
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            swarm['velocities'][i] = (w * swarm['velocities'][i] +
                                      self.c1 * r1 * (swarm['personal_best'][i] - swarm['population'][i]) +
                                      self.c2 * r2 * (swarm['global_best'] - swarm['population'][i]))
            swarm['population'][i] += swarm['velocities'][i]
            swarm['population'][i] = np.clip(swarm['population'][i], lb, ub)

    def self_adaptive_differential_evolution(self, index, lb, ub, swarm):
        F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
        idxs = [idx for idx in range(self.pop_size) if idx != index]
        a, b, c = swarm['population'][np.random.choice(idxs, 3, replace=False)]
        mutant = a + F * (b - c)
        mutant = np.clip(mutant, lb, ub)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, swarm['population'][index])
        return trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarms(lb, ub)
        while self.current_eval < self.budget:
            for swarm in self.swarms:
                for i in range(self.pop_size):
                    candidate = self.self_adaptive_differential_evolution(i, lb, ub, swarm)
                    candidate_value = func(candidate)
                    self.current_eval += 1

                    if candidate_value < swarm['personal_best_values'][i]:
                        swarm['personal_best_values'][i] = candidate_value
                        swarm['personal_best'][i] = candidate.copy()

                    if candidate_value < swarm['global_best_value']:
                        swarm['global_best_value'] = candidate_value
                        swarm['global_best'] = candidate.copy()

                self.update_particles(lb, ub, swarm)

                if swarm['global_best_value'] < self.global_best_value:
                    self.global_best_value = swarm['global_best_value']
                    self.global_best = swarm['global_best'].copy()

        return self.global_best