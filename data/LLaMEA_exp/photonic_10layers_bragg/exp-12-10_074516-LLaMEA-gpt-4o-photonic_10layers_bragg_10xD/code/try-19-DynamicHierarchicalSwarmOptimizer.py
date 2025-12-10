import numpy as np

class DynamicHierarchicalSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.particles_per_swarm = 10
        self.func_evals = 0
        self.w_decay = 0.98
        self.min_inertia = 0.3
        self.max_inertia = 0.9
        self.w = self.max_inertia
        self.c1 = 1.5
        self.c2 = 1.6

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        # Initialize swarms
        swarms = [self._initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        global_best = None
        global_best_value = np.inf

        while self.func_evals < self.budget:
            for swarm in swarms:
                swarm_best, swarm_best_value = self._evaluate_swarm(swarm, func)

                if swarm_best_value < global_best_value:
                    global_best_value = swarm_best_value
                    global_best = swarm_best

                self._update_swarm(swarm, swarm_best, global_best, lb, ub)

            self._fuzzy_adjustment(swarms, global_best_value)

        return global_best

    def _initialize_swarm(self, lb, ub):
        positions = np.random.uniform(lb, ub, (self.particles_per_swarm, self.dim))
        velocities = np.random.uniform(-1, 1, (self.particles_per_swarm, self.dim))
        personal_bests = np.copy(positions)
        personal_best_values = np.full(self.particles_per_swarm, np.inf)
        return {"positions": positions, "velocities": velocities, 
                "personal_bests": personal_bests, "personal_best_values": personal_best_values}

    def _evaluate_swarm(self, swarm, func):
        positions = swarm["positions"]
        personal_bests = swarm["personal_bests"]
        personal_best_values = swarm["personal_best_values"]

        for i, position in enumerate(positions):
            if self.func_evals >= self.budget:
                break
            value = func(position)
            self.func_evals += 1

            if value < personal_best_values[i]:
                personal_best_values[i] = value
                personal_bests[i] = position

        best_idx = np.argmin(personal_best_values)
        return personal_bests[best_idx], personal_best_values[best_idx]

    def _update_swarm(self, swarm, swarm_best, global_best, lb, ub):
        positions = swarm["positions"]
        velocities = swarm["velocities"]
        personal_bests = swarm["personal_bests"]

        for i in range(self.particles_per_swarm):
            r1, r2 = np.random.rand(2)
            velocities[i] = (self.w * velocities[i] +
                             self.c1 * r1 * (personal_bests[i] - positions[i]) +
                             self.c2 * r2 * (global_best - positions[i]))

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

        self.w = max(self.min_inertia, self.w * self.w_decay)

    def _fuzzy_adjustment(self, swarms, global_best_value):
        # Fuzzy Logic Controller for adaptive parameter tuning
        improvement = (global_best_value < 1e-6)
        if improvement:
            self.num_swarms = max(1, self.num_swarms - 1)
            self.particles_per_swarm = min(20, self.particles_per_swarm + 1)
        else:
            self.num_swarms = min(10, self.num_swarms + 1)
            self.particles_per_swarm = max(5, self.particles_per_swarm - 1)

        # Adjust cognitive and social components using fuzzy logic
        if self.num_swarms > 5:
            self.c1 = max(1.0, self.c1 * 0.95)
            self.c2 = min(2.0, self.c2 * 1.05)
        else:
            self.c1 = min(2.0, self.c1 * 1.05)
            self.c2 = max(1.0, self.c2 * 0.95)