import numpy as np

class EnhancedAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.particles_per_swarm = 15
        self.w = 0.6
        self.c1 = 1.5
        self.c2 = 1.8
        self.c3 = 0.4
        self.c4 = 0.4
        self.func_evals = 0
        self.min_inertia = 0.3
        self.max_inertia = 0.8
        self.w_decay = 0.98
        self.v_max = None
        self.velocity_adaptation_threshold = 0.05

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.v_max = (ub - lb) * 0.2

        swarms = [self._initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        global_best = None
        global_best_value = np.inf

        while self.func_evals < self.budget:
            for swarm_index, swarm in enumerate(swarms):
                swarm_best, swarm_best_value = self._evaluate_swarm(swarm, func)
                if swarm_best_value < global_best_value:
                    global_best_value = swarm_best_value
                    global_best = swarm_best

                self._update_swarm(swarm, swarm_best, global_best, lb, ub)
                self._inter_swarm_interaction(swarms, swarm_index, global_best)

            self._adaptive_adjustment(swarms, global_best_value)

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

        local_best = np.mean(personal_bests, axis=0)

        for i in range(self.particles_per_swarm):
            r1, r2, r3 = np.random.rand(3)
            velocities[i] = (self.w * velocities[i] +
                             self.c1 * r1 * (personal_bests[i] - positions[i]) +
                             self.c2 * r2 * (global_best - positions[i]) +
                             self.c3 * r3 * (local_best - positions[i]))
            velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

        if np.mean(np.abs(velocities)) < self.velocity_adaptation_threshold:
            self.w = min(self.max_inertia, self.w * (1 + self.w_decay))
        else:
            self.w = max(self.min_inertia, self.w * (1 - self.w_decay))

    def _inter_swarm_interaction(self, swarms, swarm_index, global_best):
        if swarm_index == 0:
            return
        previous_swarm = swarms[swarm_index - 1]
        current_swarm = swarms[swarm_index]
        for i in range(self.particles_per_swarm):
            r4 = np.random.rand()
            interaction_factor = np.random.rand()
            current_swarm['positions'][i] += interaction_factor * (previous_swarm['positions'][i] - current_swarm['positions'][i]) \
                                             + self.c4 * r4 * (global_best - current_swarm['positions'][i])

    def _adaptive_adjustment(self, swarms, global_best_value):
        if global_best_value < 1e-6:
            self.num_swarms = max(1, self.num_swarms - 1)
            self.particles_per_swarm = min(25, self.particles_per_swarm + 1)
        else:
            self.num_swarms = min(10, self.num_swarms + 1)
            self.particles_per_swarm = max(5, self.particles_per_swarm - 1)

        self.c1 = max(1.0, self.c1 * 0.97)
        self.c2 = min(2.1, self.c2 * 1.03)
        self.c4 = min(0.6, self.c4 * 1.02)