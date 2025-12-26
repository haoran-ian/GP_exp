import numpy as np

class HierarchicalSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.particles_per_swarm = 10
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.6  # Social component
        self.func_evals = 0
        self.min_inertia = 0.3
        self.max_inertia = 0.9
        self.w_decay = 0.98
        self.lb = None
        self.ub = None
        self.v_max_factor = 0.2

    def __call__(self, func):
        bounds = func.bounds
        self.lb, self.ub = bounds.lb, bounds.ub
        self.v_max = (self.ub - self.lb) * self.v_max_factor

        # Initialize positions and velocities
        swarms = [self._initialize_swarm() for _ in range(self.num_swarms)]
        global_best = None
        global_best_value = np.inf

        while self.func_evals < self.budget:
            for swarm in swarms:
                swarm_best, swarm_best_value = self._evaluate_swarm(swarm, func)

                if swarm_best_value < global_best_value:
                    global_best_value = swarm_best_value
                    global_best = swarm_best

                self._update_swarm(swarm, swarm_best, global_best)

            # Hierarchical adjustment based on synergy and results
            self._hierarchical_adjustment(swarms, global_best_value)

        return global_best

    def _initialize_swarm(self):
        positions = np.random.uniform(self.lb, self.ub, (self.particles_per_swarm, self.dim))
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

    def _update_swarm(self, swarm, swarm_best, global_best):
        positions = swarm["positions"]
        velocities = swarm["velocities"]
        personal_bests = swarm["personal_bests"]

        for i in range(self.particles_per_swarm):
            r1, r2 = np.random.rand(2)
            velocities[i] = (self.w * velocities[i] +
                             self.c1 * r1 * (personal_bests[i] - positions[i]) +
                             self.c2 * r2 * (global_best - positions[i]))
            velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)  # Velocity clamping

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], self.lb, self.ub)

        self.w = max(self.min_inertia, self.w * self.w_decay)

    def _hierarchical_adjustment(self, swarms, global_best_value):
        # Hierarchical adjustment considering overall performance and convergence
        if global_best_value < 1e-6:  # Threshold for significant improvement
            self.num_swarms = max(1, self.num_swarms - 1)
            self.particles_per_swarm = min(30, self.particles_per_swarm + 2)
        else:
            self.num_swarms = min(10, self.num_swarms + 1)
            self.particles_per_swarm = max(5, self.particles_per_swarm - 1)

        # Adaptive synergy-based parameter tuning
        self.c1 = max(1.0, self.c1 * 0.9)  # Dynamic cognitive component
        self.c2 = min(2.5, self.c2 * 1.1)  # Dynamic social component