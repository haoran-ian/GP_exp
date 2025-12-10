import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.particles_per_swarm = 10
        self.w_min = 0.3  # Minimum inertia weight
        self.w_max = 0.9  # Maximum inertia weight
        self.c1_min = 1.0  # Minimum cognitive component
        self.c1_max = 2.0  # Maximum cognitive component
        self.c2_min = 1.0  # Minimum social component
        self.c2_max = 2.0  # Maximum social component
        self.func_evals = 0

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        # Initialize positions and velocities
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

            # Adaptively adjust swarm parameters
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

        for i in range(self.particles_per_swarm):
            r1, r2 = np.random.rand(2)
            w = self._dynamic_inertia_weight()
            c1 = self._adaptive_learning_rate(global_best, positions[i], self.c1_min, self.c1_max)
            c2 = self._adaptive_learning_rate(global_best, positions[i], self.c2_min, self.c2_max)

            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_bests[i] - positions[i]) +
                             c2 * r2 * (global_best - positions[i]))

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

    def _dynamic_inertia_weight(self):
        """ Dynamically adjust inertia weight over time to balance exploration and exploitation. """
        return self.w_max - (self.w_max - self.w_min) * (self.func_evals / self.budget)

    def _adaptive_learning_rate(self, global_best, position, min_rate, max_rate):
        """ Adaptive learning rate based on distance to the global best. """
        distance = np.linalg.norm(global_best - position)
        norm_distance = distance / np.sqrt(self.dim)
        return min_rate + (max_rate - min_rate) * (1 - norm_distance)

    def _adaptive_adjustment(self, swarms, global_best_value):
        # Example simple adjustment strategy based on global best improvement
        if global_best_value < 1e-6:  # Threshold example
            self.num_swarms = max(1, self.num_swarms - 1)
            self.particles_per_swarm = min(20, self.particles_per_swarm + 1)
        else:
            self.num_swarms = min(10, self.num_swarms + 1)
            self.particles_per_swarm = max(5, self.particles_per_swarm - 1)