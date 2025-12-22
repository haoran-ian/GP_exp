import numpy as np

class EnhancedMultiSwarmCommunicator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.particles_per_swarm = 10
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.6  # Social component
        self.c3 = 0.5  # Local neighborhood component
        self.c4 = 0.3  # Inter-swarm communication component
        self.func_evals = 0
        self.min_inertia = 0.3
        self.max_inertia = 0.9
        self.w_decay = 0.99
        self.v_max = None  # Dynamic velocity clamping
        self.velocity_adaptation_rate = 0.1  # New parameter for velocity adaptation
        self.chaos_factor = 0.5  # Chaos factor for reshuffling

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.v_max = (ub - lb) * 0.3  # Update dynamic velocity scaling here

        # Initialize positions and velocities
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

            # Adaptive parameter adjustment based on synergy and results
            self._adaptive_adjustment(swarms, global_best_value)

            # Chaos-driven reshuffling
            if np.random.rand() < self.chaos_factor:
                self._chaos_reshuffling(swarms, lb, ub)

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

        local_best = np.mean(personal_bests, axis=0)  # Local neighborhood best

        for i in range(self.particles_per_swarm):
            r1, r2, r3 = np.random.rand(3)
            velocities[i] = (self.w * velocities[i] +
                             self.c1 * r1 * (personal_bests[i] - positions[i]) +
                             self.c2 * r2 * (global_best - positions[i]) +
                             self.c3 * r3 * (local_best - positions[i]))  # Added local component
            velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)  # Velocity clamping

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

        # Modify inertia weight dynamically based on velocity adaptation strategy
        if np.mean(np.abs(velocities)) < self.velocity_adaptation_rate:
            self.w = min(self.max_inertia, self.w * (1 + self.w_decay))
        else:
            self.w = max(self.min_inertia, self.w * (1 - self.w_decay))

    def _inter_swarm_interaction(self, swarms, swarm_index, global_best):
        if swarm_index == 0:
            return  # Skip interaction for the first swarm
        previous_swarm = swarms[swarm_index - 1]
        current_swarm = swarms[swarm_index]
        for i in range(self.particles_per_swarm):
            r4 = np.random.rand()
            interaction_factor = np.random.rand()
            current_swarm['positions'][i] += interaction_factor * (previous_swarm['positions'][i] - current_swarm['positions'][i]) \
                                             + self.c4 * r4 * (global_best - current_swarm['positions'][i])

    def _adaptive_adjustment(self, swarms, global_best_value):
        # Dynamic adjustment based on swarm synergy and convergence
        if global_best_value < 1e-6:  # Threshold for significant improvement
            self.num_swarms = max(1, self.num_swarms - 1)
            self.particles_per_swarm = min(20, self.particles_per_swarm + 1)
        else:
            self.num_swarms = min(10, self.num_swarms + 1)
            self.particles_per_swarm = max(5, self.particles_per_swarm - 1)

        # Further synergy-based adaptation
        self.c1 = max(1.0, self.c1 * 0.95)  # Dynamic cognitive component
        self.c2 = min(2.0, self.c2 * 1.05)  # Dynamic social component
        self.c4 = min(0.5, self.c4 * 1.05)  # Dynamic inter-swarm communication component

    def _chaos_reshuffling(self, swarms, lb, ub):
        for swarm in swarms:
            reshuffle_indices = np.random.permutation(self.particles_per_swarm)
            swarm["positions"] = swarm["positions"][reshuffle_indices]
            swarm["velocities"] = swarm["velocities"][reshuffle_indices]
            swarm["personal_bests"] = swarm["personal_bests"][reshuffle_indices]
            swarm["personal_best_values"] = swarm["personal_best_values"][reshuffle_indices]