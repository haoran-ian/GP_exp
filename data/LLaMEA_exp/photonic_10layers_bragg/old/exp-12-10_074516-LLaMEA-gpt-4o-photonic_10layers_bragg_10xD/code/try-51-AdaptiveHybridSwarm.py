import numpy as np

class AdaptiveHybridSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.particles_per_swarm = 10
        self.w = 0.9  # Initial inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.6  # Social component
        self.c3 = 0.5  # Local neighborhood component
        self.c4 = 0.35  # Inter-swarm communication component
        self.elite_fraction = 0.2  # Fraction of elite particles
        self.func_evals = 0
        self.v_max = None  # Dynamic velocity clamping
        self.velocity_adaptation_rate = 0.1

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.v_max = (ub - lb) * 0.3

        swarms = [self._initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        global_best = None
        global_best_value = np.inf

        while self.func_evals < self.budget:
            for swarm_index, swarm in enumerate(swarms):
                self._evaluate_swarm(swarm, func)
                swarm_best, swarm_best_value = self._get_swarm_best(swarm)

                if swarm_best_value < global_best_value:
                    global_best_value = swarm_best_value
                    global_best = swarm_best

                self._update_swarm(swarm, swarm_best, global_best, lb, ub)
                self._hierarchical_interaction(swarms, swarm_index, global_best)

            self._adaptive_weight_adjustment()
            self._preserve_elites(swarms)

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

    def _get_swarm_best(self, swarm):
        best_idx = np.argmin(swarm["personal_best_values"])
        return swarm["personal_bests"][best_idx], swarm["personal_best_values"][best_idx]

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

    def _hierarchical_interaction(self, swarms, swarm_index, global_best):
        if swarm_index == 0:
            return
        previous_swarm = swarms[swarm_index - 1]
        current_swarm = swarms[swarm_index]
        interaction_strength = self.c4

        for i in range(self.particles_per_swarm):
            r4 = np.random.rand()
            interaction_factor = np.random.rand()
            current_swarm['positions'][i] += interaction_factor * (previous_swarm['positions'][i] - current_swarm['positions'][i]) \
                                             + interaction_strength * r4 * (global_best - current_swarm['positions'][i])

    def _adaptive_weight_adjustment(self):
        self.w = max(self.w_min, self.w * 0.99)

    def _preserve_elites(self, swarms):
        num_elites = max(1, int(self.elite_fraction * self.particles_per_swarm))
        for swarm in swarms:
            sorted_indices = np.argsort(swarm["personal_best_values"])
            elites = [swarm["personal_bests"][i] for i in sorted_indices[:num_elites]]
            for i in range(num_elites, self.particles_per_swarm):
                if np.any(swarm["positions"][i] in elites):
                    continue
                random_elite = elites[np.random.randint(0, len(elites))]
                swarm["positions"][i] = random_elite