import numpy as np

class EnhancedAdaptiveMultiSwarmOptimizerV2:
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
        self.w_decay = 0.99
        self.diversity_threshold = 0.01  # Diversity threshold for dynamic topology

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

            # Adaptive parameter adjustment based on synergy and results
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
            velocities[i] = (self.w * velocities[i] +
                             self.c1 * r1 * (personal_bests[i] - positions[i]) +
                             self.c2 * r2 * (global_best - positions[i]))

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

        self.w = max(self.min_inertia, self.w * self.w_decay)

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

        # Diversity enhancement through dynamic neighborhood topology
        for swarm in swarms:
            diversity = np.mean(np.std(swarm["positions"], axis=0))
            if diversity < self.diversity_threshold:
                self._increase_diversity(swarm)

    def _increase_diversity(self, swarm):
        # Introduce random changes to increase diversity
        random_indices = np.random.choice(self.particles_per_swarm, size=2, replace=False)
        swarm["positions"][random_indices[0]] += np.random.uniform(-0.1, 0.1, self.dim)
        swarm["positions"][random_indices[1]] -= np.random.uniform(-0.1, 0.1, self.dim)
        swarm["positions"] = np.clip(swarm["positions"], func.bounds.lb, func.bounds.ub)