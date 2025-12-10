import numpy as np

class LayeredSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.particles_per_swarm = 10
        self.w = 0.9
        self.c1 = 1.6
        self.c2 = 1.7
        self.c3 = 0.6
        self.c4 = 0.45
        self.func_evals = 0
        self.v_max = None
        self.feedback_factor = 0.2
        self.stagnation_threshold = 50
        self.diversity_threshold = 0.08
        self.learning_layers = 3
        self.layer_amplification = 0.2

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.v_max = (ub - lb) * 0.3

        swarms = [self._initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        global_best = None
        global_best_value = np.inf
        stagnation_counter = 0

        while self.func_evals < self.budget:
            previous_global_best_value = global_best_value
            swarm_performance = []
            for swarm_index, swarm in enumerate(swarms):
                swarm_best, swarm_best_value = self._evaluate_swarm(swarm, func)
                swarm_performance.append(swarm_best_value)
                
                if swarm_best_value < global_best_value:
                    global_best_value = swarm_best_value
                    global_best = swarm_best
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                self._update_swarm(swarm, swarm_best, global_best, lb, ub)
                self._collaborative_interaction(swarms, swarm_index, global_best, swarm_performance)

            self._adaptive_adjustment(swarms, global_best_value, previous_global_best_value, stagnation_counter)

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
        layer_strengths = [1 + self.layer_amplification * l for l in range(self.learning_layers)]

        for i in range(self.particles_per_swarm):
            r1, r2, r3 = np.random.rand(3)
            for layer in layer_strengths:
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_bests[i] - positions[i]) +
                                 self.c2 * r2 * (global_best - positions[i]) * layer +
                                 self.c3 * r3 * (local_best - positions[i]) * layer)
            velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

    def _collaborative_interaction(self, swarms, swarm_index, global_best, swarm_performance):
        if swarm_index == 0:
            return
        previous_swarm = swarms[swarm_index - 1]
        current_swarm = swarms[swarm_index]
        performance_diversity = np.std(swarm_performance) / (np.mean(swarm_performance) + 1e-9)
        interaction_strength = self.c4 * (1 + np.tanh(performance_diversity)) * 1.1

        for i in range(self.particles_per_swarm):
            r4 = np.random.rand()
            interaction_factor = np.random.rand() * np.exp(-performance_diversity)
            current_swarm['positions'][i] += interaction_factor * (previous_swarm['positions'][i] - current_swarm['positions'][i]) \
                                             + interaction_strength * r4 * (global_best - current_swarm['positions'][i])

    def _adaptive_adjustment(self, swarms, global_best_value, previous_global_best_value, stagnation_counter):
        if global_best_value < 1e-6:
            self.num_swarms = max(1, self.num_swarms - 1)
            self.particles_per_swarm = min(20, self.particles_per_swarm + 1)
        else:
            self.num_swarms = min(10, self.num_swarms + 1)
            self.particles_per_swarm = max(5, self.particles_per_swarm - 1)

        diversity = np.mean([np.var(swarm["positions"]) for swarm in swarms])

        if stagnation_counter > self.stagnation_threshold or diversity < self.diversity_threshold:
            self.w = np.clip(self.w * 1.15, 0.4, 0.9)
            self.c1 = max(1.0, self.c1 * 1.2)
            self.c2 = min(2.0, self.c2 * 0.8)
        else:
            self.w = np.clip(self.w * 0.8, 0.4, 0.9)
            self.c1 = max(1.0, self.c1 * 0.9)
            self.c2 = min(2.0, self.c2 * 1.1)

        self.c3 *= 1.05
        self.c4 = min(0.6, self.c4 * 1.1)