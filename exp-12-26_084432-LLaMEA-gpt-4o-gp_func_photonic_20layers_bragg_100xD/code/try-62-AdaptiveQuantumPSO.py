import numpy as np
from scipy.optimize import minimize

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_component = 2.0
        self.social_component = 2.0
        self.num_swarms = 3
        self.chaotic_sequence = self._init_chaotic_sequence()

    def _init_chaotic_sequence(self):
        chaotic_sequence = np.zeros(self.budget)
        chaotic_sequence[0] = np.random.rand()
        r = 3.99
        for i in range(1, self.budget):
            chaotic_sequence[i] = r * chaotic_sequence[i-1] * (1 - chaotic_sequence[i-1])
        return chaotic_sequence

    def _opposition_based_learning(self, positions, lower_bound, upper_bound):
        return lower_bound + upper_bound - positions

    def __call__(self, func):
        np.random.seed(42)
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        swarm_size = self.population_size // self.num_swarms

        positions = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), 
                                       (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            self.inertia_weight = self.inertia_weight_min + \
                                  (0.9 - self.inertia_weight_min) * (1 - evaluations / self.budget)
            chaotic_factor = self.chaotic_sequence[evaluations] * (0.9 - self.inertia_weight_min)
            self.inertia_weight += chaotic_factor * 0.5

            for k in range(self.num_swarms):
                swarm_indices = range(k * swarm_size, (k + 1) * swarm_size)
                r1, r2 = np.random.rand(2, swarm_size, self.dim)
                self.cognitive_component = 1.5 + 0.5 * np.sin(evaluations)
                
                velocities[swarm_indices] = (self.inertia_weight * velocities[swarm_indices] +
                                             self.cognitive_component * r1 * (personal_best_positions[swarm_indices] - positions[swarm_indices]) +
                                             self.social_component * r2 * (global_best_position - positions[swarm_indices]))

                # Quantum-inspired position update
                phi = np.random.rand(swarm_size, self.dim)
                positions[swarm_indices] = 0.5 * (personal_best_positions[swarm_indices] + global_best_position) + \
                                           phi * np.abs(personal_best_positions[swarm_indices] - global_best_position)

                positions[swarm_indices] = np.clip(positions[swarm_indices], lower_bound, upper_bound)

                # Opposition-based learning
                opposition_positions = self._opposition_based_learning(positions[swarm_indices], lower_bound, upper_bound)
                opposition_scores = np.array([func(x) for x in opposition_positions])
                for i, idx in enumerate(swarm_indices):
                    if opposition_scores[i] < personal_best_scores[idx]:
                        personal_best_scores[idx] = opposition_scores[i]
                        personal_best_positions[idx] = opposition_positions[i]
                        if opposition_scores[i] < global_best_score:
                            global_best_score = opposition_scores[i]
                            global_best_position = opposition_positions[i]

            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                    if scores[i] < global_best_score:
                        global_best_score = scores[i]
                        global_best_position = positions[i]

            if evaluations < self.budget:
                best_idx = np.argmin(scores)
                res = minimize(func, positions[best_idx], bounds=[(lb, ub) for lb, ub in zip(lower_bound, upper_bound)], method='L-BFGS-B')
                if res.fun < global_best_score:
                    global_best_score = res.fun
                    global_best_position = res.x
                evaluations += res.nfev

        return global_best_position, global_best_score