import numpy as np

class HybridDynamicMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.num_swarms = 4
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_coefficient_initial = 2.0
        self.social_coefficient_initial = 2.0
        self.progressive_reduction_rate = 0.95
        self.velocity_scaling_factor = 0.8
        self.mutation_factor = 0.6

    def adaptive_inertia_weight(self, evaluations):
        return (self.inertia_weight_initial - self.inertia_weight_final) * \
               ((self.budget - evaluations) / self.budget) + self.inertia_weight_final

    def progressive_search_space_reduction(self, evaluations, lb, ub):
        reduction_factor = self.progressive_reduction_rate ** (evaluations / self.budget)
        center = (ub + lb) / 2.0
        return center - reduction_factor * (center - lb), center + reduction_factor * (ub - center)

    def swarm_communication(self, swarms):
        combined_swarm = np.vstack(swarms)
        return np.mean(combined_swarm, axis=0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarms = [np.random.uniform(lb, ub, (self.population_size // self.num_swarms, self.dim))
                  for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.population_size // self.num_swarms, self.dim))
                      for _ in range(self.num_swarms)]

        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.array([func(p) for p in swarm]) for swarm in swarms]
        global_best_positions = [swarm[np.argmin(scores)] for swarm, scores in zip(swarms, personal_best_scores)]
        global_best_scores = [np.min(scores) for scores in personal_best_scores]

        evaluations = self.population_size
        while evaluations < self.budget:
            for idx in range(self.num_swarms):
                inertia_weight = self.adaptive_inertia_weight(evaluations)
                lb_dynamic, ub_dynamic = self.progressive_search_space_reduction(evaluations, lb, ub)

                for i in range(self.population_size // self.num_swarms):
                    if evaluations >= self.budget:
                        break

                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive_term = self.cognitive_coefficient_initial * r1 * (personal_best_positions[idx][i] - swarms[idx][i])
                    social_term = self.social_coefficient_initial * r2 * (global_best_positions[idx] - swarms[idx][i])
                    velocities[idx][i] = inertia_weight * velocities[idx][i] + cognitive_term + social_term

                    # Adaptive velocity clipping
                    velocities[idx][i] = np.clip(velocities[idx][i], lb_dynamic - swarms[idx][i], ub_dynamic - swarms[idx][i])

                    swarms[idx][i] += self.velocity_scaling_factor * velocities[idx][i]
                    swarms[idx][i] = np.clip(swarms[idx][i], lb_dynamic, ub_dynamic)

                    score = func(swarms[idx][i])
                    evaluations += 1

                    if score < personal_best_scores[idx][i]:
                        personal_best_scores[idx][i] = score
                        personal_best_positions[idx][i] = swarms[idx][i]

                        if score < global_best_scores[idx]:
                            global_best_scores[idx] = score
                            global_best_positions[idx] = swarms[idx][i]

            # Inter-swarm communication
            global_best_combined = self.swarm_communication(global_best_positions)
            for idx in range(self.num_swarms):
                global_best_positions[idx] = global_best_combined

        best_swarm_idx = np.argmin(global_best_scores)
        return global_best_positions[best_swarm_idx]