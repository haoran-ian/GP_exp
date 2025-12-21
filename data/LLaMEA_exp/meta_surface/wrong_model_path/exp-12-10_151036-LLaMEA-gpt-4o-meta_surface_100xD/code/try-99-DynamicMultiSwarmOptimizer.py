import numpy as np

class DynamicMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles_per_swarm = 30
        self.num_swarms = 3
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.temperature = 100
        self.cooling_rate = 0.9
        self.diversity_threshold = 0.1
        self.compression_factor = 0.5
        self.learning_rate_adapt = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarms = [np.random.uniform(lb, ub, (self.num_particles_per_swarm, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.zeros((self.num_particles_per_swarm, self.dim)) for _ in range(self.num_swarms)]
        personal_best = [swarm.copy() for swarm in swarms]
        personal_best_values = [np.array([func(x) for x in swarm]) for swarm in personal_best]
        global_best = min((swarm[np.argmin(values)], np.min(values)) for swarm, values in zip(personal_best, personal_best_values))[0]
        best_value = min(np.min(values) for values in personal_best_values)

        eval_count = self.num_particles_per_swarm * self.num_swarms

        while eval_count < self.budget:
            new_swarms = []
            for swarm_idx, (swarm, velocity, p_best, p_best_values) in enumerate(zip(swarms, velocities, personal_best, personal_best_values)):
                if eval_count >= self.budget:
                    break

                for i in range(self.num_particles_per_swarm):
                    r1, r2 = np.random.rand(2)
                    velocity[i] = (self.inertia_weight * velocity[i] +
                                   self.cognitive_coeff * r1 * (p_best[i] - swarm[i]) +
                                   self.social_coeff * r2 * (global_best - swarm[i]))

                    swarm[i] = np.clip(swarm[i] + velocity[i], lb, ub)

                    current_value = func(swarm[i])
                    eval_count += 1

                    if current_value < p_best_values[i]:
                        p_best[i] = swarm[i]
                        p_best_values[i] = current_value
                        
                        if current_value < best_value:
                            global_best = swarm[i]
                            best_value = current_value

                # Adaptive inertia weight reduction
                self.inertia_weight = max(self.inertia_weight_min, 
                                          self.inertia_weight * self.cooling_rate + self.learning_rate_adapt * (best_value - min(p_best_values)))

                # Simulated annealing-inspired acceptance mechanism
                if np.random.rand() < np.exp(-abs(current_value - best_value) / self.temperature):
                    global_best = swarm[i]
                    best_value = current_value

                self.temperature *= self.cooling_rate

                # Check if swarm needs to be split or merged
                if np.std(p_best_values) < self.diversity_threshold and len(swarms) < 5:
                    split_swarm = np.array_split(swarm, 2)
                    split_velocity = np.array_split(velocity, 2)
                    split_pbest = np.array_split(p_best, 2)
                    split_pbest_values = np.array_split(p_best_values, 2)
                    new_swarms.extend(zip(split_swarm, split_velocity, split_pbest, split_pbest_values))
                else:
                    new_swarms.append((swarm, velocity, p_best, p_best_values))

            if len(new_swarms) > 1 and len(new_swarms) == self.num_swarms:
                # Attempt to merge two similar swarms
                for idx in range(0, len(new_swarms) - 1, 2):
                    swarm1, vel1, pbest1, pbest_values1 = new_swarms[idx]
                    swarm2, vel2, pbest2, pbest_values2 = new_swarms[idx+1]
                    if np.linalg.norm(np.mean(pbest1, axis=0) - np.mean(pbest2, axis=0)) < self.diversity_threshold:
                        merged_swarm = np.vstack((swarm1, swarm2))
                        merged_velocity = np.vstack((vel1, vel2))
                        merged_pbest = np.vstack((pbest1, pbest2))
                        merged_pbest_values = np.hstack((pbest_values1, pbest_values2))
                        new_swarms[idx] = (merged_swarm, merged_velocity, merged_pbest, merged_pbest_values)
                        new_swarms.pop(idx+1)

            swarms = [item[0] for item in new_swarms]
            velocities = [item[1] for item in new_swarms]
            personal_best = [item[2] for item in new_swarms]
            personal_best_values = [item[3] for item in new_swarms]

        return global_best