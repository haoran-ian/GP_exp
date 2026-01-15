import numpy as np

class HybridMultiSwarmDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.num_particles = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.9
        self.bounds = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.swarm_best_positions = [None] * self.num_swarms
        self.swarm_best_values = [np.inf] * self.num_swarms
        self.particles = np.random.rand(self.num_swarms, self.num_particles, self.dim)
        self.velocities = np.zeros((self.num_swarms, self.num_particles, self.dim))
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.local_search_probability = 0.1

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        eval_count = 0
        lb, ub = self.bounds
        self.particles = lb + (ub - lb) * self.particles

        while eval_count < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.num_particles):
                    particle_pos = self.particles[swarm, i]
                    particle_value = func(particle_pos)
                    eval_count += 1

                    if particle_value < self.swarm_best_values[swarm]:
                        self.swarm_best_values[swarm] = particle_value
                        self.swarm_best_positions[swarm] = particle_pos.copy()

                    if particle_value < self.global_best_value:
                        self.global_best_value = particle_value
                        self.global_best_position = particle_pos.copy()

                    if np.random.rand() < self.local_search_probability:
                        # Apply differential evolution-like update
                        a, b, c = self._select_random_particles(swarm, i)
                        mutant = self.particles[swarm, a] + self.mutation_factor * (self.particles[swarm, b] - self.particles[swarm, c])
                        trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, particle_pos)
                        trial_value = func(trial)
                        eval_count += 1
                        if trial_value < particle_value:
                            self.particles[swarm, i] = trial
                            if trial_value < self.swarm_best_values[swarm]:
                                self.swarm_best_values[swarm] = trial_value
                                self.swarm_best_positions[swarm] = trial.copy()
                            if trial_value < self.global_best_value:
                                self.global_best_value = trial_value
                                self.global_best_position = trial.copy()

                    else:
                        selected_leader = self._select_leader(swarm)
                        inertia = self.inertia_weight * self.velocities[swarm, i]
                        cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                        social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                        self.velocities[swarm, i] = inertia + cognitive + social
                        self.particles[swarm, i] += self.velocities[swarm, i]

                        for d in range(self.dim):
                            if self.particles[swarm, i, d] < lb[d] or self.particles[swarm, i, d] > ub[d]:
                                self.particles[swarm, i, d] = lb[d] + (ub[d] - self.particles[swarm, i, d]) % (ub[d] - lb[d])

                self.inertia_weight = max(0.4, self.inertia_weight * 0.99)
                self.c1 = max(1.0, self.c1 - 0.001)
                self.c2 = min(2.0, self.c2 + 0.001)

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value

    def _select_leader(self, swarm):
        if np.random.rand() < self.local_search_probability:
            return self.global_best_position
        else:
            if np.random.rand() < 0.5:
                return self.swarm_best_positions[swarm]
            else:
                return self.particles[swarm, np.random.randint(self.num_particles)]

    def _select_random_particles(self, swarm, current_index):
        indices = list(range(self.num_particles))
        indices.remove(current_index)
        return np.random.choice(indices, 3, replace=False)