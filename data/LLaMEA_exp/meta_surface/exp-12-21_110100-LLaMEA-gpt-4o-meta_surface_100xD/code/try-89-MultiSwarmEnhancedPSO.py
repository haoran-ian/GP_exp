import numpy as np

class MultiSwarmEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, budget // 2)
        self.num_swarms = 3
        self.swarm_size = self.num_particles // self.num_swarms
        self.particles = np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.rand(self.num_particles, self.dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = np.inf * np.ones(self.dim)
        self.temp = 100
        self.alpha = 0.99
        self.initial_alpha = self.alpha
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.topology_switch_threshold = 0.5
        self.exchange_rate = 5

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        self.particles = bounds[0] + self.particles * (bounds[1] - bounds[0])
        self.personal_best = self.particles.copy()
        fitness = np.apply_along_axis(func, 1, self.particles)
        pbest_fitness = fitness.copy()
        gbest_fitness = np.min(fitness)
        self.global_best = self.particles[np.argmin(fitness)]

        eval_count = self.num_particles
        topology_switch_counter = 0

        while eval_count < self.budget:
            self.inertia_weight = max(self.inertia_weight_min, self.inertia_weight * 0.995)
            
            for swarm in range(self.num_swarms):
                swarm_start = swarm * self.swarm_size
                swarm_end = (swarm + 1) * self.swarm_size
                local_particles = self.particles[swarm_start:swarm_end]
                local_velocities = self.velocities[swarm_start:swarm_end]
                local_best_idx = np.argmin(pbest_fitness[swarm_start:swarm_end])
                local_best = self.personal_best[swarm_start + local_best_idx]

                r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
                stochastic_component = np.random.randn(self.swarm_size, self.dim) * 0.01
                local_velocities = self.inertia_weight * local_velocities \
                    + self.c1 * r1 * (self.personal_best[swarm_start:swarm_end] - local_particles) \
                    + self.c2 * r2 * (local_best - local_particles) \
                    + stochastic_component
                local_particles += local_velocities
                local_particles = np.clip(local_particles, bounds[0], bounds[1])

                local_fitness = np.apply_along_axis(func, 1, local_particles)
                eval_count += self.swarm_size

                improved = local_fitness < pbest_fitness[swarm_start:swarm_end]
                self.personal_best[swarm_start:swarm_end][improved] = local_particles[improved]
                pbest_fitness[swarm_start:swarm_end][improved] = local_fitness[improved]

                if np.min(local_fitness) < gbest_fitness:
                    gbest_fitness = np.min(local_fitness)
                    self.global_best = local_particles[np.argmin(local_fitness)]

                if eval_count % (self.exchange_rate * self.swarm_size) < self.swarm_size:
                    elite_indices = np.argpartition(pbest_fitness[swarm_start:swarm_end], 2)[:2]
                    for i in elite_indices:
                        step_size = np.random.normal(0, 0.1, self.dim) * np.random.pareto(1.5, self.dim)
                        candidate = self.personal_best[swarm_start + i] + step_size
                        mutation = np.random.normal(0, 0.01, self.dim)
                        candidate += mutation
                        candidate = np.clip(candidate, bounds[0], bounds[1])
                        candidate_fitness = func(candidate)
                        eval_count += 1
                        if candidate_fitness < pbest_fitness[swarm_start + i]:
                            self.personal_best[swarm_start + i] = candidate
                            pbest_fitness[swarm_start + i] = candidate_fitness
                            if candidate_fitness < gbest_fitness:
                                gbest_fitness = candidate_fitness
                                self.global_best = candidate

            self.temp *= self.alpha
            self.alpha = max(0.9 * self.alpha, self.initial_alpha * 0.1)
            topology_switch_counter += 1

        return self.global_best