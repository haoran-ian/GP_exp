import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.F = 0.7
        self.CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9
        self.vel_limit = 0.1
        self.search_space_factor = 0.05
        self.success_rate_threshold = 0.2
        self.num_swarms = 3
        self.regrouping_interval = budget // 5

    def adaptive_operator_selection(self, success_rates):
        if success_rates['DE'] > self.success_rate_threshold:
            return 'DE'
        return 'PSO'

    def local_search(self, solution, func, lb, ub):
        local_best = solution
        local_best_value = func(local_best)
        for _ in range(5):
            candidate = local_best + np.random.uniform(-1, 1, self.dim) * self.search_space_factor * (ub - lb)
            candidate = np.clip(candidate, lb, ub)
            candidate_value = func(candidate)
            if candidate_value < local_best_value:
                local_best, local_best_value = candidate, candidate_value
        return local_best, local_best_value

    def regroup_swarms(self, populations, velocities):
        merged_pop = np.vstack(populations)
        np.random.shuffle(merged_pop)
        split_indices = np.array_split(merged_pop, self.num_swarms)
        new_velocities = [np.zeros((len(swarm), self.dim)) for swarm in split_indices]
        return split_indices, new_velocities

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size + int(0.1 * self.budget / self.dim)
        swarms = [np.random.uniform(lb, ub, (pop_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.zeros((pop_size, self.dim)) for _ in range(self.num_swarms)]
        personal_bests = [swarm.copy() for swarm in swarms]
        personal_best_values = [np.array([func(ind) for ind in swarm]) for swarm in personal_bests]
        global_best = min([swarm[np.argmin(pbv)] for swarm, pbv in zip(swarms, personal_best_values)], key=func)
        global_best_value = min([np.min(pbv) for pbv in personal_best_values])
        evaluations = self.num_swarms * pop_size
        success_rates = {'DE': 0.5, 'PSO': 0.5}
        iteration = 0

        while evaluations < self.budget:
            selected_operator = self.adaptive_operator_selection(success_rates)

            for swarm_idx, (swarm, vel, pbest, pbest_values) in enumerate(zip(swarms, velocities, personal_bests, personal_best_values)):
                if selected_operator == 'DE':
                    successful_updates = 0
                    for i in range(pop_size):
                        indices = list(range(i)) + list(range(i + 1, pop_size))
                        a, b, c = np.random.choice(indices, 3, replace=False)
                        mutant = np.clip(swarm[a] + self.F * (swarm[b] - swarm[c]), lb, ub)
                        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, swarm[i])
                        trial_value = func(trial)
                        evaluations += 1

                        if trial_value < pbest_values[i]:
                            pbest[i], pbest_values[i] = trial, trial_value
                            if trial_value < global_best_value:
                                global_best, global_best_value = trial, trial_value
                            successful_updates += 1
                    success_rates['DE'] = successful_updates / pop_size
                    self.F = 0.5 + 0.5 * np.random.rand()

                if selected_operator == 'PSO':
                    successful_updates = 0
                    for i in range(pop_size):
                        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                        vel[i] = self.w * vel[i] + self.c1 * r1 * (pbest[i] - swarm[i]) + self.c2 * r2 * (global_best - swarm[i])
                        vel[i] = np.clip(vel[i], -self.vel_limit * (ub - lb), self.vel_limit * (ub - lb))
                        swarm[i] = np.clip(swarm[i] + vel[i], lb, ub)

                        new_value = func(swarm[i])
                        evaluations += 1
                        if new_value < pbest_values[i]:
                            pbest[i], pbest_values[i] = swarm[i], new_value
                            if new_value < global_best_value:
                                global_best, global_best_value = swarm[i], new_value
                            successful_updates += 1
                    success_rates['PSO'] = successful_updates / pop_size
                    self.w = 0.4 + 0.5 * np.random.rand()

            global_best, global_best_value = self.local_search(global_best, func, lb, ub)

            iteration += 1
            if iteration % self.regrouping_interval == 0:
                swarms, velocities = self.regroup_swarms(swarms, velocities)

        return global_best