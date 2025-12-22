import numpy as np

class EnhancedDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.swarm_count = 3
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.memory_sharing_probability = 0.1
        self.mutation_factor = 0.8
        self.swarms = [self.initialize_swarm() for _ in range(self.swarm_count)]

    def initialize_swarm(self):
        return {
            "population": None,
            "velocities": None,
            "personal_best_pos": None,
            "personal_best_val": None,
            "global_best_pos": None,
            "global_best_val": float('inf')
        }

    def chaotic_map_init(self, N, D):
        chaos_seq = np.zeros((N, D))
        x0 = np.random.rand()
        for d in range(D):
            x = x0
            for n in range(N):
                x = 4 * x * (1 - x)
                chaos_seq[n, d] = x
        return chaos_seq

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        for swarm in self.swarms:
            chaos_population = self.chaotic_map_init(self.population_size, self.dim)
            swarm["population"] = lb + (ub - lb) * chaos_population
            swarm["velocities"] = np.random.uniform(-1, 1, (self.population_size, self.dim))
            swarm["personal_best_pos"] = np.copy(swarm["population"])
            swarm["personal_best_val"] = np.array([func(ind) for ind in swarm["population"]])
            best_idx = np.argmin(swarm["personal_best_val"])
            swarm["global_best_pos"] = swarm["personal_best_pos"][best_idx]
            swarm["global_best_val"] = swarm["personal_best_val"][best_idx]

        evaluations = self.swarm_count * self.population_size
        overall_best_val = min(swarm["global_best_val"] for swarm in self.swarms)
        overall_best_pos = None

        while evaluations < self.budget:
            inter_swarm_best_pos = min(self.swarms, key=lambda s: s["global_best_val"])["global_best_pos"]
            for swarm in self.swarms:
                if np.random.rand() < self.memory_sharing_probability:
                    swarm["global_best_pos"] = inter_swarm_best_pos
                    swarm["global_best_val"] = min(swarm["global_best_val"], min(s["global_best_val"] for s in self.swarms))

                w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
                chaotic_factor = np.sin(5 * np.pi * (evaluations / self.budget))

                r1, r2, r3 = (np.random.rand(self.population_size, self.dim) for _ in range(3))
                swarm["velocities"] = (w * swarm["velocities"]
                                       + self.c1 * r1 * (swarm["personal_best_pos"] - swarm["population"])
                                       + self.c2 * r2 * (swarm["global_best_pos"] - swarm["population"])
                                       + self.c2 * r3 * chaotic_factor * (inter_swarm_best_pos - swarm["population"]))

                swarm["population"] = np.clip(swarm["population"] + swarm["velocities"], lb, ub)

                for i in range(self.population_size):
                    if np.random.rand() < self.memory_sharing_probability:
                        idxs = [idx for idx in range(self.population_size) if idx != i]
                        a, b, c = swarm["population"][np.random.choice(idxs, 3, replace=False)]
                        mutant = np.clip(a + self.mutation_factor * chaotic_factor * (b - c), lb, ub)
                        crossover = np.random.rand(self.dim) < (0.5 + 0.5 * chaotic_factor)
                        trial = np.where(crossover, mutant, swarm["population"][i])
                        trial_val = func(trial)
                        evaluations += 1

                        if trial_val < swarm["personal_best_val"][i]:
                            swarm["personal_best_pos"][i] = trial
                            swarm["personal_best_val"][i] = trial_val

                            if trial_val < swarm["global_best_val"]:
                                swarm["global_best_pos"] = trial
                                swarm["global_best_val"] = trial_val

                        if evaluations >= self.budget:
                            break

                if swarm["global_best_val"] < overall_best_val:
                    overall_best_val = swarm["global_best_val"]
                    overall_best_pos = swarm["global_best_pos"]

        return overall_best_pos