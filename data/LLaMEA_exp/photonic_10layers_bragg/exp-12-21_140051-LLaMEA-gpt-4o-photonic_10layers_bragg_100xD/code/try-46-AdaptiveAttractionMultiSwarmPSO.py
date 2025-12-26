import numpy as np

class AdaptiveAttractionMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.swarm_count = 3
        self.w_max = 0.9
        self.w_min = 0.2
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.swarms = [self.initialize_swarm() for _ in range(self.swarm_count)]

    def initialize_swarm(self):
        return {
            "population": None,
            "velocities": None,
            "personal_best_pos": None,
            "personal_best_val": float('inf'),
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
        evaluations = 0
        overall_best_val = float('inf')
        overall_best_pos = None

        for swarm in self.swarms:
            chaos_population = self.chaotic_map_init(self.population_size, self.dim)
            swarm["population"] = lb + (ub - lb) * chaos_population
            swarm["velocities"] = np.random.uniform(-1, 1, (self.population_size, self.dim))
            swarm["personal_best_pos"] = np.copy(swarm["population"])
            swarm["personal_best_val"] = np.array([func(ind) for ind in swarm["population"]])
            best_idx = np.argmin(swarm["personal_best_val"])
            swarm["global_best_pos"] = swarm["personal_best_pos"][best_idx]
            swarm["global_best_val"] = swarm["personal_best_val"][best_idx]
            evaluations += self.population_size

        while evaluations < self.budget:
            for swarm in self.swarms:
                w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
                global_contribution = 1 if evaluations < self.budget / 2 else 0.5
                r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
                swarm["velocities"] = (w * swarm["velocities"]
                                       + self.c1 * r1 * (swarm["personal_best_pos"] - swarm["population"])
                                       + global_contribution * self.c2 * r2 * (swarm["global_best_pos"] - swarm["population"]))
                # Adaptive attraction based on swarm performance
                attraction_strength = 1 + np.tanh(1 - swarm["global_best_val"] / (overall_best_val + 1e-10))
                swarm["velocities"] *= attraction_strength
                swarm["population"] = np.clip(swarm["population"] + swarm["velocities"], lb, ub)

                for i in range(self.population_size):
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = swarm["population"][np.random.choice(idxs, 3, replace=False)]
                    mutant = a + self.mutation_factor * (b - c)
                    mutant = np.clip(mutant, lb, ub)
                    crossover = np.random.rand(self.dim) < self.crossover_rate
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