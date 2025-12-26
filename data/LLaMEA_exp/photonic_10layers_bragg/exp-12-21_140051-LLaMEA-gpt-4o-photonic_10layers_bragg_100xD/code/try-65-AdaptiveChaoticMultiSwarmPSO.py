import numpy as np

class AdaptiveChaoticMultiSwarmPSO:
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
        self.memory_sharing_probability = 0.05
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

                w = self.w_max - (self.w_max - self.w_min) * (evaluations**0.5 / self.budget**0.5)
                adaptive_lr = 0.1 + 0.9 * (1 - evaluations / self.budget)
                self.mutation_factor = 0.5 + 0.5 * np.sin(evaluations * np.pi / self.budget) * adaptive_lr

                convergence_speed = abs(swarm["global_best_val"] - overall_best_val)
                dynamic_lr = 0.5 + 0.5 * np.tanh(convergence_speed)

                c_damp = 1 - evaluations / self.budget
                r1, r2, r3 = (np.random.rand(self.population_size, self.dim) for _ in range(3))
                swarm["velocities"] = (w * swarm["velocities"]
                                       + self.c1 * c_damp * r1 * (swarm["personal_best_pos"] - swarm["population"])
                                       + self.c2 * c_damp * r2 * (swarm["global_best_pos"] - swarm["population"])
                                       + self.c2 * c_damp * r3 * (inter_swarm_best_pos - swarm["population"]))
                swarm["velocities"] *= (dynamic_lr * (0.1 + 0.9 * (1 - evaluations / self.budget)**2))
                
                if np.random.rand() < 0.1:
                    swarm["velocities"] += np.random.normal(0, 0.1, swarm["velocities"].shape)
                swarm["population"] = np.clip(swarm["population"] + swarm["velocities"], lb, ub)

                for i in range(self.population_size):
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = swarm["population"][np.random.choice(idxs, 3, replace=False)]
                    chaos_factor = np.sin(evaluations * np.pi / self.budget)
                    mutant = np.clip(a + self.mutation_factor * chaos_factor * (b - c), lb, ub)
                    crossover_rate_varied = self.crossover_rate * (1 - evaluations / self.budget)
                    crossover = np.random.rand(self.dim) < crossover_rate_varied
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