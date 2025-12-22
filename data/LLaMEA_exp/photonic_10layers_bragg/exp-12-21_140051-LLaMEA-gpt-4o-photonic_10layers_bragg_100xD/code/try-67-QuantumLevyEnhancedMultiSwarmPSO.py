import numpy as np

class QuantumLevyEnhancedMultiSwarmPSO:
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
        self.levy_alpha = 1.5
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

    def levy_flight(self, size, alpha):
        num = np.random.normal(0, 1, size)
        denom = np.abs(np.random.normal(0, 1, size))**(1/alpha)
        return num / denom

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        for swarm in self.swarms:
            chaos_population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
            swarm["population"] = chaos_population
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
                adaptive_lr = 0.1 + 0.9 * (1 - evaluations / self.budget)
                c_damp = 1 - evaluations / self.budget
                r1, r2, r3 = (np.random.rand(self.population_size, self.dim) for _ in range(3))

                swarm["velocities"] = (w * swarm["velocities"]
                                       + self.c1 * c_damp * r1 * (swarm["personal_best_pos"] - swarm["population"])
                                       + self.c2 * c_damp * r2 * (swarm["global_best_pos"] - swarm["population"])
                                       + self.c2 * c_damp * r3 * (inter_swarm_best_pos - swarm["population"]))
                swarm["population"] = np.clip(swarm["population"] + swarm["velocities"], lb, ub)

                for i in range(self.population_size):
                    if np.random.rand() < 0.1:
                        levy_step = self.levy_flight(self.dim, self.levy_alpha)
                        swarm["population"][i] = np.clip(swarm["population"][i] + levy_step, lb, ub)

                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = swarm["population"][np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
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