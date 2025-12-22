import numpy as np

class ChaosEnhancedMultiSwarmDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_initial_size = 30
        self.population_variability = 5
        self.F_initial = 0.8
        self.CR_initial = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.velocity_clamp = np.array([1.0] * dim)
        self.num_swarms = 3
        self.swarms = []

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = self.population_initial_size

        def chaotic_map(x):
            return 4 * x * (1 - x)

        def initialize_swarm():
            swarm = {
                'population': np.random.uniform(lb, ub, (population_size, self.dim)),
                'velocities': np.random.uniform(-1, 1, (population_size, self.dim)),
                'fitness': np.full(population_size, np.inf),
                'personal_best': None,
                'personal_best_fitness': np.full(population_size, np.inf),
                'global_best': None,
                'global_best_fitness': np.inf
            }
            swarm['fitness'] = np.array([func(ind) for ind in swarm['population']])
            global_best_idx = np.argmin(swarm['fitness'])
            swarm['personal_best'] = swarm['population'].copy()
            swarm['personal_best_fitness'] = swarm['fitness'].copy()
            swarm['global_best'] = swarm['population'][global_best_idx]
            swarm['global_best_fitness'] = swarm['fitness'][global_best_idx]
            return swarm

        for _ in range(self.num_swarms):
            self.swarms.append(initialize_swarm())

        evaluations = self.num_swarms * population_size
        chaotic_param = 0.7

        while evaluations < self.budget:
            for swarm in self.swarms:
                progress = evaluations / self.budget
                F, CR = self.F_initial * (1 - progress), self.CR_initial * progress

                for i in range(population_size):
                    indices = list(range(population_size))
                    indices.remove(i)
                    a, b, c = swarm['population'][np.random.choice(indices, 3, replace=False)]

                    mutant = np.clip(a + F * (b - c), lb, ub)
                    crossover = np.random.rand(self.dim) < CR
                    trial = np.where(crossover, mutant, swarm['population'][i])

                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < swarm['fitness'][i]:
                        swarm['population'][i] = trial
                        swarm['fitness'][i] = trial_fitness
                        if trial_fitness < swarm['personal_best_fitness'][i]:
                            swarm['personal_best'][i] = trial
                            swarm['personal_best_fitness'][i] = trial_fitness
                            if trial_fitness < swarm['global_best_fitness']:
                                swarm['global_best'] = trial
                                swarm['global_best_fitness'] = trial_fitness

                chaotic_param = chaotic_map(chaotic_param)
                inertia_weight = self.inertia_weight_initial - progress * (self.inertia_weight_initial - self.inertia_weight_final)
                for i in range(population_size):
                    r1, r2 = np.random.rand(), np.random.rand()
                    swarm['velocities'][i] = (inertia_weight * swarm['velocities'][i] +
                                              self.c1 * r1 * (swarm['personal_best'][i] - swarm['population'][i]) * chaotic_param +
                                              self.c2 * r2 * (swarm['global_best'] - swarm['population'][i]) * chaotic_param)
                    swarm['velocities'][i] = np.clip(swarm['velocities'][i], -self.velocity_clamp, self.velocity_clamp)
                    swarm['population'][i] = np.clip(swarm['population'][i] + swarm['velocities'][i], lb, ub)

            # Multi-Swarm Cooperation
            elite_candidates = [swarm['global_best'] for swarm in self.swarms]
            elite_fitnesses = [swarm['global_best_fitness'] for swarm in self.swarms]
            best_elite_idx = np.argmin(elite_fitnesses)
            for swarm in self.swarms:
                if swarm['global_best_fitness'] > elite_fitnesses[best_elite_idx]:
                    swarm['global_best'] = elite_candidates[best_elite_idx]
                    swarm['global_best_fitness'] = elite_fitnesses[best_elite_idx]

        # Final selection of the best solution across all swarms
        final_best_idx = np.argmin([swarm['global_best_fitness'] for swarm in self.swarms])
        final_best = self.swarms[final_best_idx]['global_best']
        return final_best, func(final_best)