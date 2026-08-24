import numpy as np

class RefinedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Initial differential evolution scaling factor
        self.CR = 0.9  # Crossover rate
        self.w = 0.9  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.chaotic_sequence = self.generate_chaotic_sequence(self.population_size * 2)

    def generate_chaotic_sequence(self, size):
        # Logistic map for chaotic sequence generation
        x = 0.7
        sequence = []
        for _ in range(size):
            x = 4 * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        pbest = pop.copy()
        pbest_fitness = fitness.copy()
        gbest = pop[np.argmin(fitness)]
        gbest_fitness = np.min(fitness)

        evaluations = self.population_size
        chaos_idx = 0
        while evaluations < self.budget:
            # Adaptive Differential Evolution Phase
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                self.F = 0.5 + 0.5 * self.chaotic_sequence[chaos_idx % len(self.chaotic_sequence)]
                chaos_idx += 1
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < pbest_fitness[i]:
                        pbest[i] = trial
                        pbest_fitness[i] = trial_fitness
                        if trial_fitness < gbest_fitness:
                            gbest = trial
                            gbest_fitness = trial_fitness

            if evaluations >= self.budget:
                break

            # Chaotic Particle Swarm Optimization Phase
            for i in range(self.population_size):
                r1, r2 = self.chaotic_sequence[chaos_idx % len(self.chaotic_sequence)], self.chaotic_sequence[(chaos_idx + 1) % len(self.chaotic_sequence)]
                chaos_idx += 2
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (pbest[i] - pop[i]) +
                                 self.c2 * r2 * (gbest - pop[i]))
                pop[i] = np.clip(pop[i] + velocities[i], lb, ub)
                particle_fitness = func(pop[i])
                evaluations += 1
                if particle_fitness < fitness[i]:
                    fitness[i] = particle_fitness
                    if particle_fitness < pbest_fitness[i]:
                        pbest[i] = pop[i]
                        pbest_fitness[i] = particle_fitness
                        if particle_fitness < gbest_fitness:
                            gbest = pop[i]
                            gbest_fitness = particle_fitness

        return gbest