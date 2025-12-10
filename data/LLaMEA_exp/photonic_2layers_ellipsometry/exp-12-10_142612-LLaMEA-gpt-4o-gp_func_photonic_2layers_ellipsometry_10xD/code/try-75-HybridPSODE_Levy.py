import numpy as np

class HybridPSODE_Levy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = 0.9

    def levy_flight(self, L=1.5):
        sigma = (np.math.gamma(1 + L) * np.sin(np.pi * L / 2) /
                 (np.math.gamma((1 + L) / 2) * L * 2 ** ((L - 1) / 2))) ** (1 / L)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v) ** (1 / L))
        return step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.c1 * r1 * (personal_best[i] - pop[i]) +
                               self.c2 * r2 * (global_best - pop[i]))
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)

            for i in range(self.population_size):
                new_fitness = func(pop[i])
                evaluations += 1
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = pop[i]
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < func(global_best):
                        global_best = pop[i]

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < func(global_best):
                        global_best = trial

            if evaluations < self.budget:
                # Implementing Levy Flight for stochastic exploration
                for i in range(self.population_size):
                    levy_step = self.levy_flight()
                    pop[i] = np.clip(pop[i] + levy_step, lb, ub)
                    levy_fitness = func(pop[i])
                    evaluations += 1
                    if levy_fitness < personal_best_fitness[i]:
                        personal_best[i] = pop[i]
                        personal_best_fitness[i] = levy_fitness
                        if levy_fitness < func(global_best):
                            global_best = pop[i]

        return global_best