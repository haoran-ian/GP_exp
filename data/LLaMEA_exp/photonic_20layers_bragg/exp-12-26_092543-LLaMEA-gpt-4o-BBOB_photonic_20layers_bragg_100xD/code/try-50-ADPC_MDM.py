import numpy as np

class ADPC_MDM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_init = 0.5
        self.CR_init = 0.9
        self.pop = None
        self.bounds = None

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.population_size
        successful_mutations = []

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(self.population_size):
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                F_local = np.clip(self.F_init + 0.1 * np.random.randn(), 0, 1)
                CR_local = np.clip(self.CR_init + 0.05 * np.random.randn(), 0, 1)

                # Multi-Directional Mutation
                mutant1 = r1 + F_local * (r2 - r3)
                mutant2 = r1 + F_local * (self.pop[i] - r2)
                mutant1 = np.clip(mutant1, lb, ub)
                mutant2 = np.clip(mutant2, lb, ub)

                cross_points1 = np.random.rand(self.dim) < CR_local
                cross_points2 = np.random.rand(self.dim) < CR_local
                
                if not np.any(cross_points1):
                    cross_points1[np.random.randint(0, self.dim)] = True
                if not np.any(cross_points2):
                    cross_points2[np.random.randint(0, self.dim)] = True

                trial1 = np.where(cross_points1, mutant1, self.pop[i])
                trial2 = np.where(cross_points2, mutant2, self.pop[i])

                trial1_fitness = func(trial1)
                trial2_fitness = func(trial2)
                evaluations += 2

                # Selection
                if trial1_fitness < fitness[i] or trial2_fitness < fitness[i]:
                    if trial1_fitness < trial2_fitness:
                        new_pop[i] = trial1
                        fitness[i] = trial1_fitness
                        successful_mutations.append((F_local, CR_local))
                    else:
                        new_pop[i] = trial2
                        fitness[i] = trial2_fitness
                        successful_mutations.append((F_local, CR_local))
                else:
                    new_pop[i] = self.pop[i]

            self.pop = new_pop

            if successful_mutations:
                self.F_init = np.mean([x[0] for x in successful_mutations])
                self.CR_init = np.mean([x[1] for x in successful_mutations])
                successful_mutations.clear()
                self.population_size = max(5 * self.dim, int(10 * self.dim * len(successful_mutations) / self.population_size))

            if evaluations < self.budget and np.random.rand() < 0.1:
                rand_index = np.random.randint(self.population_size)
                self.pop[rand_index] = lb + (ub - lb) * np.random.rand(self.dim)
                fitness[rand_index] = func(self.pop[rand_index])
                evaluations += 1

        best_index = np.argmin(fitness)
        return self.pop[best_index]