import numpy as np

class IntegratedMultiSwarmPerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_count = 3
        swarms = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(swarm_count)]
        velocities = [np.zeros((self.population_size, self.dim)) for _ in range(swarm_count)]
        personal_bests = [swarm.copy() for swarm in swarms]
        personal_best_scores = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        global_best_indexes = [np.argmin(scores) for scores in personal_best_scores]
        global_bests = [personal_bests[i][global_best_indexes[i]] for i in range(swarm_count)]
        global_best_scores = [scores[global_best_indexes[i]] for i, scores in enumerate(personal_best_scores)]
        best_global_best_index = np.argmin(global_best_scores)
        overall_best = global_bests[best_global_best_index]
        overall_best_score = global_best_scores[best_global_best_index]
        self.history.append(overall_best_score)

        iter_count = 0
        inertia_weight = 0.9
        min_inertia = 0.4

        while iter_count < self.budget - self.population_size * swarm_count:
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            for i in range(swarm_count):
                cognitive_component = np.random.uniform(size=(self.population_size, self.dim)) * (personal_bests[i] - swarms[i])
                social_component = np.random.uniform(size=(self.population_size, self.dim)) * (global_bests[i] - swarms[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                swarms[i] += velocities[i]
                swarms[i] = np.clip(swarms[i], lb, ub)

                scores = np.array([func(ind) for ind in swarms[i]])
                iter_count += self.population_size

                better_indices = scores < personal_best_scores[i]
                personal_bests[i][better_indices] = swarms[i][better_indices]
                personal_best_scores[i][better_indices] = scores[better_indices]

                min_index = np.argmin(personal_best_scores[i])
                if personal_best_scores[i][min_index] < global_best_scores[i]:
                    global_bests[i] = personal_bests[i][min_index]
                    global_best_scores[i] = personal_best_scores[i][min_index]

                if personal_best_scores[i][min_index] < overall_best_score:
                    overall_best = personal_bests[i][min_index]
                    overall_best_score = personal_best_scores[i][min_index]

            if iter_count % (self.population_size * swarm_count * 5) == 0:
                best_global_best_index = np.argmin(global_best_scores)
                overall_best = global_bests[best_global_best_index]
                overall_best_score = global_best_scores[best_global_best_index]
                perturbation = np.random.normal(0, 0.1, size=(self.population_size, self.dim))
                swarms = [swarm + perturbation for swarm in swarms]

                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation_scale = 0.1 * (ub - lb)
                    perturbation = np.random.normal(0, perturbation_scale, size=(self.population_size, self.dim))
                    swarms = [swarm + perturbation for swarm in swarms]

                self.history.append(overall_best_score)

        return overall_best_score, overall_best