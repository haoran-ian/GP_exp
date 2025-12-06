# fmt: off
import os
import sys
import ioh
sys.path.insert(0, os.getcwd())
from problems.fluid_dynamics.problem import get_pipes_topology_problem
from utils.extract_top_algs import extract_top_algs
# fmt: on


def benchmark_alg(solution, problem, budget, exp_name, runs=20):
    code = solution.code
    algorithm_name = solution.name
    dim = problem.meta_data.n_variables
    exec(code, globals())
    l1 = ioh.logger.Analyzer(
        folder_name=f"data/benchmark_algs/{exp_name}",
        algorithm_name=algorithm_name,
        triggers=[ioh.logger.trigger.ALWAYS],
        store_positions=True
    )
    problem.attach_logger(l1)
    problem.reset()
    l1.reset()
    for _ in range(runs):
        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
        algorithm(problem)
        problem.reset()
        l1.reset()
    l1.close()


if __name__ == "__main__":
    budget_cof = 10
    problem_name = "fluid_dynamics_3pipes_iid0"
    exp_name = f"{problem_name}/gp_func_fluid_dynamics_3pipes_iid0_100xD"
    problem = get_pipes_topology_problem(iid=0, num_pipes=3)
    dim = problem.meta_data.n_variables
    if not os.path.exists(f"data/benchmark_algs/{problem_name}"):
        os.mkdir(f"data/benchmark_algs/{problem_name}")
    # exp_paths = [
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_144502-LLaMEA-gpt-4o-BBOB_23D_10xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_145645-LLaMEA-gpt-4o-BBOB_23D_10xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_151137-LLaMEA-gpt-4o-BBOB_23D_10xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_152244-LLaMEA-gpt-4o-BBOB_23D_10xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_153656-LLaMEA-gpt-4o-BBOB_23D_10xD",
    # ]
    # exp_paths = [
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_164808-LLaMEA-gpt-4o-BBOB_23D_100xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_172231-LLaMEA-gpt-4o-BBOB_23D_100xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-05_174743-LLaMEA-gpt-4o-BBOB_23D_100xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-06_013027-LLaMEA-gpt-4o-BBOB_23D_100xD",
    #     "data/LLaMEA_exp/LLaMEA_BBOB/exp-12-06_020115-LLaMEA-gpt-4o-BBOB_23D_100xD",
    # ]
    # exp_paths = [
    #     "data/LLaMEA_exp/fluid_dynamics/exp-12-05_164348-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_10xD",
    #     "data/LLaMEA_exp/fluid_dynamics/exp-12-05_170250-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_10xD",
    #     "data/LLaMEA_exp/fluid_dynamics/exp-12-05_171530-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_10xD",
    #     "data/LLaMEA_exp/fluid_dynamics/exp-12-05_173408-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_10xD",
    #     "data/LLaMEA_exp/fluid_dynamics/exp-12-05_175010-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_10xD",
    # ]
    exp_paths = [
        "data/LLaMEA_exp/fluid_dynamics/exp-12-05_164716-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_100xD",
        "data/LLaMEA_exp/fluid_dynamics/exp-12-05_172029-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_100xD",
        "data/LLaMEA_exp/fluid_dynamics/exp-12-05_180009-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_100xD",
        "data/LLaMEA_exp/fluid_dynamics/exp-12-05_182943-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_100xD",
        "data/LLaMEA_exp/fluid_dynamics/exp-12-05_185757-LLaMEA-gpt-4o-gp_func_fluid_dynamics_3pipes_iid0_100xD",
    ]
    for i in range(len(exp_paths)):
        exp_path = exp_paths[i]
        solutions = extract_top_algs(exp_path)
        for j in range(len(solutions)):
            exp_name_detailed = f"{exp_name}_run{i}_best{j}"
            solution = solutions[j]
            benchmark_alg(solution=solution, problem=problem,
                          budget=budget_cof*dim, exp_name=exp_name_detailed)
