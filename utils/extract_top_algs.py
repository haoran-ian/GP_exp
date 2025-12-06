# fmt: off
import os
import json
import numpy as np
from LLaMEA.llamea.solution import Solution
# fmt: on


def nbest_algs_id(LLaMEA_exp_path: str, nbest: int = 1):
    fitness_record = []
    log_path = os.path.join(LLaMEA_exp_path, "log.jsonl")
    f = open(log_path, "r")
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        content = json.loads(line[:-1])
        fitness_record += [[i, float(content["fitness"])]]
    fitness_record = np.array(fitness_record)
    result = fitness_record[np.argsort(-fitness_record[:, 1])
                            [:nbest], 0].astype(int)
    f.close()
    return result


def extract_top_algs(LLaMEA_exp_path: str, nbest: int = 1):
    solutions = []
    log_path = os.path.join(LLaMEA_exp_path, "log.jsonl")
    f = open(log_path, "r")
    lines = f.readlines()
    alg_ids = nbest_algs_id(LLaMEA_exp_path, nbest)
    for alg_id in alg_ids:
        line = lines[alg_id]
        content = json.loads(line[:-1])
        solutions += [Solution(code=content["code"], name=content["name"])]
    f.close()
    return solutions
