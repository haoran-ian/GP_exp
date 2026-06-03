# fmt: off
import os
import sys
import ioh
import types
import inspect
from ioh import ProblemClass, OptimizationType
# fmt: on


def class_code_to_ioh_problem(code_str: str, dim: int = 5, *,
                              name: str = None, instance: int = 1,
                              problem_class=ProblemClass.REAL,
                              optimization_type=OptimizationType.MIN,
                              lb: float = -5.0, ub: float = 5.0,):
    try:
        namespace = {}
        exec(code_str, namespace)
        classes = [
            v for v in namespace.values()
            if inspect.isclass(v)
            and v.__module__ in ("builtins", None) == False
        ]
        classes = [
            v for v in namespace.values()
            if inspect.isclass(v)
            and hasattr(v, "__init__")
            and hasattr(v, "f")
            and callable(getattr(v, "f"))
        ]
        if len(classes) == 0:
            raise ValueError("没有找到包含 f(self, x) 方法的 class。")
        if len(classes) > 1:
            names = [c.__name__ for c in classes]
            raise ValueError(f"找到多个候选 class：{names}，请确保代码字符串里只有一个问题类。")
        cls = classes[0]
        obj = cls(dim=dim)
        if not hasattr(obj, "f") or not callable(obj.f):
            raise TypeError(f"{cls.__name__} 的实例没有可调用的 f(x) 方法。")

        def objective(x):
            return float(obj.f(x))
        problem = ioh.wrap_problem(
            objective,
            name=name or cls.__name__,
            problem_class=problem_class,
            dimension=dim,
            instance=instance,
            optimization_type=optimization_type,
            lb=lb,
            ub=ub,
        )
        return problem
    except Exception:
        return None


def extract_llm_generated_problems():
    root_dir = "/data/hyin/GP_exp/XAI-liacs-LLaMEA-6d8b3c1"
    dirs = os.listdir(root_dir)
    exp_dirs = [exp_dir for exp_dir in dirs if exp_dir.startswith("exp-")]
    problems = []
    for exp_dir in exp_dirs:
        for code_path in os.listdir(f"{root_dir}/{exp_dir}/code/"):
            with open(f"{root_dir}/{exp_dir}/code/{code_path}", "r", encoding="utf-8") as f:
                code_str = f.read()
            problem = class_code_to_ioh_problem(code_str)
            if problem != None:
                problems += [problem]
    return problems
