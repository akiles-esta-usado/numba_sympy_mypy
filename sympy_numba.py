from math import log
import sympy as sp
from numba import njit
import timeit


def evaluation_us(stms: str, context: dict) -> float:

  number = 1_000_000

  results = min( timeit.repeat(stms, repeat=5, number=number, globals=context))

  # 1e6 evaluations over 1e6 convertion ratio to us resolves into 1.
  return results


def sympy_evaluation():
  x, y = sp.symbols("x y")
  expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)
  expr = expr_sp.subs({x: 17, y: 42})

  results = evaluation_us("expr.evalf()", locals())

  print(f"sympy evaluation: {expr.evalf():0.4f}, {results:0.3f} [us]")


def lambda_evaluation():
  expr = lambda x, y : 3*x**2 + log(x**2 + y**2 + 1)
  
  results = evaluation_us("expr(17, 42)", locals())

  print(f"lambda function: {expr(17, 42):0.4f}, {results:0.3f} [us]")


def lambdify_evaluation(module = "math"):
  x, y = sp.symbols("x y")
  expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)
  expr = sp.lambdify([x, y], expr_sp, module)

  results = evaluation_us("expr(17, 42)", locals())

  print(f"lambdify with {module}: {expr(17, 42):0.4f}, {results:0.3f} [us]")



def lambdify_jitted_evaluation(module = "math"):
  x, y = sp.symbols("x y")
  expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)
  expr = njit(sp.lambdify([x, y], expr_sp, module))

  results = evaluation_us("expr(17, 42)", locals())

  print(f"lambdify with numba and {module}: {expr(17, 42):0.4f}, {results:0.3f} [us]")


sympy_evaluation()
lambda_evaluation()
lambdify_evaluation("math")
lambdify_evaluation("numpy")
lambdify_jitted_evaluation("math")
lambdify_jitted_evaluation("numpy")
