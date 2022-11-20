from cmath import log
import sympy as sp
from time import perf_counter_ns
from numba import njit


def results(title, interval_ns):
  print (f"{title}. {interval_ns * 1e-3:.2f} [us]\n")





def sympy_evaluation():
  x, y = sp.symbols("x y")
  expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)

  f1 = expr_sp.subs({x: 17, y: 42})
  print(f"result in {f1.evalf()}")

  start = perf_counter_ns()

  f1.evalf()
  f1.evalf()
  f1.evalf()
  f1.evalf()
  f1.evalf()
  f1.evalf()
  f1.evalf()
  f1.evalf()
  f1.evalf()
  f1.evalf()

  end = perf_counter_ns()
  results("sympy evalf", (end - start) / 10)


def lambda_evaluation():
  expr = lambda x, y : 3*x**2 + log(x**2 + y**2 + 1)
  print(f"result in {expr(17, 42)}")
  

  start = perf_counter_ns()

  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)

  end = perf_counter_ns()
  results("lambda evaluation", (end - start) / 10)


def lambdify_evaluation(module = "math"):
  x, y = sp.symbols("x y")
  expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)

  expr = sp.lambdify([x, y], expr_sp, module)
  print(f"result in {expr(17, 42)}")

  start = perf_counter_ns()

  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)

  end = perf_counter_ns()
  results(f"lambdify evaluation with {module}", (end - start) / 10)



def lambdify_jitted_evaluation(module = "math"):
  x, y = sp.symbols("x y")
  expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)

  expr = njit(sp.lambdify([x, y], expr_sp, module))
  print(f"result in {expr(17, 42)}")

  start = perf_counter_ns()

  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)
  expr(17, 42)

  end = perf_counter_ns()
  results(f"lambdify evaluation with numba. {module}", (end - start) / 10)



def lambdify_jitted_timeit(module = "math"):

  import timeit


  setup=f"""
import math
import sympy as sp
from numba import njit

x, y = sp.symbols("x y")
expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)

expr = njit(sp.lambdify([x, y], expr_sp, modules=[{module}]))
print("hola")
  """

  results = timeit.timeit("expr(17, 42)",setup=setup)
  print(results)

  #results(f"lambdify evaluation with numba and {module}", (end - start) / 10)




def better_timeit(module = "math"):
  x, y = sp.symbols("x y")
  expr_sp = 3*x**2 + sp.log(x**2 + y**2 + 1)

  expr = njit(sp.lambdify([x, y], expr_sp, module))
  print(f"result in {expr(17, 42)}")
  import timeit

  number = 1 << 20

  timeit_result = timeit.timeit("expr(17, 42)", globals=locals(), number=2**20)
  print(f"lambdify evaluation with numba and {module} took {timeit_result * 1e6 / number} [us]")


# sympy_evaluation()
# lambda_evaluation()
# lambdify_evaluation("math")
# lambdify_evaluation("numpy")
# lambdify_jitted_evaluation("math")
# lambdify_jitted_evaluation("numpy")

#lambdify_jitted_timeit()
better_timeit()
