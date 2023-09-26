import sympy as sp


x = sp.symbols("x")
f_x_str = input("Function: ").strip()
f_x = sp.sympify(f_x_str)

diff = sp.diff(f_x)

lim = sp.limit(f_x , x , 0)

print(f_x)
print(f"\n{diff}\n")
print(lim)