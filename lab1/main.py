import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import eval_laguerre  
from numpy.polynomial import Polynomial
from prettytable import PrettyTable


def integrand(t):
    if np.isclose(t, 0):
        return 0.0
    return (1 - np.cos(t)) / t

def f_quad(x, epsabs=1e-12, epsrel=1e-12):
    if x == 0:
        return 0.0
    res, _ = quad(integrand, 0, x, epsabs=epsabs, epsrel=epsrel)
    return res


x_vals = np.arange(2, 3.1, 0.1)
f_vals = [f_quad(x) for x in x_vals]


cs = CubicSpline(x_vals, f_vals, bc_type='natural')

coeffs = np.polyfit(x_vals, f_vals, deg=10)  
lagrange_poly = Polynomial(coeffs[::-1])     


x_compare = np.arange(2.05, 3.0, 0.1) 
f_exact = [f_quad(x, epsabs=1e-14, epsrel=1e-14) for x in x_compare]
f_spline = cs(x_compare)
f_lagrange = lagrange_poly(x_compare)


table = PrettyTable()
table.field_names = ["x", "Точное f(x)", "Сплайн", "Лагранж", "Ошибка сплайна", "Ошибка Лагранжа"]
for i, x in enumerate(x_compare):
    err_spline = abs(f_spline[i] - f_exact[i])
    err_lagrange = abs(f_lagrange[i] - f_exact[i])
    table.add_row([
        round(x, 2),
        round(f_exact[i], 10),
        round(f_spline[i], 10),
        round(f_lagrange[i], 10),
        f"{err_spline:.2e}",
        f"{err_lagrange:.2e}"
    ])
print(table)


print("\n" + "="*60)
print("Исследование влияния особенности в t=0 при интегрировании QUANC8")
print("="*60)

def f_quad_broken(x, epsabs=1e-12, epsrel=1e-12):
    res, _ = quad(lambda t: (1 - np.cos(t))/t, 0, x, epsabs=epsabs, epsrel=epsrel)
    return res

def f_quad_split(x, epsilon, epsabs=1e-12, epsrel=1e-12):
    res1, _ = quad(lambda t: (1 - np.cos(t))/t, 0, epsilon, epsabs=epsabs, epsrel=epsrel)
    res2, _ = quad(lambda t: (1 - np.cos(t))/t, epsilon, x, epsabs=epsabs, epsrel=epsrel)
    return res1 + res2

x_test = 2.5
epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

print(f"\nСравнение для x = {x_test}:")
print(f"Точное значение (quad с высокой точностью): {f_quad(x_test):.12f}")

ref = f_quad(x_test)  # эталон

print("\nОбычное интегрирование (без разбиения):")
val_broken = f_quad_broken(x_test)
print(f"  Значение: {val_broken:.12f}, Ошибка: {abs(val_broken - ref):.2e}")

print("\nС разбиением на [0, epsilon] + [epsilon, x]:")
for eps in epsilons:
    val_split = f_quad_split(x_test, eps)
    err = abs(val_split - ref)
    print(f"  epsilon = {eps:.0e}, значение = {val_split:.12f}, ошибка = {err:.2e}")

plt.figure(figsize=(12, 6))
plt.plot(x_vals, f_vals, 'ko', label='Узлы интерполяции')
x_plot = np.linspace(2, 3, 200)
plt.plot(x_plot, cs(x_plot), 'b-', label='Кубический сплайн')
plt.plot(x_plot, lagrange_poly(x_plot), 'r--', label='Полином Лагранжа 10-й степени')
plt.plot(x_compare, f_exact, 'gx', label='Точные значения (сравнение)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Аппроксимация функции f(x)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 5))
plt.semilogy(x_compare, np.abs(f_spline - f_exact), 'b-o', label='Ошибка сплайна')
plt.semilogy(x_compare, np.abs(f_lagrange - f_exact), 'r-s', label='Ошибка полинома Лагранжа')
plt.xlabel('x')
plt.ylabel('Абсолютная ошибка')
plt.title('Сравнение ошибок интерполяции')
plt.legend()
plt.grid(True)
plt.show()
