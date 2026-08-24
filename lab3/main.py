import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')  # Игнорируем предупреждения

# ============================================================
# 1. Приведение к системе ДУ первого порядка
# ============================================================
def system(t, y):
    """
    Система ДУ:
    y[0]' = y[1]
    y[1]' = (2t/(t²-1))*y[1] - (2/(t²-1))*y[0]
    """
    y1, y2 = y
    
    # При t = 0 используем предельный переход
    if abs(t) < 1e-10:
        return [y2, 0.0]
    
    # Защита от деления на ноль
    denom = t**2 - 1
    if abs(denom) < 1e-8:
        # Вблизи особой точки используем приближенные значения
        # При t → 1, решение стремится к y = t, y' = 1
        return [y2, 0.0]
    
    dy1 = y2
    dy2 = (2*t/denom)*y2 - (2/denom)*y1
    return [dy1, dy2]

# Точное решение
def exact_solution(t):
    return t

# Начальные условия
t0 = 0.0
y0 = [0.0, 1.0]  # y(0)=0, y'(0)=1
t_end = 0.9  # интервал интегрирования (избегаем t=1)
t_span = (t0, t_end)

# ============================================================
# 1) Решение с помощью RKF45
# ============================================================
print("=" * 60)
print("1) Решение методом RKF45")
print("=" * 60)

eps = 5e-5
h_print = 0.1
t_print = np.arange(t0, t_end + h_print/2, h_print)

try:
    solution_rkf45 = solve_ivp(
        system, 
        t_span, 
        y0,
        method='RK45',
        rtol=eps,
        atol=eps,
        t_eval=t_print,
        max_step=0.1
    )
    
    print(f"Использованная погрешность EPS = {eps}")
    print("\nРезультаты RKF45:")
    print("     t        y_RKF45      y_точное     Погрешность")
    print("-" * 55)
    for i, t in enumerate(solution_rkf45.t):
        y_rkf45 = solution_rkf45.y[0, i]
        y_exact = exact_solution(t)
        error = abs(y_rkf45 - y_exact)
        print(f"{t:8.4f}  {y_rkf45:10.6f}  {y_exact:10.6f}  {error:10.2e}")
        
except Exception as e:
    print(f"Ошибка: {e}")
    # Используем odeint как запасной вариант
    from scipy.integrate import odeint
    
    def system_odeint(y, t):
        return system(t, y)
    
    t_solve = np.linspace(t0, t_end, 100)
    solution_odeint = odeint(system_odeint, y0, t_solve)
    
    print("\nРезультаты (используя odeint):")
    print("     t        y_RKF45      y_точное     Погрешность")
    print("-" * 55)
    for i in range(0, len(t_solve), 10):
        t = t_solve[i]
        y_num = solution_odeint[i, 0]
        y_exact = exact_solution(t)
        error = abs(y_num - y_exact)
        print(f"{t:8.4f}  {y_num:10.6f}  {y_exact:10.6f}  {error:10.2e}")
    
    solution_rkf45 = type('obj', (object,), {'t': t_solve, 'y': [solution_odeint[:, 0]]})()

# ============================================================
# 5) Неявный метод трапеций
# ============================================================
print("\n" + "=" * 60)
print("5) Решение неявным методом трапеций")
print("=" * 60)

def system_for_trapezoidal(t, y):
    """Безопасная версия системы для метода трапеций"""
    y1, y2 = y
    
    # При t = 0
    if abs(t) < 1e-10:
        return np.array([y2, 0.0])
    
    denom = t**2 - 1
    if abs(denom) < 1e-8:
        return np.array([y2, 0.0])
    
    dy1 = y2
    dy2 = (2*t/denom)*y2 - (2/denom)*y1
    return np.array([dy1, dy2])

def trapezoidal_method(system_func, t0, y0, t_end, h, max_iter=50, tol=1e-10):
    """
    Неявный метод трапеций для системы ДУ.
    """
    n_steps = int(np.ceil((t_end - t0) / h))
    t_values = [t0]
    y_values = [np.array(y0, dtype=float)]
    
    t_current = t0
    y_current = np.array(y0, dtype=float)
    
    for step in range(n_steps):
        t_next = min(t_current + h, t_end)
        h_actual = t_next - t_current
        
        if h_actual < 1e-12:
            break
            
        # Начальное приближение - метод Эйлера
        f_current = system_func(t_current, y_current)
        y_next = y_current + h_actual * f_current
        
        # Итерации для решения нелинейного уравнения
        for iter_count in range(max_iter):
            f_next = system_func(t_next, y_next)
            y_new = y_current + h_actual/2 * (f_current + f_next)
            
            diff = np.linalg.norm(y_new - y_next)
            if diff < tol:
                y_next = y_new
                break
            y_next = y_new
        else:
            if step == 0:
                print(f"  Предупреждение: слабая сходимость на шаге {step+1}")
        
        t_values.append(t_next)
        y_values.append(y_next)
        t_current = t_next
        y_current = y_next
    
    return np.array(t_values), np.array(y_values).T

# Исследование влияния шага интегрирования
print("\nИсследование влияния шага интегрирования:")
print("-" * 70)
print("h_int      Локальная погрешность    Глобальная погрешность")
print("-" * 70)

h_values = [0.1, 0.05, 0.025, 0.0125]
results = {}

for h in h_values:
    try:
        t_vals, y_vals = trapezoidal_method(system_for_trapezoidal, t0, y0, t_end, h)
        
        # Локальная погрешность на первом шаге
        if len(t_vals) > 1:
            y1_first_step = y_vals[0, 1]
            t_first = t_vals[1]
            y_exact_first = exact_solution(t_first)
            local_error = abs(y1_first_step - y_exact_first)
        else:
            local_error = 1e-20  # маленькое значение вместо нуля
        
        # Глобальная погрешность в конце интервала
        y_last = y_vals[0, -1]
        y_exact_last = exact_solution(t_vals[-1])
        global_error = abs(y_last - y_exact_last)
        
        results[h] = (t_vals, y_vals, max(local_error, 1e-20), max(global_error, 1e-20))
        
        print(f"{h:8.4f}    {local_error:18.6e}    {global_error:18.6e}")
        
    except Exception as e:
        print(f"{h:8.4f}    Ошибка: {str(e)[:40]}")
        results[h] = None

# Вывод результатов для h = 0.1
h_main = 0.1
if h_main in results and results[h_main] is not None:
    t_vals, y_vals, _, _ = results[h_main]
    
    print(f"\nРезультаты неявного метода трапеций (h = {h_main}):")
    print("     t        y_трапеции    y_точное     Погрешность")
    print("-" * 55)
    for i in range(len(t_vals)):
        y_trap = y_vals[0, i]
        y_exact = exact_solution(t_vals[i])
        error = abs(y_trap - y_exact)
        print(f"{t_vals[i]:8.4f}  {y_trap:10.6f}  {y_exact:10.6f}  {error:10.2e}")

# ============================================================
# Построение графиков
# ============================================================
plt.figure(figsize=(12, 10))

# График 1: Сравнение решений
plt.subplot(2, 2, 1)
t_fine = np.linspace(t0, t_end, 200)
y_exact_fine = exact_solution(t_fine)

plt.plot(t_fine, y_exact_fine, 'k-', linewidth=2, label='Точное решение $y=t$')

if 'solution_rkf45' in locals():
    plt.plot(solution_rkf45.t, solution_rkf45.y[0], 'ro', markersize=4, label='RKF45 (EPS=5e-5)')

colors = ['b', 'g', 'c', 'm']
for h, color in zip(h_values, colors):
    if h in results and results[h] is not None:
        t_vals, y_vals, _, _ = results[h]
        plt.plot(t_vals, y_vals[0], 'o-', color=color, markersize=3, 
                label=f'Трапеции h={h}')

plt.xlabel('t')
plt.ylabel('y')
plt.title('Сравнение численных методов с точным решением')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# График 2: Погрешности методов
plt.subplot(2, 2, 2)

if 'solution_rkf45' in locals():
    error_rkf45 = abs(solution_rkf45.y[0] - exact_solution(solution_rkf45.t))
    # Фильтруем нулевые значения для логарифмического масштаба
    error_rkf45_nonzero = np.maximum(error_rkf45, 1e-20)
    plt.semilogy(solution_rkf45.t, error_rkf45_nonzero, 'ro-', markersize=4, label='RKF45')

for h, color in zip(h_values, colors):
    if h in results and results[h] is not None:
        t_vals, y_vals, _, _ = results[h]
        error_trap = abs(y_vals[0] - exact_solution(t_vals))
        error_trap_nonzero = np.maximum(error_trap, 1e-20)
        plt.semilogy(t_vals, error_trap_nonzero, 'o-', color=color, markersize=3, 
                    label=f'Трапеции h={h}')

plt.xlabel('t')
plt.ylabel('Погрешность')
plt.title('Сравнение погрешностей методов')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# График 3: Зависимость погрешности от шага (в конце интервала)
plt.subplot(2, 2, 3)

h_plot = []
global_errors_plot = []
for h in h_values:
    if h in results and results[h] is not None:
        _, _, _, global_error = results[h]
        if global_error > 1e-20:  # Только положительные значения
            h_plot.append(h)
            global_errors_plot.append(global_error)

if len(h_plot) >= 2:
    plt.loglog(h_plot, global_errors_plot, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Шаг интегрирования h')
    plt.ylabel('Глобальная погрешность')
    plt.title('Зависимость глобальной погрешности от шага')
    plt.grid(True, alpha=0.3)
    
    # Линия для сравнения порядка точности
    h_ref = np.array(h_plot)
    expected = global_errors_plot[0] * (h_ref/h_ref[0])**2
    plt.loglog(h_ref, expected, 'r--', label='Порядок 2 (теоретический)')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Недостаточно данных\nдля построения графика', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Шаг интегрирования h')
    plt.ylabel('Глобальная погрешность')
    plt.title('Зависимость глобальной погрешности от шага')
    plt.grid(True, alpha=0.3)

# График 4: Локальная погрешность (первый шаг)
plt.subplot(2, 2, 4)

h_plot_local = []
local_errors_plot = []
for h in h_values:
    if h in results and results[h] is not None:
        _, _, local_error, _ = results[h]
        if local_error > 1e-20:
            h_plot_local.append(h)
            local_errors_plot.append(local_error)

if len(h_plot_local) >= 2:
    plt.loglog(h_plot_local, local_errors_plot, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Шаг интегрирования h')
    plt.ylabel('Локальная погрешность (1-й шаг)')
    plt.title('Зависимость локальной погрешности от шага')
    plt.grid(True, alpha=0.3)
    
    # Линия для сравнения порядка точности
    h_ref = np.array(h_plot_local)
    expected_local = local_errors_plot[0] * (h_ref/h_ref[0])**3
    plt.loglog(h_ref, expected_local, 'r--', label='Порядок 3 (теоретический)')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Недостаточно данных\nдля построения графика', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Шаг интегрирования h')
    plt.ylabel('Локальная погрешность (1-й шаг)')
    plt.title('Зависимость локальной погрешности от шага')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Анализ порядка точности
# ============================================================
print("\n" + "=" * 60)
print("Анализ порядка точности неявного метода трапеций")
print("=" * 60)

valid_h = []
valid_errors = []
for h in h_values:
    if h in results and results[h] is not None:
        _, _, _, global_error = results[h]
        if global_error > 1e-20:
            valid_h.append(h)
            valid_errors.append(global_error)

if len(valid_h) >= 2:
    print("\nОценка порядка точности (по глобальной погрешности):")
    for i in range(len(valid_h) - 1):
        p = np.log(valid_errors[i+1] / valid_errors[i]) / np.log(valid_h[i+1] / valid_h[i])
        print(f"h: {valid_h[i]:.4f} -> {valid_h[i+1]:.4f}, порядок: {p:.2f}")
    
    # Средний порядок
    orders = []
    for i in range(len(valid_h) - 1):
        p = np.log(valid_errors[i+1] / valid_errors[i]) / np.log(valid_h[i+1] / valid_h[i])
        orders.append(p)
    avg_order = np.mean(orders)
    print(f"\nСредний порядок точности: {avg_order:.2f}")
    print("Теоретический порядок точности метода трапеций: 2")
    print(f"Отклонение от теоретического: {abs(avg_order - 2):.3f}")
else:
    print("Недостаточно данных для оценки порядка точности")

# ============================================================
# Таблица сравнения методов
# ============================================================
print("\n" + "=" * 60)
print("Сравнение методов в точке t = 0.9")
print("=" * 60)

y_exact_end = exact_solution(t_end)

# RKF45
if 'solution_rkf45' in locals():
    y_rkf45_end = solution_rkf45.y[0, -1]
    error_rkf45_end = abs(y_rkf45_end - y_exact_end)
    print(f"RKF45 (EPS={eps}):          y={y_rkf45_end:.8f}, погрешность={error_rkf45_end:.2e}")

# Метод трапеций для разных шагов
for h in h_values:
    if h in results and results[h] is not None:
        t_vals, y_vals, _, _ = results[h]
        y_trap_end = y_vals[0, -1]
        error_trap_end = abs(y_trap_end - y_exact_end)
        print(f"Метод трапеций (h={h}):    y={y_trap_end:.8f}, погрешность={error_trap_end:.2e}")

print(f"\nТочное решение: y({t_end}) = {y_exact_end}")
