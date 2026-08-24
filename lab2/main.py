import numpy as np
from scipy.linalg import svd, norm
from prettytable import PrettyTable



def create_matrix_B(n, p=4):
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            B[i, j] = (p + (i+1) + (j+1)) / ((i+1) + (j+1))
    return B



def DECOMP(A):
    
    n = len(A)
    LU = A.copy().astype(float)
    piv = np.arange(n)
    
    for k in range(n-1):
        # Поиск ведущего элемента
        max_val = abs(LU[k, k])
        max_row = k
        for i in range(k+1, n):
            if abs(LU[i, k]) > max_val:
                max_val = abs(LU[i, k])
                max_row = i
        
        # Перестановка строк
        if max_row != k:
            LU[[k, max_row]] = LU[[max_row, k]]
            piv[[k, max_row]] = piv[[max_row, k]]
        
        # Проверка на вырожденность
        if abs(LU[k, k]) < 1e-15:
            raise ValueError(f"Матрица вырождена на шаге {k}")
        
        # Исключение Гаусса
        for i in range(k+1, n):
            LU[i, k] = LU[i, k] / LU[k, k]
            for j in range(k+1, n):
                LU[i, j] = LU[i, j] - LU[i, k] * LU[k, j]
    
    return LU, piv



def SOLVE(LU, piv, b):
    
    n = len(LU)
    # Перестановка правой части
    x = b[piv].copy()
    
    # Прямой ход (решение Ly = b)
    for i in range(n):
        for j in range(i):
            x[i] -= LU[i, j] * x[j]
    
    # Обратный ход (решение Ux = y)
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i] -= LU[i, j] * x[j]
        x[i] /= LU[i, i]
    
    return x


def inverse_matrix(B):
    
    n = len(B)
    LU, piv = DECOMP(B)
    B_inv = np.zeros((n, n))
    
    # Решение для каждого столбца единичной матрицы
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        B_inv[:, j] = SOLVE(LU, piv, e_j)
    
    return B_inv



def matrix_cond(A):

    s = svd(A, compute_uv=False)
    if s.min() < 1e-15:
        return np.inf
    return s.max() / s.min()



def main():
    
    # Размерности матриц
    dimensions = [4, 6, 8, 10, 12]
    
    # Таблица для результатов
    table = PrettyTable()
    table.field_names = ["n", "cond(B)", "||R||"]
    
    print("\nРезультаты вычислений:")
    print("-" * 80)
    
    for n in dimensions:
        # Формирование матрицы B
        B = create_matrix_B(n)
        
        # Вычисление обратной матрицы
        B_inv = inverse_matrix(B)
        
        # Вычисление числа обусловленности
        cond_B = matrix_cond(B)
        
        # Вычисление матрицы невязки R = B * B^(-1) - E
        E = np.eye(n)
        R = B @ B_inv - E
        norm_R = norm(R, 'fro')
        
        # Добавление в таблицу
        table.add_row([
            n,
            f"{cond_B:.4e}",
            f"{norm_R:.4e}"
        ])
        
        print(f"n = {n}: cond(B) = {cond_B:.4e}, ||R|| = {norm_R:.4e}")
    
    # Вывод таблицы
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(table)
    
    
    print("\n" + "=" * 80)
    print("МАТРИЦЫ ДЛЯ n = 4")
    print("=" * 80)
    
    B4 = create_matrix_B(4)
    B4_inv = inverse_matrix(B4)
    
    print("\nИсходная матрица B (n=4):")
    print(B4)
    
    print("\nОбратная матрица B^(-1) (n=4):")
    print(B4_inv)
    
    print("\nПроверка: B * B^(-1) (должна быть единичной):")
    print(B4 @ B4_inv)
    
    # ========================================================================
    # 8. Анализ связи cond(B) и ||R||
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ СВЯЗИ ЧИСЛА ОБУСЛОВЛЕННОСТИ И НОРМЫ НЕВЯЗКИ")
    print("=" * 80)
    
    eps_mach = np.finfo(float).eps
    print(f"\nМашинная точность ε = {eps_mach:.2e}\n")
    
    for i, n in enumerate(dimensions):
        B = create_matrix_B(n)
        B_inv = inverse_matrix(B)
        cond_B = matrix_cond(B)
        E = np.eye(n)
        R = B @ B_inv - E
        norm_R = norm(R, 'fro')
        
        # Теоретическая оценка
        theoretical = cond_B * eps_mach
        
        print(f"n = {n}:")
        print(f"  cond(B) = {cond_B:.4e}")
        print(f"  ||R||    = {norm_R:.4e}")
        print(f"  cond(B) * ε = {theoretical:.4e}")
        print(f"  Отношение ||R|| / (cond(B)*ε) = {norm_R/theoretical:.2f}")
        print()
    
    # ========================================================================
    # 9. Выводы
    # ========================================================================
    
    print("=" * 80)
    print("ВЫВОДЫ")
    print("=" * 80)
    
    # Извлечение последних значений для выводов
    last_n = dimensions[-1]
    last_cond = matrix_cond(create_matrix_B(last_n))
    last_norm_R = norm(inverse_matrix(create_matrix_B(last_n)) @ create_matrix_B(last_n) - np.eye(last_n), 'fro')
    eps_mach = np.finfo(float).eps
    
    

if __name__ == "__main__":
    main()
