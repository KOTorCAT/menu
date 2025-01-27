import math
import numpy as np


def calculate_integral_parabolic(func, a, b, n):
    h = (b - a) / n
    result = func(a) + func(b)
    for i in range(1, n, 2):
        result += 4 * func(a + i * h)
    for i in range(2, n - 1, 2):
        result += 2 * func(a + i * h)
    result *= h / 3
    return result


def calculate_integral_trapezoid(func, a, b, n):
    h = (b - a) / n
    result = func(a) + func(b)
    for i in range(1, n):
        result += 2 * func(a + i * h)
    result *= h / 2
    return result


def calculate_integral_right_rect(func, a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(1, n + 1):
        result += func(a + i * h)
    result *= h
    return result


def calculate_integral_left_rect(func, a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        result += func(a + i * h)
    result *= h
    return result


def calculate_integral_slow(func, a, b, tolerance):
    h_v = (b - a) / 2
    h_d = h_v
    h_s = h_v / 2
    log_1 = h_v * (func(a) + func(b))
    differ = 100

    while differ > tolerance:
        value = 0
        for i in range(int((a + h_s) / h_d), int((b - h_d - h_s) / h_d) + 1):
            value += func(a + h_s + i * h_d)

        log_2 = value * h_v
        differ = abs(log_2 - log_1)

        log_1 = log_2

        h_d = h_v
        h_s = h_v / 2
        h_v /= 2

    return log_2


def calculate_integral_fast(func, a, b, tolerance):
    h = (b - a) / 2
    log_1 = h * (func(a) + func(b))
    differ = 100
    while differ > tolerance:
        value = 0
        for i in range(int(a / h) + 1, int(b / h)):
            value += func(i * h)
        log_2 = value * h
        differ = abs(log_2 - log_1)
        log_1 = log_2
        h /= 2

    return log_2


import numpy as np


def euler_method_first_order(f, y0, t0, t_end, h):
    t = t0
    y = y0
    result = [(t, y)]

    while t < t_end:
        y = y + h * f(t, y)
        t += h
        result.append((t, y))

    return result


def runge_kutta_method_first_order(f, y0, t0, t_end, h):
    t = t0
    y = y0
    result = [(t, y)]

    while t <= t_end:
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)

        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        result.append((t, y))

    return result


def solve_second_order_ode_euler(f, y0, y_prime_0, t0, t_end, h):
    t = t0
    y = y0
    y_prime = y_prime_0
    results = [(t, y)]

    while t < t_end:
        y_next = y + h * y_prime
        y_prime_next = y_prime + h * f(t, y, y_prime)
        t += h
        y = y_next
        y_prime = y_prime_next
        results.append((t, y))
    return results


def solve_second_order_ode_runge_kutta(f, y0, y_prime_0, t0, t_end, h):
    t = t0
    y = y0
    y_prime = y_prime_0
    results = [(t, y)]

    while t < t_end:
        k1_y = h * y_prime
        k1_y_prime = h * f(t, y, y_prime)

        k2_y = h * (y_prime + k1_y_prime / 2)
        k2_y_prime = h * f(t + h / 2, y + k1_y / 2, y_prime + k1_y_prime / 2)

        k3_y = h * (y_prime + k2_y_prime / 2)
        k3_y_prime = h * f(t + h / 2, y + k2_y / 2, y_prime + k2_y_prime / 2)

        k4_y = h * (y_prime + k3_y_prime)
        k4_y_prime = h * f(t + h, y + k3_y, y_prime + k3_y_prime)

        y_next = y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        y_prime_next = y_prime + (k1_y_prime + 2 * k2_y_prime + 2 * k3_y_prime + k4_y_prime) / 6

        t += h
        y = y_next
        y_prime = y_prime_next
        results.append((t, y))
    return results


def elementary_function_iteration(func_name, x, tolerance, max_iterations=100):
    func_name = func_name.lower()
    try:
        if func_name == "sin":
            iter_func = lambda x0: np.arcsin(x)
        elif func_name == "cos":
            iter_func = lambda x0: np.arccos(x)
        elif func_name == "exp":
            iter_func = lambda x0: np.log(x)
        elif func_name == "log":
            if x <= 0:
                raise ValueError("Логарифм определён только для положительных чисел.")
            return np.log(x)  # Прямое вычисление для log
        elif func_name == "tan":
            iter_func = lambda x0: np.arctan(x)
        elif func_name == "sinh":
            iter_func = lambda x0: np.arcsinh(x)
        elif func_name == "cosh":
            if x < 1:
                raise ValueError("Обратный гиперболический косинус определён только для чисел >=1")
            return np.arccosh(x)  # Прямое вычисление для cosh
        elif func_name == "tanh":
            if abs(x) >= 1:
                raise ValueError(
                    "Обратный гиперболический тангенс определён только для чисел с абсолютным значением <1")
            return np.arctanh(x)  # Прямое вычисление для tanh
        elif func_name == "sqrt":
            iter_func = lambda x0: 0.5 * (x0 + x / x0)
        elif func_name == "arcsin":
            iter_func = lambda x0: np.sin(x)
        elif func_name == "arccos":
            iter_func = lambda x0: np.cos(x)
        elif func_name == "arctan":
            iter_func = lambda x0: np.tan(x)
        else:
            raise ValueError("Неизвестная функция")

        initial_guess = better_starting_point(func_name, x)
        result = iterate(iter_func, initial_guess, tolerance, max_iterations)
        if result is None:
            raise ValueError("Метод не сошелся.")
        return result

    except (ValueError, TypeError) as e:
        return f"Ошибка: {e}"


def iterate(func, x0, tolerance, max_iterations=100):
    x1 = func(x0)
    for i in range(max_iterations):
        if abs(x1 - x0) < tolerance:
            return x1
        if np.isnan(x1) or np.isinf(x1):
            return None
        x0 = x1
        x1 = func(x0)
    # Добавлено: возвращаем лучшее приближение, если метод не сошелся за max_iterations
    return x1  # Возвращаем последнее значение


def elementary_function_chebyshev(func_name, x, tolerance):
    func_name = func_name.lower()
    if func_name == "sin":
        return np.sin(x)
    elif func_name == "cos":
        return np.cos(x)
    elif func_name == "exp":
        return np.exp(x)
    elif func_name == "log":
        return np.log(x)
    elif func_name == "tan":
        return np.tan(x)
    elif func_name == "sinh":
        return np.sinh(x)
    elif func_name == "cosh":
        return np.cosh(x)
    elif func_name == "tanh":
        return np.tanh(x)
    elif func_name == "sqrt":
        return np.sqrt(x)
    elif func_name == "arcsin":
        return np.arcsin(x)
    elif func_name == "arccos":
        return np.arccos(x)
    elif func_name == "arctan":
        return np.arctan(x)

    else:
        raise ValueError("Неизвестная функция")


def iterate(func, x0, tolerance, max_iterations=100):
    x1 = func(x0)
    for i in range(max_iterations):
        if abs(x1 - x0) < tolerance:
            return x1
        if np.isnan(x1) or np.isinf(x1):
            return None
        x0 = x1
        x1 = func(x0)
    return None


def better_starting_point(func_name, x):
    if func_name in ["sin", "cos", "tan", "sinh", "cosh", "tanh"]:
        return x
    elif func_name == "exp":
        return x > 0 and x or 1
    elif func_name == "log":
        return x > 0 and x or 1
    elif func_name == "sqrt":
        return x > 0 and x or 1
    elif func_name in ["arcsin", "arccos", "arctan"]:
        return x
    else:
        return x


def system_of_equations(t, y, equations):
    dydt = np.zeros_like(y)
    for i, eq in enumerate(equations):
        dydt[i] = eq(t, y)
    return dydt


def solve_system_euler(f, y0, t0, t_end, h):
    try:
        t = t0
        y = np.array(y0, dtype=float)
        results = [(t, y.copy())]
        while t < t_end:
            y = y + h * np.array(f(t, y))
            t = t + h
            results.append((t, y.copy()))
        return results
    except Exception as e:
        print(f"Ошибка в методе Эйлера для системы ОДУ: {e}")
        return None


def newton_method(func, deriv, x0, tolerance, max_iterations):
    x = x0
    for i in range(max_iterations):
        x_next = x - func(x) / deriv(x)
        if abs(x_next - x) < tolerance:
            return x_next
        x = x_next
    return None


def secant_method(func, x0, x1, tolerance, max_iterations):
    x_prev = x0
    x = x1
    for i in range(max_iterations):
        x_next = x - func(x) * (x - x_prev) / (func(x) - func(x_prev))
        if abs(x_next - x) < tolerance:
            return x_next
        x_prev = x
        x = x_next
    return None


def dichotomy_method(func, a, b, tolerance, max_iterations):
    if func(a) * func(b) >= 0:
        raise ValueError("Функция должна менять знак на отрезке [a, b].")

    for i in range(max_iterations):
        c = (a + b) / 2
        if func(c) == 0 or (b - a) / 2 < tolerance:
            return c
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2
