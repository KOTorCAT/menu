import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
from functools import partial
import Function
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MenuApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_method = None

    def initUI(self):
        self.setWindowTitle("Меню :)")
        self.setGeometry(100, 100, 400, 400)
        self.examples = {
            "Элементарные функции": {
                "Метод итераций": [
                    {
                        "input": {
                            "x": 14.76,
                            "func_name": "sqrt",
                            "y0": 3.8,
                            "tolerance": 0.0000002
                        },
                        "expected": 3.841874542459709,
                        "history_text": "x=14.76, func_name=sqrt, y0=3.8, tolerance=0.0000002"
                    },
                    {
                        "input": {
                            "x": 0.142,
                            "func_name": "sqrt",
                            "y0": 0.4,
                            "tolerance": 0.000000006
                        },
                        "expected": 0.3768288736283354,
                        "history_text": "x=0.142, func_name=sqrt, y0=0.4, tolerance=0.000000006"
                    },
                    {
                        "input": {
                            "x": 17.32,
                            "func_name": "1/sqrt",
                            "y0": 0.24,
                            "tolerance": 0.0000002
                        },
                        "expected": 0.24028466566610637,
                        "history_text": "x=17.32, func_name=1/sqrt, y0=0.24, tolerance=0.0000002"
                    },
                    {
                        "input": {
                            "x": 0.464,
                            "func_name": "1/sqrt",
                            "y0": 1.5,
                            "tolerance": 0.000000006
                        },
                        "expected": 1.4680505487867588,
                        "history_text": "x=0.464, func_name=1/sqrt, y0=1.5, tolerance=0.000000006"
                    }
                ],
                "Метод Чебышева": [
                    {
                        "input": {
                            "x": np.pi / 2,
                            "func_name": "sin",
                            "tolerance": 0.0000000000001
                        },
                        "expected": 0.999999988460539,
                        "history_text": "x=pi/2, func_name=sin, tolerance=0.0000000000001"
                    },
                    {
                        "input": {
                            "x": 0,
                            "func_name": "exp",
                            "tolerance": 0.0000002
                        },
                        "expected": 2.718281600000000,
                        "history_text": "x=0, func_name=exp, tolerance=0.0000002"
                    }
                ]
            }
        }
        self.setStyleSheet("""
    QMainWindow {
        background-color: #2E2E2E;
        font-family: sans-serif;
        color: #eee;
    }
    QPushButton {
        background-color: #444;
        color: #eee;
        border: 2px solid #555;
        border-radius: 20px;
        padding: 12px 20px;
        font-size: 18px;
        margin: 8px;
        text-align: left;
        transition: background-color 0.3s;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
    }
    QPushButton:hover {
        background-color: #555;
        transform: scale(1.05);
    }
    QPushButton:pressed {
        background-color: #666;
    }
    QStackedWidget {
        background-color: #2E2E2E;
    }
    QLabel {
        font-size: 16px;
        color: white;
        margin-bottom: 5px;
    }
    QLineEdit {
        font-size: 16px;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #888;
        background-color: #444;
        color: white;
        margin-bottom: 5px;
        width: 300px;
    }
    QDialog {
        background-color: #333;
        font-family: sans-serif;
        color: #eee;
    }
    QGridLayout, QVBoxLayout {
        margin: 20px;
        spacing: 15px;
    }
    QTextEdit {
        font-size: 14px;
        color: #fff;
        background-color: #444;
        border: 1px solid #888;
        border-radius: 5px;
        font-family: monospace;
    }
    QDialog {
        background-color: #333;
    }
        """)

        self.central_widget = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_menu = self.create_menu([
            ("Интегрирование", self.show_method_selection_menu),
            ("Дифференциальные уравнения", self.show_diff_eq_menu),
            ("Элементарные функции", self.show_elementary_functions_menu),
            ("Нелинейные уравнения", self.show_nonlinear_equations_menu),
            ("Выход", self.close)
        ])

        self.method_selection_menu = self.create_menu([
            ("Интегралы с постоянным шагом", self.show_integral_menu),
            ("Интегралы с переменным шагом", self.show_variable_step_menu),
            ("Назад", self.show_main_menu)
        ])

        self.integral_menu = self.create_menu([
            ("Метод парабол", partial(self.show_input_dialog, "Метод парабол")),
            ("Метод трапеций", partial(self.show_input_dialog, "Метод трапеций")),
            ("Метод прямоугольников (правых частей)",
             partial(self.show_input_dialog, "Метод прямоугольников (правых частей)")),
            ("Метод прямоугольников (левых частей)",
             partial(self.show_input_dialog, "Метод прямоугольников (левых частей)")),
            ("Назад", self.show_method_selection_menu)
        ])

        self.variable_step_menu = self.create_menu([
            ("Метод трапеций с переменным шагом", partial(self.show_input_dialog, "Метод трапеций с переменным шагом")),
            ("Двойной пересчёт", partial(self.show_input_dialog, "Двойной пересчёт")),
            ("Назад", self.show_method_selection_menu)
        ])

        self.diff_eq_menu = self.create_menu([
            ("Метод Эйлера", partial(self.show_diff_eq_input_dialog, "Метод Эйлера")),
            ("Метод Рунге-Кутты",
             partial(self.show_diff_eq_input_dialog, "Метод Рунге-Кутты")),
            ("Система дифференциальных уравнений",
             partial(self.show_diff_eq_input_dialog, "Система дифференциальных уравнений")),
            ("Контрольный пример 1",
             partial(self.show_diff_eq_input_dialog, "Контрольный пример 1")),
            ("Контрольный пример 3",
             partial(self.show_diff_eq_input_dialog, "Контрольный пример 3")),
            ("Контрольный пример 2",
             partial(self.show_diff_eq_input_dialog, "Контрольный пример 2")),
            ("Назад", self.show_main_menu)
        ])

        self.nonlinear_equations_menu = self.create_menu([
            ("Метод касательных", partial(self.show_nonlinear_input_dialog, "Метод касательных")),
            ("Метод хорд", partial(self.show_nonlinear_input_dialog, "Метод хорд")),
            ("Метод дихотомии", partial(self.show_nonlinear_input_dialog, "Метод дихотомии")),
            ("Назад", self.show_main_menu)
        ])

        self.elementary_functions_menu = self.create_menu([
            ("Метод итераций", partial(self.show_elementary_functions_input_dialog, "Метод итераций")),
            ("Метод Чебышева", partial(self.show_elementary_functions_input_dialog, "Метод Чебышева")),
            ("Назад", self.show_main_menu)
        ])

        self.central_widget.addWidget(self.main_menu)
        self.central_widget.addWidget(self.method_selection_menu)
        self.central_widget.addWidget(self.integral_menu)
        self.central_widget.addWidget(self.variable_step_menu)
        self.central_widget.addWidget(self.diff_eq_menu)
        self.central_widget.addWidget(self.nonlinear_equations_menu)
        self.central_widget.addWidget(self.elementary_functions_menu)
        self.show_main_menu()

    def create_menu(self, button_data):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        for text, command in button_data:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(command)
            layout.addWidget(button)
        return widget

    def show_main_menu(self):
        self.central_widget.setCurrentWidget(self.main_menu)

    def show_method_selection_menu(self):
        self.central_widget.setCurrentWidget(self.method_selection_menu)

    def show_integral_menu(self):
        self.central_widget.setCurrentWidget(self.integral_menu)

    def show_variable_step_menu(self):
        self.central_widget.setCurrentWidget(self.variable_step_menu)

    def show_diff_eq_menu(self):
        self.central_widget.setCurrentWidget(self.diff_eq_menu)

    def show_nonlinear_equations_menu(self):
        self.central_widget.setCurrentWidget(self.nonlinear_equations_menu)

    def show_elementary_functions_menu(self):
        self.central_widget.setCurrentWidget(self.elementary_functions_menu)

    def show_input_dialog(self, method_name):
        self.current_method = method_name
        self.create_and_show_input_dialog(method_name,
                                          ["Нижний предел (a):", "Верхний предел (b):", "Число разбиений (n):",
                                           "Функция (например, 'x**2 + 1'):"],
                                          additional_fields=["Точность (например, 0.001):"] if method_name in [
                                              "Метод трапеций с переменным шагом", "Двойной пересчёт"] else [])

    def create_and_show_input_dialog(self, method_name, labels, additional_fields=[]):
        input_dialog = QtWidgets.QDialog(self)
        input_dialog.setWindowTitle("Введите параметры")
        layout = QtWidgets.QVBoxLayout()
        entries = []

        for label_text in labels:
            label = QtWidgets.QLabel(label_text)
            entry = QtWidgets.QLineEdit()
            layout.addWidget(label)
            layout.addWidget(entry)
            entries.append(entry)

        for label_text in additional_fields:
            label = QtWidgets.QLabel(label_text)
            entry = QtWidgets.QLineEdit()
            layout.addWidget(label)
            layout.addWidget(entry)
            entries.append(entry)

        calculate_button = QtWidgets.QPushButton("Рассчитать")
        calculate_button.clicked.connect(partial(self.calculate_and_show_integral, entries, input_dialog))
        layout.addWidget(calculate_button)

        close_button = QtWidgets.QPushButton("Закрыть")
        close_button.clicked.connect(input_dialog.close)
        layout.addWidget(close_button)

        input_dialog.setLayout(layout)
        input_dialog.exec_()

    def calculate_and_show_integral(self, entries, input_dialog):
        try:
            a = float(entries[0].text())
            b = float(entries[1].text())
            n = int(entries[2].text())
            func_str = entries[3].text()
            func = self.safe_eval(f"lambda x: {func_str}")
            tolerance = float(entries[4].text()) if len(entries) > 4 else None
            result = self.calculate_integral(self.current_method, func, a, b, n, tolerance)
            QtWidgets.QMessageBox.information(self, "Результат", f"Результат: {result:.4f}")
        except (ValueError, SyntaxError, NameError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))

    def calculate_integral(self, method_name, func, a, b, n, tolerance):
        if method_name == "Метод парабол":
            return Function.calculate_integral_parabolic(func, a, b, n)
        elif method_name == "Метод трапеций":
            return Function.calculate_integral_trapezoid(func, a, b, n)
        elif method_name == "Метод прямоугольников (правых частей)":
            return Function.calculate_integral_right_rect(func, a, b, n)
        elif method_name == "Метод прямоугольников (левых частей)":
            return Function.calculate_integral_left_rect(func, a, b, n)
        elif method_name == "Метод трапеций с переменным шагом":
            return Function.calculate_integral_slow(func, a, b, tolerance)
        elif method_name == "Двойной пересчёт":
            return Function.calculate_integral_fast(func, a, b, tolerance)
        else:
            raise ValueError("Неизвестный метод")

    def show_diff_eq_input_dialog(self, method_name):
        self.current_method = method_name
        input_dialog = QtWidgets.QDialog(self)
        input_dialog.setWindowTitle("Введите параметры")
        layout = QtWidgets.QVBoxLayout()

        if method_name == "Контрольный пример 3":
            t0_label = QtWidgets.QLabel("Начальное значение t (t0):")
            t0_entry = QtWidgets.QLineEdit()
            t0_entry.setText("0")
            layout.addWidget(t0_label)
            layout.addWidget(t0_entry)

            y0_label = QtWidgets.QLabel("Начальные значения x, y, z (через запятую):")
            y0_entry = QtWidgets.QLineEdit()
            y0_entry.setText("2, 1, 1")
            layout.addWidget(y0_label)
            layout.addWidget(y0_entry)

            t_end_label = QtWidgets.QLabel("Конечное значение t (t_end):")
            t_end_entry = QtWidgets.QLineEdit()
            t_end_entry.setText("0.3")
            layout.addWidget(t_end_label)
            layout.addWidget(t_end_entry)

            h_label = QtWidgets.QLabel("Шаг (h):")
            h_entry = QtWidgets.QLineEdit()
            h_entry.setText("0.003")
            layout.addWidget(h_label)
            layout.addWidget(h_entry)

        elif method_name == "Контрольный пример 1":
            t0_label = QtWidgets.QLabel("Начальное значение t (t0):")
            t0_entry = QtWidgets.QLineEdit()
            t0_entry.setText("0")
            layout.addWidget(t0_label)
            layout.addWidget(t0_entry)

            y0_label = QtWidgets.QLabel("Начальное значение y (y0):")
            y0_entry = QtWidgets.QLineEdit()
            y0_entry.setText("1")
            layout.addWidget(y0_label)
            layout.addWidget(y0_entry)

            t_end_label = QtWidgets.QLabel("Конечное значение t (t_end):")
            t_end_entry = QtWidgets.QLineEdit()
            t_end_entry.setText("1")
            layout.addWidget(t_end_label)
            layout.addWidget(t_end_entry)

            h_label = QtWidgets.QLabel("Шаг (h):")
            h_entry = QtWidgets.QLineEdit()
            h_entry.setText("0.1")
            layout.addWidget(h_label)
            layout.addWidget(h_entry)

        elif method_name == "Контрольный пример 2":
            t0_label = QtWidgets.QLabel("Начальное значение t (t0):")
            t0_entry = QtWidgets.QLineEdit()
            t0_entry.setText("1")
            layout.addWidget(t0_label)
            layout.addWidget(t0_entry)

            y0_label = QtWidgets.QLabel("Начальное значение y (y0):")
            y0_entry = QtWidgets.QLineEdit()
            y0_entry.setText("0.77")
            layout.addWidget(y0_label)
            layout.addWidget(y0_entry)

            y_prime_0_label = QtWidgets.QLabel("Начальное значение y' (y'0):")
            y_prime_0_entry = QtWidgets.QLineEdit()
            y_prime_0_entry.setText("-0.44")
            layout.addWidget(y_prime_0_label)
            layout.addWidget(y_prime_0_entry)

            t_end_label = QtWidgets.QLabel("Конечное значение t (t_end):")
            t_end_entry = QtWidgets.QLineEdit()
            t_end_entry.setText("1.5")
            layout.addWidget(t_end_label)
            layout.addWidget(t_end_entry)

            h_label = QtWidgets.QLabel("Шаг (h):")
            h_entry = QtWidgets.QLineEdit()
            h_entry.setText("0.1")
            layout.addWidget(h_label)
            layout.addWidget(h_entry)
        else:

            func_label = QtWidgets.QLabel(
                "Функция f(t, y):" if method_name != "Система дифференциальных уравнений" else "Функции f(t, y) (через точку с запятой):")
            func_entry = QtWidgets.QLineEdit()
            layout.addWidget(func_label)
            layout.addWidget(func_entry)
            t0_label = QtWidgets.QLabel("Начальное значение t (t0):")
            t0_entry = QtWidgets.QLineEdit()
            layout.addWidget(t0_label)
            layout.addWidget(t0_entry)
            y0_label = QtWidgets.QLabel(
                "Начальное значение y (y0):" if method_name != "Система дифференциальных уравнений" else "Начальные значения y (через запятую):")
            y0_entry = QtWidgets.QLineEdit()
            layout.addWidget(y0_label)
            layout.addWidget(y0_entry)
            t_end_label = QtWidgets.QLabel("Конечное значение t (t_end):")
            t_end_entry = QtWidgets.QLineEdit()
            layout.addWidget(t_end_label)
            layout.addWidget(t_end_entry)
            h_label = QtWidgets.QLabel("Шаг (h):")
            h_entry = QtWidgets.QLineEdit()
            layout.addWidget(h_label)
            layout.addWidget(h_entry)

            if method_name == "Система дифференциальных уравнений":
                num_equations_label = QtWidgets.QLabel("Количество уравнений:")
                num_equations_entry = QtWidgets.QLineEdit()
                num_equations_entry.setText("2")
                layout.addWidget(num_equations_label)
                layout.addWidget(num_equations_entry)

        calculate_button = QtWidgets.QPushButton("Рассчитать")
        calculate_button.clicked.connect(
            partial(self.calculate_diff_eq_and_show,
                    func_entry if method_name not in ["Контрольный пример 3", "Контрольный пример 1",
                                                      "Контрольный пример 2"] else None,
                    t0_entry,
                    y0_entry,
                    t_end_entry,
                    h_entry,
                    input_dialog,
                    num_equations_entry if method_name == "Система дифференциальных уравнений" else None,
                    y_prime_0_entry if method_name == "Контрольный пример 2" else None
                    ))
        layout.addWidget(calculate_button)

        close_button = QtWidgets.QPushButton("Закрыть")
        close_button.clicked.connect(input_dialog.close)
        layout.addWidget(close_button)

        input_dialog.setLayout(layout)
        input_dialog.exec_()

    def calculate_diff_eq_and_show(self, func_entry, t0_entry, y0_entry, t_end_entry, h_entry, input_dialog,
                                   num_equations_entry=None, y_prime_0_entry=None):
        try:
            if self.current_method == "Система дифференциальных уравнений":
                func_strs = func_entry.text().split(';')
                funcs = [self.safe_eval(f"lambda t, y: {func_str.strip()}") for func_str in func_strs]
                num_equations = int(num_equations_entry.text())
                if len(funcs) != num_equations:
                    raise ValueError("Неверное количество функций")
                y0_str = y0_entry.text()
                y0_values = [float(x.strip()) for x in y0_str.split(',')]
                y0 = np.array(y0_values)
                if len(y0) != num_equations:
                    raise ValueError("Неверное количество начальных условий")
                result = self.choose_diff_eq_method(funcs, y0, float(t0_entry.text()), float(t_end_entry.text()),
                                                    float(h_entry.text()))
            elif self.current_method == "Контрольный пример 3":
                t0 = float(t0_entry.text())
                t_end = float(t_end_entry.text())
                h = float(h_entry.text())
                y0_str = y0_entry.text()
                y0_values = [float(x.strip()) for x in y0_str.split(',')]
                y0 = np.array(y0_values)

                def system_equations_example(t, y):
                    x, y_val, z = y
                    return np.array([-2 * x + 5 * z, np.sin(t - 1) * x - y_val + 3 * z, -x + 2 * z])

                result = self.choose_diff_eq_method(system_equations_example, y0, t0, t_end, h)
            elif self.current_method == "Контрольный пример 1":
                t0 = float(t0_entry.text())
                t_end = float(t_end_entry.text())
                h = float(h_entry.text())
                y0 = float(y0_entry.text())

                def example_function(t, y):
                    return y * (1 - t)

                result = self.choose_diff_eq_method(example_function, y0, t0, t_end, h)
            elif self.current_method == "Контрольный пример 2":
                y0 = float(y0_entry.text())
                y_prime_0 = float(y_prime_0_entry.text())
                t0 = float(t0_entry.text())
                t_end = float(t_end_entry.text())
                h = float(h_entry.text())

                def second_order_ode(t, y, y_prime):
                    return -y_prime / t - y

                result_euler = Function.solve_second_order_ode_euler(second_order_ode, y0, y_prime_0, t0, t_end, h)
                result_runge_kutta = Function.solve_second_order_ode_runge_kutta(second_order_ode, y0, y_prime_0, t0,
                                                                                 t_end, h)
                result = {"euler": result_euler, "runge_kutta": result_runge_kutta}
            else:
                func = self.safe_eval(f"lambda t, y: {func_entry.text()}")
                result = self.choose_diff_eq_method(func, float(y0_entry.text()), float(t0_entry.text()),
                                                    float(t_end_entry.text()), float(h_entry.text()))

            if result is None:
                return

            self.show_diff_eq_results(result)

        except (ValueError, SyntaxError, NameError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))

    def show_diff_eq_results(self, result):
        if result is None or len(result) == 0:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Вычисления не дали результата.")
            return

        result_str = "Результаты:\n"
        t_values = []
        if self.current_method == "Контрольный пример 2":
            result_euler = result["euler"]
            result_runge_kutta = result["runge_kutta"]
            result_str += "Метод Эйлера:\n"
            t_values_euler = []
            y_values_euler = []
            for t, y in result_euler:
                result_str += f"t = {t:.4f}, y = {y:.4f}\n"
                t_values_euler.append(t)
                y_values_euler.append(y)
            result_str += "\nМетод Рунге-Кутты:\n"
            t_values_runge_kutta = []
            y_values_runge_kutta = []
            for t, y in result_runge_kutta:
                result_str += f"t = {t:.4f}, y = {y:.4f}\n"
                t_values_runge_kutta.append(t)
                y_values_runge_kutta.append(y)
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(t_values_euler, y_values_euler, label='Метод Эйлера')
            self.ax.plot(t_values_runge_kutta, y_values_runge_kutta, label='Метод Рунге-Кутты')
            self.ax.set_xlabel("t")
            self.ax.set_ylabel("y")
            self.ax.set_title("Контрольный пример 2")
            self.ax.legend()
            self.canvas.draw()


        elif isinstance(result[0][1], np.ndarray):
            num_y = len(result[0][1])
            y_values = [[] for _ in range(num_y)]
            for t, y in result:
                result_str += f"t = {t:.4f}, "
                for i, val in enumerate(y):
                    result_str += f"y{i + 1} = {val:.4f}, "
                    y_values[i].append(val)
                result_str += "\n"
                t_values.append(t)
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            for i in range(len(y_values)):
                self.ax.plot(t_values, y_values[i], label=f'y{i + 1}')
            self.ax.set_xlabel("t")
            self.ax.set_ylabel("y")
            self.ax.set_title(self.current_method)
            self.ax.legend()
            self.canvas.draw()

        elif isinstance(result[0][1], (int, float)):
            y_values = []
            for t, y in result:
                result_str += f"t = {t:.4f}, y = {y:.4f}\n"
                y_values.append(y)
                t_values.append(t)
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.plot(t_values, y_values, label='y')
            self.ax.set_xlabel("t")
            self.ax.set_ylabel("y")
            self.ax.set_title(self.current_method)
            self.ax.legend()
            self.canvas.draw()

        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Неверный формат результата.")
            return

        plot_dialog = QtWidgets.QDialog(self)
        plot_dialog.setWindowTitle("График и Расчеты")
        plot_layout = QtWidgets.QVBoxLayout()
        text_area = QtWidgets.QTextEdit()
        text_area.setText(result_str)
        text_area.setReadOnly(True)
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(text_area)
        plot_dialog.setLayout(plot_layout)
        plot_dialog.resize(800, 600)
        plot_dialog.exec_()

    def choose_diff_eq_method(self, *args):
        methods = {
            "Система дифференциальных уравнений": Function.solve_system_euler,
            "Метод Эйлера": Function.euler_method_first_order,
            "Метод Рунге-Кутты": Function.runge_kutta_method_first_order,
            "Контрольный пример 3": Function.solve_system_euler,
            "Контрольный пример 1": Function.euler_method_first_order
        }
        method = methods.get(self.current_method)
        if method:
            try:
                result = method(*args)
                return result
            except (ValueError, TypeError) as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Ошибка при вычислении: {e}")
                return None
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Неизвестный метод")
            return None

    def show_nonlinear_equations_menu(self):
        self.central_widget.setCurrentWidget(self.nonlinear_equations_menu)

    def show_nonlinear_input_dialog(self, method_name):
        self.current_method = method_name
        self.create_and_show_nonlinear_input_dialog(method_name)

    def create_and_show_nonlinear_input_dialog(self, method_name):
        input_dialog = QtWidgets.QDialog(self)
        input_dialog.setWindowTitle("Введите параметры")
        layout = QtWidgets.QVBoxLayout()
        function_label = QtWidgets.QLabel("Функция f(x):")
        function_entry = QtWidgets.QLineEdit()
        layout.addWidget(function_label)
        layout.addWidget(function_entry)
        if method_name == "Метод касательных":
            derivative_label = QtWidgets.QLabel("Производная f'(x):")
            derivative_entry = QtWidgets.QLineEdit()
            layout.addWidget(derivative_label)
            layout.addWidget(derivative_entry)

        x0_label = QtWidgets.QLabel("Начальное приближение x0:")
        x0_entry = QtWidgets.QLineEdit()
        layout.addWidget(x0_label)
        layout.addWidget(x0_entry)

        if method_name == "Метод хорд":
            x1_label = QtWidgets.QLabel("Второе приближение x1:")
            x1_entry = QtWidgets.QLineEdit()
            layout.addWidget(x1_label)
            layout.addWidget(x1_entry)
        elif method_name == "Метод дихотомии":
            a_label = QtWidgets.QLabel("Левая граница a:")
            a_entry = QtWidgets.QLineEdit()

            b_label = QtWidgets.QLabel("Правая граница b:")
            b_entry = QtWidgets.QLineEdit()
            layout.addWidget(a_label)
            layout.addWidget(a_entry)
            layout.addWidget(b_label)
            layout.addWidget(b_entry)

        tolerance_label = QtWidgets.QLabel("Точность:")
        tolerance_entry = QtWidgets.QLineEdit()
        layout.addWidget(tolerance_label)
        layout.addWidget(tolerance_entry)

        max_iterations_label = QtWidgets.QLabel("Максимальное число итераций:")
        max_iterations_entry = QtWidgets.QLineEdit()
        max_iterations_entry.setText("100")
        layout.addWidget(max_iterations_label)
        layout.addWidget(max_iterations_entry)

        calculate_button = QtWidgets.QPushButton("Рассчитать")
        calculate_button.clicked.connect(
            partial(self.calculate_nonlinear_and_show, function_entry, x0_entry, tolerance_entry, max_iterations_entry,
                    input_dialog, derivative_entry if method_name == "Метод касательных" else None,
                    x1_entry if method_name == "Метод хорд" else None,
                    a_entry if method_name == "Метод дихотомии" else None,
                    b_entry if method_name == "Метод дихотомии" else None))
        layout.addWidget(calculate_button)

        close_button = QtWidgets.QPushButton("Закрыть")
        close_button.clicked.connect(input_dialog.close)
        layout.addWidget(close_button)

        input_dialog.setLayout(layout)
        input_dialog.exec_()

    def calculate_nonlinear_and_show(self, function_entry, x0_entry, tolerance_entry, max_iterations_entry,
                                     input_dialog, derivative_entry, x1_entry, a_entry, b_entry):
        try:
            func = self.safe_eval(f"lambda x: {function_entry.text()}")
            tolerance = float(tolerance_entry.text())
            max_iterations = int(max_iterations_entry.text())
            entries = [function_entry, x0_entry, tolerance_entry, max_iterations_entry]

            if self.current_method == "Метод касательных":
                deriv = self.safe_eval(f"lambda x: {derivative_entry.text()}")
                result = Function.newton_method(func, deriv, float(x0_entry.text()), tolerance, max_iterations)
            elif self.current_method == "Метод хорд":
                result = Function.secant_method(func, float(x0_entry.text()), float(x1_entry.text()), tolerance,
                                                max_iterations)
            elif self.current_method == "Метод дихотомии":
                result = Function.dichotomy_method(func, float(a_entry.text()), float(b_entry.text()), tolerance,
                                                   max_iterations)

            if result is not None:
                QtWidgets.QMessageBox.information(self, "Результат", f"Результат: {result:.4f}")
            else:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Метод не сошелся.")
        except (ValueError, SyntaxError, NameError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))

    def show_elementary_functions_input_dialog(self, method_name):
        self.current_method = method_name
        input_dialog = QtWidgets.QDialog(self)
        input_dialog.setWindowTitle("Введите параметры")
        layout = QtWidgets.QVBoxLayout()

        x_label = QtWidgets.QLabel("Значение x:")
        x_entry = QtWidgets.QLineEdit()
        layout.addWidget(x_label)
        layout.addWidget(x_entry)

        func_label = QtWidgets.QLabel("Функция:")
        func_entry = QtWidgets.QLineEdit()
        layout.addWidget(func_label)
        layout.addWidget(func_entry)

        tolerance_label = QtWidgets.QLabel("Точность (например, 0.001):")
        tolerance_entry = QtWidgets.QLineEdit()
        layout.addWidget(tolerance_label)
        layout.addWidget(tolerance_entry)

        show_examples_button = QtWidgets.QPushButton("Показать ответы")
        show_examples_button.clicked.connect(partial(self.show_examples_dialog, "Элементарные функции", method_name))
        layout.addWidget(show_examples_button)

        functions_button = QtWidgets.QPushButton("Список функций")
        functions_button.clicked.connect(self.show_function_list)
        layout.addWidget(functions_button)

        calculate_button = QtWidgets.QPushButton("Рассчитать")
        calculate_button.clicked.connect(
            partial(self.calculate_elementary_function_and_show, x_entry, func_entry, tolerance_entry, input_dialog))
        layout.addWidget(calculate_button)

        close_button = QtWidgets.QPushButton("Закрыть")
        close_button.clicked.connect(input_dialog.close)
        layout.addWidget(close_button)

        input_dialog.setLayout(layout)
        input_dialog.exec_()

    def calculate_elementary_function_and_show(self, x_entry, func_entry, tolerance_entry, input_dialog):
        try:
            x = float(x_entry.text())
            func_name = func_entry.text().lower()
            tolerance = float(tolerance_entry.text())
            entries = [x_entry, func_entry, tolerance_entry]
            if self.current_method == "Метод итераций":
                result = Function.elementary_function_iteration(func_name, x, tolerance)
            elif self.current_method == "Метод Чебышева":
                result = Function.elementary_function_chebyshev(func_name, x, tolerance)
            else:
                raise ValueError("Неизвестный метод")

            if result is not None:
                QtWidgets.QMessageBox.information(self, "Результат", f"Результат: {result:.4f}")
            else:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Метод не сошелся.")

        except (ValueError, SyntaxError, NameError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))

    def show_function_list(self):
        functions = ["sin - синус", "cos - косинус", "exp - экспонента", "log - натуральный логарифм",
                     "tan - тангенс", "sinh - гиперболический синус", "cosh - гиперболический косинус",
                     "tanh - гиперболический тангенс", "sqrt - квадратный корень", "arcsin - арксинус",
                     "arccos - арккосинус", "arctan - арктангенс"]
        message = "Поддерживаемые элементарные функции:\n" + "\n".join(functions)
        QtWidgets.QMessageBox.information(self, "Список функций", message)

    def safe_eval(self, expression):
        try:
            return eval(expression)
        except (SyntaxError, NameError, TypeError) as e:
            raise ValueError(f"Ошибка в выражении: {e}")

    def show_examples_dialog(self, group_name, method_name):
        examples = self.examples.get(group_name, {}).get(method_name, [])
        if not examples:
            QtWidgets.QMessageBox.information(self, "Примеры", "Нет примеров для данного метода.")
            return

        examples_str = f"Примеры для '{group_name}' - '{method_name}':\n\n"
        for i, example in enumerate(examples):
            input_str = ", ".join(f"{key}={value}" for key, value in example["input"].items())
            examples_str += f"задача {i + 1}:\n"
            examples_str += f" Ввод: {input_str}\n"
            examples_str += f" результат: {example['expected']}\n"
            examples_str += f"  {example['history_text']}\n\n"

        QtWidgets.QMessageBox.information(self, "Примеры", examples_str)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MenuApp()
    window.show()
    sys.exit(app.exec_())
