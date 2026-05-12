import csv
import json
import random
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from itertools import combinations

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ModuleNotFoundError:
    train_test_split = None
    LinearRegression = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None




BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "products.json"
DEMAND_DATA_PATH = BASE_DIR / "Refrigerator.csv"
TITLE_FONT = ("Arial", 16, "bold")

DEMAND_FEATURES = [
    "avg_refrigerator_price",
    "energy_efficiency_class",
    "is_seasonal_demand",
    "advertising_budget",
    "warranty_period",
]

DEMAND_TARGET = "monthly_refrigerator_sales"

DEMAND_FEATURE_LABELS = {
    "avg_refrigerator_price": "Средняя цена холодильного оборудования, тыс. руб.",
    "energy_efficiency_class": "Класс энергоэффективности (A=3, B=2, C=1)",
    "is_seasonal_demand": "Сезонный спрос (1 - да, 0 - нет)",
    "advertising_budget": "Рекламный бюджет, тыс. руб.",
    "warranty_period": "Срок гарантии, лет",
}


QUALITY_METRICS = {
    "cooling_capacity": {
        "name": "Холодопроизводительность",
        "weight": 0.25,
        "direction": "max",
    },
    "volume": {
        "name": "Объем",
        "weight": 0.15,
        "direction": "max",
    },
    "energy_efficiency": {
        "name": "Энергоэффективность",
        "weight": 0.20,
        "direction": "max",
    },
    "temperature_stability": {
        "name": "Стабильность температуры",
        "weight": 0.15,
        "direction": "min",
    },
    "noise": {
        "name": "Уровень шума",
        "weight": 0.10,
        "direction": "min",
    },
    "service_life": {
        "name": "Срок службы",
        "weight": 0.15,
        "direction": "max",
    },
}


def load_products():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_quality(products):
    results = []

    max_values = {}
    min_values = {}

    for metric in QUALITY_METRICS:
        values = [product[metric] for product in products]
        max_values[metric] = max(values)
        min_values[metric] = min(values)

    for product in products:
        quality_score = 0

        for metric, settings in QUALITY_METRICS.items():
            value = product[metric]
            weight = settings["weight"]
            direction = settings["direction"]

            if direction == "max":
                normalized_value = value / max_values[metric]
            else:
                normalized_value = min_values[metric] / value

            quality_score += normalized_value * weight

        results.append({
            "name": product["name"],
            "type": product["type"],
            "quality_score": quality_score,
            "price": product["price"],
        })

    results.sort(key=lambda item: item["quality_score"], reverse=True)
    return results

def get_normalized_product_values(product, products):
    normalized_values = []
    metric_names = []

    for metric, settings in QUALITY_METRICS.items():
        values = [item[metric] for item in products]

        max_value = max(values)
        min_value = min(values)

        value = product[metric]

        if settings["direction"] == "max":
            normalized_value = value / max_value
        else:
            normalized_value = min_value / value

        normalized_values.append(normalized_value)
        metric_names.append(settings["name"])

    return metric_names, normalized_values




def calculate_forecast(sales_history, forecast_periods=3):
    n = len(sales_history)
    x_values = list(range(1, n + 1))
    y_values = sales_history

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    numerator = sum(
        (x_values[i] - x_mean) * (y_values[i] - y_mean)
        for i in range(n)
    )
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator

    intercept = y_mean - slope * x_mean

    forecast = []

    for year_number in range(n + 1, n + forecast_periods + 1):
        value = slope * year_number + intercept
        forecast.append(round(max(value, 0), 2))

    return forecast


class ManualLinearRegression:
    def __init__(self):
        self.means = []
        self.stds = []
        self.coefficients = []

    def fit(self, X, y):
        feature_count = len(X[0])
        self.means = []
        self.stds = []

        for column_index in range(feature_count):
            column_values = [row[column_index] for row in X]
            mean_value = sum(column_values) / len(column_values)
            variance = sum((value - mean_value) ** 2 for value in column_values) / len(column_values)
            std_value = variance ** 0.5

            self.means.append(mean_value)
            self.stds.append(std_value if std_value != 0 else 1)

        prepared_X = []

        for row in X:
            prepared_row = [1]

            for index, value in enumerate(row):
                prepared_row.append((value - self.means[index]) / self.stds[index])

            prepared_X.append(prepared_row)

        xtx = self._multiply_transpose_by_matrix(prepared_X)
        xty = self._multiply_transpose_by_vector(prepared_X, y)
        self.coefficients = self._solve_linear_system(xtx, xty)

        return self

    def predict(self, X):
        predictions = []

        for row in X:
            prepared_row = [1]

            for index, value in enumerate(row):
                prepared_row.append((value - self.means[index]) / self.stds[index])

            prediction = sum(
                coefficient * value
                for coefficient, value in zip(self.coefficients, prepared_row)
            )
            predictions.append(prediction)

        return predictions

    def _multiply_transpose_by_matrix(self, matrix):
        result_size = len(matrix[0])
        result = [
            [0 for _ in range(result_size)]
            for _ in range(result_size)
        ]

        for row in matrix:
            for i in range(result_size):
                for j in range(result_size):
                    result[i][j] += row[i] * row[j]

        return result

    def _multiply_transpose_by_vector(self, matrix, vector):
        result_size = len(matrix[0])
        result = [0 for _ in range(result_size)]

        for row, value in zip(matrix, vector):
            for i in range(result_size):
                result[i] += row[i] * value

        return result

    def _solve_linear_system(self, matrix, vector):
        size = len(vector)
        augmented = [
            [float(value) for value in matrix[row_index]] + [float(vector[row_index])]
            for row_index in range(size)
        ]

        for column_index in range(size):
            pivot_row = max(
                range(column_index, size),
                key=lambda row_index: abs(augmented[row_index][column_index])
            )

            if abs(augmented[pivot_row][column_index]) < 1e-12:
                raise ValueError("Не удалось обучить модель: матрица признаков вырождена.")

            augmented[column_index], augmented[pivot_row] = augmented[pivot_row], augmented[column_index]
            pivot = augmented[column_index][column_index]

            for value_index in range(column_index, size + 1):
                augmented[column_index][value_index] /= pivot

            for row_index in range(size):
                if row_index == column_index:
                    continue

                factor = augmented[row_index][column_index]

                for value_index in range(column_index, size + 1):
                    augmented[row_index][value_index] -= factor * augmented[column_index][value_index]

        return [augmented[row_index][size] for row_index in range(size)]


def load_demand_dataset(csv_path=DEMAND_DATA_PATH):
    rows = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = DEMAND_FEATURES + [DEMAND_TARGET]

        if reader.fieldnames is None:
            raise ValueError("CSV-файл пустой или не содержит заголовков.")

        missing_columns = [
            column for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                "В CSV-файле отсутствуют столбцы: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            rows.append(row)

    if len(rows) < 10:
        raise ValueError("Для обучения модели недостаточно строк в датасете.")

    X = []
    y = []

    for row in rows:
        X.append([float(row[feature]) for feature in DEMAND_FEATURES])
        y.append(float(row[DEMAND_TARGET]))

    return X, y, rows


def split_train_test_manual(X, y, test_size=0.2, random_state=101):
    indexes = list(range(len(X)))
    random.Random(random_state).shuffle(indexes)
    test_count = max(1, int(len(indexes) * test_size))
    test_indexes = indexes[:test_count]
    train_indexes = indexes[test_count:]

    X_train = [X[index] for index in train_indexes]
    X_test = [X[index] for index in test_indexes]
    y_train = [y[index] for index in train_indexes]
    y_test = [y[index] for index in test_indexes]

    return X_train, X_test, y_train, y_test


def calculate_regression_metrics(actual_values, predicted_values):
    count = len(actual_values)
    mae = sum(
        abs(actual - predicted)
        for actual, predicted in zip(actual_values, predicted_values)
    ) / count
    mse = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_values, predicted_values)
    ) / count
    rmse = mse ** 0.5
    average_actual = sum(actual_values) / count
    ss_res = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(actual_values, predicted_values)
    )
    ss_tot = sum((actual - average_actual) ** 2 for actual in actual_values)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    return mae, rmse, r2


def train_demand_model(csv_path=DEMAND_DATA_PATH):
    X, y, rows = load_demand_dataset(csv_path)

    if LinearRegression is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=101
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_predictions)
        mse = mean_squared_error(y_test, test_predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, test_predictions)
        model_name = "LinearRegression из scikit-learn"
    else:
        X_train, X_test, y_train, y_test = split_train_test_manual(X, y)
        model = ManualLinearRegression()
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        mae, rmse, r2 = calculate_regression_metrics(y_test, test_predictions)
        model_name = "встроенная линейная регрессия"

    return {
        "model": model,
        "rows": rows,
        "test_actual": list(y_test),
        "test_predictions": list(test_predictions),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "csv_path": csv_path,
        "model_name": model_name,
    }


def predict_monthly_demand(model_info, input_values):
    model = model_info["model"]
    X_new = [[float(input_values[feature]) for feature in DEMAND_FEATURES]]
    prediction = model.predict(X_new)[0]

    return round(max(prediction, 0), 2)


def find_optimal_equipment_set(products, budget, required_volume, selected_type):
    quality_results = calculate_quality(products)

    quality_by_name = {
        result["name"]: result["quality_score"]
        for result in quality_results
    }

    candidates = [
        product for product in products
        if product["type"] == selected_type
    ]

    best_combination = None
    best_score = -1

    for count in range(1, len(candidates) + 1):
        for combination in combinations(candidates, count):
            total_price = sum(product["price"] for product in combination)
            total_volume = sum(product["volume"] for product in combination)

            if total_price > budget:
                continue

            if total_volume < required_volume:
                continue

            usefulness = sum(
                quality_by_name[product["name"]] * product["volume"]
                for product in combination
            )

            if usefulness > best_score:
                best_score = usefulness
                best_combination = combination

    if best_combination is None:
        return None

    total_price = sum(product["price"] for product in best_combination)
    total_volume = sum(product["volume"] for product in best_combination)
    average_quality = sum(
        quality_by_name[product["name"]]
        for product in best_combination
    ) / len(best_combination)

    return {
        "products": best_combination,
        "total_price": total_price,
        "total_volume": total_volume,
        "average_quality": average_quality,
        "usefulness": best_score,
        "quality_by_name": quality_by_name,
    }


def create_optimization_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Оптимизация подбора оборудования",
        font=TITLE_FONT
    )
    title.pack(pady=10)

    product_types = sorted(set(product["type"] for product in products))
    selected_type = tk.StringVar(value=product_types[0])

    parameters_frame = ttk.LabelFrame(parent, text="Параметры подбора")
    parameters_frame.pack(fill="x", padx=10, pady=8)

    ttk.Label(parameters_frame, text="Тип продукции:").grid(
        row=0,
        column=0,
        padx=5,
        pady=8,
        sticky="w"
    )

    type_box = ttk.Combobox(
        parameters_frame,
        textvariable=selected_type,
        values=product_types,
        state="readonly",
        width=40
    )
    type_box.grid(row=0, column=1, padx=5, pady=8, sticky="w")

    ttk.Label(parameters_frame, text="Бюджет, руб.:").grid(
        row=0,
        column=2,
        padx=5,
        pady=8,
        sticky="w"
    )

    budget_entry = ttk.Entry(parameters_frame, width=18)
    budget_entry.grid(row=0, column=3, padx=5, pady=8, sticky="w")
    budget_entry.insert(0, "2000000")

    ttk.Label(parameters_frame, text="Мин. объем, м³:").grid(
        row=0,
        column=4,
        padx=5,
        pady=8,
        sticky="w"
    )

    volume_entry = ttk.Entry(parameters_frame, width=18)
    volume_entry.grid(row=0, column=5, padx=5, pady=8, sticky="w")
    volume_entry.insert(0, "200")

    result_frame = ttk.LabelFrame(parent, text="Результат оптимизации")
    result_frame.pack(fill="both", expand=True, padx=10, pady=8)

    result_label = ttk.Label(
        result_frame,
        text="",
        wraplength=950,
        justify="left"
    )
    result_label.pack(fill="x", padx=10, pady=8)

    columns = ("name", "quality", "volume", "price")

    result_table = ttk.Treeview(result_frame, columns=columns, show="headings", height=7)

    result_table.heading("name", text="Название")
    result_table.heading("quality", text="Q")
    result_table.heading("volume", text="Объем, м³")
    result_table.heading("price", text="Цена, руб.")

    result_table.column("name", width=360)
    result_table.column("quality", width=100)
    result_table.column("volume", width=120)
    result_table.column("price", width=140)

    result_table.pack(fill="both", expand=True, padx=10, pady=10)

    def calculate_optimization():
        for row in result_table.get_children():
            result_table.delete(row)

        try:
            budget = float(budget_entry.get())
            required_volume = float(volume_entry.get())
        except ValueError:
            result_label.config(
                text="Ошибка: бюджет и объем должны быть числовыми значениями."
            )
            return

        if budget <= 0:
            result_label.config(text="Ошибка: бюджет должен быть больше нуля.")
            return

        if required_volume <= 0:
            result_label.config(text="Ошибка: требуемый объем должен быть больше нуля.")
            return

        result = find_optimal_equipment_set(
            products,
            budget,
            required_volume,
            selected_type.get()
        )

        if result is None:
            result_label.config(
                text=(
                    "Не удалось подобрать набор оборудования: "
                    "увеличьте бюджет или уменьшите требуемый объем."
                )
            )
            return

        result_label.config(
            text=(
                f"Подобран оптимальный набор оборудования.\n"
                f"Тип продукции: {selected_type.get()}\n"
                f"Суммарная цена: {result['total_price']} руб.\n"
                f"Суммарный объем: {result['total_volume']} м³\n"
                f"Средний показатель качества Q: {round(result['average_quality'], 3)}"
            )
        )

        for product in result["products"]:
            product_quality = result["quality_by_name"][product["name"]]

            result_table.insert(
                "",
                tk.END,
                values=(
                    product["name"],
                    round(product_quality, 3),
                    product["volume"],
                    product["price"],
                )
            )

    ttk.Button(
        parameters_frame,
        text="Подобрать оборудование",
        command=calculate_optimization
    ).grid(row=0, column=6, padx=10, pady=8, sticky="w")




def create_forecast_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Прогноз продаж продукции",
        font=TITLE_FONT
    )
    title.pack(pady=10)

    product_types = sorted(set(product["type"] for product in products))
    selected_type = tk.StringVar(value=product_types[0])
    product_query = tk.StringVar()

    controls_frame = ttk.LabelFrame(parent, text="Параметры прогноза")
    controls_frame.pack(fill="x", padx=10, pady=8)

    ttk.Label(controls_frame, text="Тип продукции:").grid(
        row=0,
        column=0,
        padx=5,
        pady=5,
        sticky="w"
    )

    type_box = ttk.Combobox(
        controls_frame,
        textvariable=selected_type,
        values=product_types,
        state="readonly",
        width=35
    )
    type_box.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(controls_frame, text="Товар:").grid(
        row=1,
        column=0,
        padx=5,
        pady=5,
        sticky="w"
    )

    product_entry = ttk.Entry(
        controls_frame,
        textvariable=product_query,
        width=55
    )
    product_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    dropdown_button = ttk.Button(
        controls_frame,
        text="▼",
        width=3
    )
    dropdown_button.grid(row=1, column=2, padx=2, pady=5, sticky="w")

    dropdown_frame = ttk.Frame(controls_frame)
    dropdown_frame.grid(row=2, column=1, columnspan=2, padx=5, sticky="w")
    dropdown_frame.grid_remove()

    product_listbox = tk.Listbox(
        dropdown_frame,
        height=7,
        width=58,
        exportselection=False
    )
    product_listbox.grid(row=0, column=0, sticky="nsew")

    list_scrollbar = ttk.Scrollbar(
        dropdown_frame,
        orient="vertical",
        command=product_listbox.yview
    )
    list_scrollbar.grid(row=0, column=1, sticky="ns")

    product_listbox.configure(yscrollcommand=list_scrollbar.set)

    result_label = ttk.Label(parent, text="")
    result_label.pack(pady=10)

    ttk.Label(
        parent,
        text="Пунктирная линия отделяет фактические данные от прогнозных значений.",
        foreground="#555555"
    ).pack(pady=2)

    chart_frame = ttk.LabelFrame(parent, text="График прогноза")
    chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def get_products_by_type():
        filtered_products = [
            product for product in products
            if product["type"] == selected_type.get()
        ]

        return sorted(filtered_products, key=lambda product: product["name"])

    def get_product_by_name(product_name):
        for product in products:
            same_type = product["type"] == selected_type.get()
            same_name = product["name"] == product_name

            if same_type and same_name:
                return product

        return None

    def clear_chart_with_message(message):
        for widget in chart_frame.winfo_children():
            widget.destroy()

        result_label.config(text="")

        ttk.Label(
            chart_frame,
            text=message,
            font=("Arial", 14)
        ).pack(pady=30)

    def get_filtered_product_names():
        typed_text = product_query.get().lower()

        filtered_products = [
            product for product in get_products_by_type()
            if typed_text in product["name"].lower()
        ]

        return [product["name"] for product in filtered_products]

    def update_dropdown():
        product_names = get_filtered_product_names()

        product_listbox.delete(0, tk.END)

        for name in product_names:
            product_listbox.insert(tk.END, name)

        if product_names:
            dropdown_frame.grid()
        else:
            dropdown_frame.grid_remove()

    def show_forecast(product):
        for widget in chart_frame.winfo_children():
            widget.destroy()

        sales_history = product["sales_history"]
        forecast = calculate_forecast(sales_history)

        result_label.config(
            text=f"Прогноз на 3 следующих года: {forecast}"
        )

        years = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
        values = sales_history + forecast

        figure = Figure(figsize=(8, 3), dpi=100)
        axes = figure.add_subplot(111)

        axes.plot(years, values, marker="o")
        axes.axvline(2025, linestyle="--")
        axes.set_title(f"Прогноз продаж: {product['name']}")
        axes.set_xlabel("Год")
        axes.set_ylabel("Количество продаж")
        axes.grid(True)

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def select_product_from_list(event=None):
        selected_indexes = product_listbox.curselection()

        if not selected_indexes:
            return

        product_name = product_listbox.get(selected_indexes[0])
        product_query.set(product_name)
        dropdown_frame.grid_remove()

        product = get_product_by_name(product_name)

        if product is not None:
            show_forecast(product)

    def on_product_typing(event=None):
        update_dropdown()

        product = get_product_by_name(product_query.get())

        if product is not None:
            show_forecast(product)
        elif not get_filtered_product_names():
            clear_chart_with_message("Товар не найден")

    def open_dropdown(event=None):
        update_dropdown()
        product_entry.focus_set()

    def reset_products_for_type(event=None):
        type_products = get_products_by_type()

        if not type_products:
            product_query.set("")
            dropdown_frame.grid_remove()
            clear_chart_with_message("Для выбранного типа товары не найдены")
            return

        first_product = type_products[0]
        product_query.set(first_product["name"])
        dropdown_frame.grid_remove()
        show_forecast(first_product)

    def select_first_from_dropdown(event=None):
        if product_listbox.size() == 0:
            return

        product_listbox.selection_clear(0, tk.END)
        product_listbox.selection_set(0)
        select_product_from_list()

    type_box.bind("<<ComboboxSelected>>", reset_products_for_type)

    product_entry.bind("<FocusIn>", open_dropdown)
    product_entry.bind("<Button-1>", open_dropdown)
    product_entry.bind("<KeyRelease>", on_product_typing)
    product_entry.bind("<Return>", select_first_from_dropdown)

    dropdown_button.configure(command=open_dropdown)

    product_listbox.bind("<<ListboxSelect>>", select_product_from_list)
    product_listbox.bind("<Return>", select_product_from_list)

    reset_products_for_type()



def create_data_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Данные о холодильном оборудовании",
        font=TITLE_FONT
    )
    title.pack(pady=10)

    search_frame = ttk.Frame(parent)
    search_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(search_frame, text="Поиск по названию:").pack(side="left", padx=5)

    search_var = tk.StringVar()

    search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
    search_entry.pack(side="left", padx=5)

    status_label = ttk.Label(search_frame, text="")
    status_label.pack(side="left", padx=10)

    columns = (
        "name",
        "type",
        "cooling_capacity",
        "volume",
        "energy_efficiency",
        "temperature_stability",
        "noise",
        "service_life",
        "price",
    )

    table_frame = ttk.LabelFrame(parent, text="Список продукции")
    table_frame.pack(fill="both", expand=True, padx=10, pady=10)

    tree = ttk.Treeview(table_frame, columns=columns, show="headings")

    tree.heading("name", text="Название")
    tree.heading("type", text="Тип")
    tree.heading("cooling_capacity", text="Холодопр., кВт")
    tree.heading("volume", text="Объем, м³")
    tree.heading("energy_efficiency", text="Энергоэфф.")
    tree.heading("temperature_stability", text="Стабильн., ±°C")
    tree.heading("noise", text="Шум, дБ")
    tree.heading("service_life", text="Срок службы, лет")
    tree.heading("price", text="Цена, руб.")

    tree.column("name", width=260)
    tree.column("type", width=180)
    tree.column("cooling_capacity", width=120)
    tree.column("volume", width=100)
    tree.column("energy_efficiency", width=110)
    tree.column("temperature_stability", width=130)
    tree.column("noise", width=90)
    tree.column("service_life", width=130)
    tree.column("price", width=120)

    def fill_table(filtered_products):
        for row in tree.get_children():
            tree.delete(row)

        for product in filtered_products:
            tree.insert(
                "",
                tk.END,
                values=(
                    product["name"],
                    product["type"],
                    product["cooling_capacity"],
                    product["volume"],
                    product["energy_efficiency"],
                    product["temperature_stability"],
                    product["noise"],
                    product["service_life"],
                    product["price"],
                ),
            )

        status_label.config(text=f"Найдено: {len(filtered_products)} из {len(products)}")

    def update_search(*args):
        search_text = search_var.get().lower()

        filtered_products = [
            product for product in products
            if search_text in product["name"].lower()
        ]

        fill_table(filtered_products)

    search_var.trace_add("write", update_search)

    vertical_scrollbar = ttk.Scrollbar(
        table_frame,
        orient="vertical",
        command=tree.yview
    )
    horizontal_scrollbar = ttk.Scrollbar(
        table_frame,
        orient="horizontal",
        command=tree.xview
    )

    tree.configure(
        yscrollcommand=vertical_scrollbar.set,
        xscrollcommand=horizontal_scrollbar.set
    )

    tree.grid(row=0, column=0, sticky="nsew")
    vertical_scrollbar.grid(row=0, column=1, sticky="ns")
    horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    fill_table(products)


def create_quality_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Оценка технического уровня продукции",
        font=TITLE_FONT
    )
    title.pack(pady=10)

    quality_results = calculate_quality(products)

    search_frame = ttk.Frame(parent)
    search_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(search_frame, text="Поиск по названию:").pack(side="left", padx=5)

    search_var = tk.StringVar()

    search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
    search_entry.pack(side="left", padx=5)

    status_label = ttk.Label(search_frame, text="")
    status_label.pack(side="left", padx=10)

    columns = ("place", "name", "type", "quality_score", "price")

    table_frame = ttk.LabelFrame(parent, text="Рейтинг продукции")
    table_frame.pack(fill="both", expand=True, padx=10, pady=10)

    tree = ttk.Treeview(table_frame, columns=columns, show="headings")

    tree.heading("place", text="Место")
    tree.heading("name", text="Название")
    tree.heading("type", text="Тип")
    tree.heading("quality_score", text="Качество Q")
    tree.heading("price", text="Цена, руб.")

    tree.column("place", width=70)
    tree.column("name", width=320)
    tree.column("type", width=220)
    tree.column("quality_score", width=120)
    tree.column("price", width=120)

    def fill_quality_table(filtered_results):
        for row in tree.get_children():
            tree.delete(row)

        for index, result in enumerate(filtered_results, start=1):
            tree.insert(
                "",
                tk.END,
                values=(
                    index,
                    result["name"],
                    result["type"],
                    round(result["quality_score"], 3),
                    result["price"],
                ),
            )

        status_label.config(text=f"Найдено: {len(filtered_results)} из {len(quality_results)}")

    def update_search(*args):
        search_text = search_var.get().lower()

        filtered_results = [
            result for result in quality_results
            if search_text in result["name"].lower()
        ]

        fill_quality_table(filtered_results)

    search_var.trace_add("write", update_search)

    vertical_scrollbar = ttk.Scrollbar(
        table_frame,
        orient="vertical",
        command=tree.yview
    )

    tree.configure(yscrollcommand=vertical_scrollbar.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vertical_scrollbar.grid(row=0, column=1, sticky="ns")

    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    fill_quality_table(quality_results)

    weights_text = "Весовые коэффициенты: "

    for metric in QUALITY_METRICS.values():
        weights_text += f"{metric['name']} - {metric['weight']}; "

    ttk.Label(
        parent,
        text=weights_text,
        wraplength=1000,
        justify="left"
    ).pack(padx=10, pady=8)

    product_types = [
        "Холодильные камеры",
        "Морозильные установки",
        "Сплит-системы",
    ]

    figure = Figure(figsize=(12, 4.5), dpi=100)

    colors = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]

    for index, product_type in enumerate(product_types, start=1):
        type_results = [
            result for result in quality_results
            if result["type"] == product_type
        ]

        axes = figure.add_subplot(1, 3, index)

        names = []
        scores = []

        for result in type_results:
            short_name = result["name"]
            short_name = short_name.replace("Холодильная камера ", "")
            short_name = short_name.replace("Морозильная установка ", "")
            short_name = short_name.replace("Сплит-система ", "")

            names.append(short_name)
            scores.append(result["quality_score"])

        legend_labels = [
            f"{names[i]} - {scores[i]:.3f}"
            for i in range(len(names))
        ]

        wedges, _ = axes.pie(
            scores,
            startangle=90,
            colors=colors[:len(scores)],
            wedgeprops={"width": 0.42, "edgecolor": "white"}
        )

        best_score = max(scores)

        axes.text(
            0,
            0,
            f"max Q\n{best_score:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

        axes.set_title(product_type, fontsize=11)

        axes.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=7,
            frameon=False
        )

    figure.suptitle(
        "Распределение показателя качества Q по группам продукции",
        fontsize=13
    )
    figure.tight_layout()

    canvas = FigureCanvasTkAgg(figure, parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    



def create_product_profile_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Радиальная диаграмма характеристик товара",
        font=TITLE_FONT
    )
    title.pack(pady=10)

    product_types = sorted(set(product["type"] for product in products))
    selected_type = tk.StringVar(value=product_types[0])
    product_query = tk.StringVar()

    controls_frame = ttk.LabelFrame(parent, text="Выбор товара")
    controls_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(controls_frame, text="Тип продукции:").grid(
        row=0,
        column=0,
        padx=5,
        pady=5,
        sticky="w"
    )

    type_box = ttk.Combobox(
        controls_frame,
        textvariable=selected_type,
        values=product_types,
        state="readonly",
        width=35
    )
    type_box.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    ttk.Label(controls_frame, text="Товар:").grid(
        row=1,
        column=0,
        padx=5,
        pady=5,
        sticky="w"
    )

    product_entry = ttk.Entry(
        controls_frame,
        textvariable=product_query,
        width=55
    )
    product_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    dropdown_button = ttk.Button(
        controls_frame,
        text="▼",
        width=3
    )
    dropdown_button.grid(row=1, column=2, padx=2, pady=5, sticky="w")

    dropdown_frame = ttk.Frame(controls_frame)
    dropdown_frame.grid(row=2, column=1, columnspan=2, padx=5, sticky="w")
    dropdown_frame.grid_remove()

    product_listbox = tk.Listbox(
        dropdown_frame,
        height=7,
        width=58,
        exportselection=False
    )
    product_listbox.grid(row=0, column=0, sticky="nsew")

    list_scrollbar = ttk.Scrollbar(
        dropdown_frame,
        orient="vertical",
        command=product_listbox.yview
    )
    list_scrollbar.grid(row=0, column=1, sticky="ns")

    product_listbox.configure(yscrollcommand=list_scrollbar.set)

    chart_frame = ttk.LabelFrame(parent, text="Профиль характеристик")
    chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def get_products_by_type():
        filtered_products = [
            product for product in products
            if product["type"] == selected_type.get()
        ]

        return sorted(filtered_products, key=lambda product: product["name"])

    def get_product_by_name(product_name):
        for product in products:
            same_type = product["type"] == selected_type.get()
            same_name = product["name"] == product_name

            if same_type and same_name:
                return product

        return None

    def clear_chart_with_message(message):
        for widget in chart_frame.winfo_children():
            widget.destroy()

        ttk.Label(
            chart_frame,
            text=message,
            font=("Arial", 14)
        ).pack(pady=30)

    def get_filtered_product_names():
        typed_text = product_query.get().lower()

        filtered_products = [
            product for product in get_products_by_type()
            if typed_text in product["name"].lower()
        ]

        return [product["name"] for product in filtered_products]

    def update_dropdown():
        product_names = get_filtered_product_names()

        product_listbox.delete(0, tk.END)

        for name in product_names:
            product_listbox.insert(tk.END, name)

        if product_names:
            dropdown_frame.grid()
        else:
            dropdown_frame.grid_remove()

    def show_profile(product):
        for widget in chart_frame.winfo_children():
            widget.destroy()

        metric_names, values = get_normalized_product_values(product, products)

        values.append(values[0])
        angles = []

        for index in range(len(metric_names)):
            angle = 2 * 3.14159 * index / len(metric_names)
            angles.append(angle)

        angles.append(angles[0])

        figure = Figure(figsize=(6, 5), dpi=100)
        axes = figure.add_subplot(111, polar=True)

        axes.plot(angles, values, linewidth=2)
        axes.fill(angles, values, alpha=0.25)

        axes.set_xticks(angles[:-1])
        axes.set_xticklabels(metric_names, fontsize=8)

        axes.set_ylim(0, 1)
        axes.set_title(product["name"], fontsize=12)
        axes.grid(True)

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def select_product_from_list(event=None):
        selected_indexes = product_listbox.curselection()

        if not selected_indexes:
            return

        product_name = product_listbox.get(selected_indexes[0])
        product_query.set(product_name)
        dropdown_frame.grid_remove()

        product = get_product_by_name(product_name)

        if product is not None:
            show_profile(product)

    def on_product_typing(event=None):
        update_dropdown()

        product = get_product_by_name(product_query.get())

        if product is not None:
            show_profile(product)
        elif not get_filtered_product_names():
            clear_chart_with_message("Товар не найден")

    def open_dropdown(event=None):
        update_dropdown()
        product_entry.focus_set()

    def reset_products_for_type(event=None):
        type_products = get_products_by_type()

        if not type_products:
            product_query.set("")
            dropdown_frame.grid_remove()
            clear_chart_with_message("Для выбранного типа товары не найдены")
            return

        first_product = type_products[0]
        product_query.set(first_product["name"])
        dropdown_frame.grid_remove()
        show_profile(first_product)

    def select_first_from_dropdown(event=None):
        if product_listbox.size() == 0:
            return

        product_listbox.selection_clear(0, tk.END)
        product_listbox.selection_set(0)
        select_product_from_list()

    type_box.bind("<<ComboboxSelected>>", reset_products_for_type)

    product_entry.bind("<FocusIn>", open_dropdown)
    product_entry.bind("<Button-1>", open_dropdown)
    product_entry.bind("<KeyRelease>", on_product_typing)
    product_entry.bind("<Return>", select_first_from_dropdown)

    dropdown_button.configure(command=open_dropdown)

    product_listbox.bind("<<ListboxSelect>>", select_product_from_list)
    product_listbox.bind("<Return>", select_product_from_list)

    reset_products_for_type()


def create_demand_tab(parent):
    title = ttk.Label(
        parent,
        text="Расчёт спроса на холодильное оборудование",
        font=TITLE_FONT
    )
    title.pack(pady=10)

    model_info = {"data": None}

    dataset_frame = ttk.LabelFrame(parent, text="Датасет для обучения модели")
    dataset_frame.pack(fill="x", padx=10, pady=8)

    dataset_label = ttk.Label(
        dataset_frame,
        text=f"Файл: {DEMAND_DATA_PATH}"
    )
    dataset_label.pack(side="left", padx=10, pady=8)

    ttk.Button(
        dataset_frame,
        text="Загрузить CSV",
        command=lambda: choose_csv_file()
    ).pack(side="right", padx=10, pady=8)

    metrics_frame = ttk.LabelFrame(parent, text="Качество модели")
    metrics_frame.pack(fill="x", padx=10, pady=8)

    metrics_label = ttk.Label(metrics_frame, text="", justify="left")
    metrics_label.pack(fill="x", padx=10, pady=8)

    input_frame = ttk.LabelFrame(parent, text="Входные параметры")
    input_frame.pack(fill="x", padx=10, pady=8)

    input_widgets = {}

    default_values = {
        "avg_refrigerator_price": "120",
        "energy_efficiency_class": "3",
        "is_seasonal_demand": "1",
        "advertising_budget": "500",
        "warranty_period": "5",
    }

    for row_index, feature in enumerate(DEMAND_FEATURES):
        ttk.Label(
            input_frame,
            text=DEMAND_FEATURE_LABELS[feature] + ":"
        ).grid(row=row_index, column=0, padx=8, pady=5, sticky="w")

        if feature == "energy_efficiency_class":
            widget = ttk.Combobox(
                input_frame,
                values=["1", "2", "3"],
                state="readonly",
                width=24
            )
            widget.set(default_values[feature])
        elif feature == "is_seasonal_demand":
            widget = ttk.Combobox(
                input_frame,
                values=["0", "1"],
                state="readonly",
                width=24
            )
            widget.set(default_values[feature])
        else:
            widget = ttk.Entry(input_frame, width=27)
            widget.insert(0, default_values[feature])

        widget.grid(row=row_index, column=1, padx=8, pady=5, sticky="w")
        input_widgets[feature] = widget

    result_label = ttk.Label(
        input_frame,
        text="Введите параметры и нажмите «Рассчитать спрос».",
        font=("Arial", 12, "bold")
    )
    result_label.grid(
        row=0,
        column=2,
        rowspan=2,
        padx=16,
        pady=5,
        sticky="w"
    )

    ttk.Button(
        input_frame,
        text="Рассчитать спрос",
        command=lambda: calculate_demand()
    ).grid(
        row=len(DEMAND_FEATURES),
        column=0,
        columnspan=2,
        padx=8,
        pady=10,
        sticky="w"
    )

    chart_frame = ttk.LabelFrame(parent, text="Фактические и прогнозные значения на тестовой выборке")
    chart_frame.pack(fill="both", expand=True, padx=10, pady=8)

    def draw_demand_chart():
        for widget in chart_frame.winfo_children():
            widget.destroy()

        current_model = model_info["data"]

        if current_model is None:
            return

        actual_values = current_model["test_actual"]
        predicted_values = current_model["test_predictions"]
        x_values = list(range(1, len(actual_values) + 1))

        figure = Figure(figsize=(8, 3.4), dpi=100)
        axes = figure.add_subplot(111)

        axes.plot(
            x_values,
            actual_values,
            marker="o",
            label="Фактические продажи"
        )
        axes.plot(
            x_values,
            predicted_values,
            marker="s",
            label="Прогноз модели"
        )

        axes.set_xlabel("Наблюдение тестовой выборки")
        axes.set_ylabel("Продажи, ед.")
        axes.set_title("Проверка модели линейной регрессии")
        axes.grid(True)
        axes.legend()
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_metrics():
        current_model = model_info["data"]

        if current_model is None:
            metrics_label.config(text="")
            return

        metrics_label.config(
            text=(
                f"Модель: {current_model['model_name']}\n"
                f"Количество строк в датасете: {len(current_model['rows'])}\n"
                f"MAE: {current_model['mae']:.2f}\n"
                f"RMSE: {current_model['rmse']:.2f}\n"
                f"R2: {current_model['r2']:.4f}"
            )
        )

    def train_from_csv(csv_path):
        try:
            model_info["data"] = train_demand_model(csv_path)
            dataset_label.config(text=f"Файл: {csv_path}")
            update_metrics()
            draw_demand_chart()
        except Exception as error:
            model_info["data"] = None
            messagebox.showerror("Ошибка загрузки датасета", str(error))

    def choose_csv_file():
        file_path = filedialog.askopenfilename(
            title="Выберите CSV-файл",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ]
        )

        if file_path:
            train_from_csv(Path(file_path))

    def calculate_demand():
        current_model = model_info["data"]

        if current_model is None:
            messagebox.showerror("Ошибка", "Сначала загрузите датасет.")
            return

        try:
            input_values = {
                feature: input_widgets[feature].get()
                for feature in DEMAND_FEATURES
            }
            prediction = predict_monthly_demand(current_model, input_values)
            result_label.config(
                text=f"Прогнозируемый месячный объём продаж: {prediction} ед."
            )
        except ValueError:
            messagebox.showerror(
                "Ошибка ввода",
                "Все входные параметры должны быть числовыми."
            )

    train_from_csv(DEMAND_DATA_PATH)


def create_info_tab(parent):
    title = ttk.Label(
        parent,
        text="Информация о программе",
        font=TITLE_FONT
    )
    title.pack(pady=10)

    info_frame = ttk.LabelFrame(parent, text="Назначение и методика")
    info_frame.pack(fill="both", expand=True, padx=16, pady=12)

    info_text = (
        "Информационно-аналитическая система разработана для предприятия "
        "«Производитель холодильного оборудования».\n\n"
        "В программе анализируются три типа продукции: холодильные камеры, "
        "морозильные установки и сплит-системы.\n\n"
        "Оценка технического уровня выполняется по квалиметрической модели. "
        "Итоговый показатель качества Q рассчитывается как сумма произведений "
        "нормированных показателей на весовые коэффициенты:\n\n"
        "Q = Σ(weight * q)\n\n"
        "Для показателей, где большее значение лучше, используется нормирование "
        "q = x / max(x). Для показателей, где меньшее значение лучше, используется "
        "q = min(x) / x.\n\n"
        "Прогноз продаж строится по линейному тренду на основе истории продаж за "
        "предыдущие годы.\n\n"
        "Оптимизация подбирает набор оборудования выбранного типа с учетом бюджета "
        "и минимального требуемого объема."
    )

    ttk.Label(
        info_frame,
        text=info_text,
        wraplength=980,
        justify="left"
    ).pack(fill="x", padx=14, pady=12)

    weights_frame = ttk.LabelFrame(parent, text="Весовые коэффициенты")
    weights_frame.pack(fill="x", padx=16, pady=8)

    for row_index, metric in enumerate(QUALITY_METRICS.values()):
        ttk.Label(
            weights_frame,
            text=metric["name"]
        ).grid(row=row_index, column=0, padx=12, pady=4, sticky="w")

        ttk.Label(
            weights_frame,
            text=str(metric["weight"])
        ).grid(row=row_index, column=1, padx=12, pady=4, sticky="w")


def main():
    products = load_products()

    root = tk.Tk()
    root.title("Информационно-аналитическая система")
    root.geometry("1200x720")

    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True)

    data_tab = ttk.Frame(tabs)
    quality_tab = ttk.Frame(tabs)
    profile_tab = ttk.Frame(tabs)
    forecast_tab = ttk.Frame(tabs)
    demand_tab = ttk.Frame(tabs)
    optimization_tab = ttk.Frame(tabs)
    info_tab = ttk.Frame(tabs)

    tabs.add(data_tab, text="Данные")
    tabs.add(quality_tab, text="Оценка качества")
    tabs.add(profile_tab, text="Профиль товара")
    tabs.add(forecast_tab, text="Прогноз")
    tabs.add(demand_tab, text="Расчёт спроса")
    tabs.add(optimization_tab, text="Оптимизация")
    tabs.add(info_tab, text="Информация")


    create_data_tab(data_tab, products)
    create_quality_tab(quality_tab, products)
    create_product_profile_tab(profile_tab, products)
    create_forecast_tab(forecast_tab, products)
    create_demand_tab(demand_tab)
    create_optimization_tab(optimization_tab, products)
    create_info_tab(info_tab)


    root.mainloop()


if __name__ == "__main__":
    main()
