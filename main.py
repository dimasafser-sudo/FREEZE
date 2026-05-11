import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from itertools import combinations




BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "products.json"


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


def create_product_profile_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Радиальная диаграмма характеристик товара",
        font=("Arial", 16)
    )
    title.pack(pady=10)

    product_names = [product["name"] for product in products]
    selected_product = tk.StringVar(value=product_names[0])

    ttk.Label(parent, text="Выберите товар:").pack(pady=5)

    product_box = ttk.Combobox(
        parent,
        textvariable=selected_product,
        values=product_names,
        state="readonly",
        width=55
    )
    product_box.pack(pady=5)

    chart_frame = ttk.Frame(parent)
    chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def show_profile():
        for widget in chart_frame.winfo_children():
            widget.destroy()

        product = next(
            item for item in products
            if item["name"] == selected_product.get()
        )

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

    ttk.Button(
        parent,
        text="Построить диаграмму",
        command=show_profile
    ).pack(pady=10)

    show_profile()



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
        font=("Arial", 16)
    )
    title.pack(pady=10)

    product_types = sorted(set(product["type"] for product in products))
    selected_type = tk.StringVar(value=product_types[0])

    ttk.Label(parent, text="Тип продукции:").pack(pady=5)

    type_box = ttk.Combobox(
        parent,
        textvariable=selected_type,
        values=product_types,
        state="readonly",
        width=40
    )
    type_box.pack(pady=5)

    ttk.Label(parent, text="Бюджет, руб.:").pack(pady=5)

    budget_entry = ttk.Entry(parent, width=30)
    budget_entry.pack(pady=5)
    budget_entry.insert(0, "2000000")

    ttk.Label(parent, text="Минимальный требуемый объем, м³:").pack(pady=5)

    volume_entry = ttk.Entry(parent, width=30)
    volume_entry.pack(pady=5)
    volume_entry.insert(0, "200")

    result_label = ttk.Label(
        parent,
        text="",
        wraplength=950,
        justify="left"
    )
    result_label.pack(pady=10)

    columns = ("name", "quality", "volume", "price")

    result_table = ttk.Treeview(parent, columns=columns, show="headings", height=7)

    result_table.heading("name", text="Название")
    result_table.heading("quality", text="Q")
    result_table.heading("volume", text="Объем, м³")
    result_table.heading("price", text="Цена, руб.")

    result_table.column("name", width=360)
    result_table.column("quality", width=100)
    result_table.column("volume", width=120)
    result_table.column("price", width=140)

    result_table.pack(fill="x", padx=10, pady=10)

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
        parent,
        text="Подобрать оборудование",
        command=calculate_optimization
    ).pack(pady=10)


    def calculate_optimization():
        try:
            budget = float(budget_entry.get())
        except ValueError:
            result_label.config(text="Ошибка: введите числовое значение бюджета.")
            return

        if budget <= 0:
            result_label.config(text="Ошибка: бюджет должен быть больше нуля.")
            return

        best_product = find_best_product_by_budget(products, budget)

        if best_product is None:
            result_label.config(
                text="Нет продукции, которая укладывается в указанный бюджет."
            )
            return

        result_label.config(
            text=(
                f"Оптимальный выбор: {best_product['name']}\n"
                f"Тип: {best_product['type']}\n"
                f"Цена: {best_product['price']} руб.\n"
                f"Показатель качества Q: {round(best_product['quality_score'], 3)}"
            )
        )




def create_forecast_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Прогноз продаж продукции",
        font=("Arial", 16)
    )
    title.pack(pady=10)

    product_names = [product["name"] for product in products]

    selected_product = tk.StringVar(value=product_names[0])

    ttk.Label(parent, text="Выберите продукцию:").pack(pady=5)

    product_box = ttk.Combobox(
        parent,
        textvariable=selected_product,
        values=product_names,
        state="readonly",
        width=50
    )
    product_box.pack(pady=5)

    result_label = ttk.Label(parent, text="")
    result_label.pack(pady=10)

    chart_frame = ttk.Frame(parent)
    chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def show_forecast():
        for widget in chart_frame.winfo_children():
            widget.destroy()

        product = next(
            item for item in products
            if item["name"] == selected_product.get()
        )

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
        axes.set_title("Прогноз продаж")
        axes.set_xlabel("Год")
        axes.set_ylabel("Количество продаж")
        axes.grid(True)

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    ttk.Button(
        parent,
        text="Построить прогноз",
        command=show_forecast
    ).pack(pady=10)



def create_data_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Данные о холодильном оборудовании",
        font=("Arial", 16)
    )
    title.pack(pady=10)

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

    table_frame = ttk.Frame(parent)
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

    for product in products:
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


def create_quality_tab(parent, products):
    title = ttk.Label(
        parent,
        text="Оценка технического уровня продукции",
        font=("Arial", 16)
    )
    title.pack(pady=10)

    quality_results = calculate_quality(products)

    columns = ("place", "name", "type", "quality_score", "price")

    table_frame = ttk.Frame(parent)
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

    for index, result in enumerate(quality_results, start=1):
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

    



def main():
    products = load_products()

    root = tk.Tk()
    root.title("Информационно-аналитическая система")
    root.geometry("1100x650")

    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True)
    profile_tab = ttk.Frame(tabs)

    data_tab = ttk.Frame(tabs)
    quality_tab = ttk.Frame(tabs)
    forecast_tab = ttk.Frame(tabs)
    optimization_tab = ttk.Frame(tabs)

    tabs.add(data_tab, text="Данные")
    tabs.add(quality_tab, text="Оценка качества")
    tabs.add(forecast_tab, text="Прогноз")
    tabs.add(optimization_tab, text="Оптимизация")
    tabs.add(profile_tab, text="Профиль товара")


    create_product_profile_tab(profile_tab, products)
    create_data_tab(data_tab, products)
    create_quality_tab(quality_tab, products)
    create_forecast_tab(forecast_tab, products)
    create_optimization_tab(optimization_tab, products)


    root.mainloop()


if __name__ == "__main__":
    main()
