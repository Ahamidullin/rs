"""Build the Russian-language defense decks from the archived and computed results."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


INK = "#172332"
MUTED = "#46586D"
BLUE = "#2152A1"
TEAL = "#00857B"
LIGHT = "#EAF1FA"
BG = "#FCFDFF"
DATASETS = ["industry11", "international9", "factors3", "ff21", "ff24", "ff28", "industry52"]
LABELS = ["10 отраслей\n+ рынок", "Страны\n+ мир", "Три\nфактора", "FF-1", "FF-4", "FF25\n+ FF3", "49 отраслей\n+ FF3"]


def page(title, header, source):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.065, 0.933, header, fontsize=16, color=BLUE, weight="bold")
    fig.text(0.065, 0.845, title, fontsize=37, color=INK, weight="bold")
    fig.text(0.065, 0.068, source, fontsize=13, color=MUTED)
    return fig


def label(fig, x, y, value, size=19, color=INK, **kwargs):
    fig.text(x, y, value, fontsize=size, color=color, linespacing=1.45, **kwargs)


def intro():
    fig = page("Когда 1/N помогает Марковицу", "MARKOWITZ MEETS TALMUD",
               "Tu и Zhou (2011), Journal of Financial Economics, 99, с. 204–215.")
    label(fig, 0.065, 0.73, "Ошибка оценки средних и ковариации может привести\nк большим и нестабильным позициям.", 24)
    label(fig, 0.065, 0.53, r"$w=(1-\delta)e+\delta v$", 55, BLUE)
    label(fig, 0.065, 0.41, "e — равные веса 1/N    •    v — оценочный портфель", 21, MUTED)
    label(fig, 0.065, 0.26, "1. Улучшает ли смесь свой оценочный компонент?\n2. В каких выборках смесь обгоняет 1/N?", 24, weight="bold")
    return fig


def methods():
    fig = page("Четыре оценочных компонента", "СТАТЬЯ · МЕТОДЫ",
               "Tu и Zhou (2011), §2.1–2.5, уравнения (1), (7), (12), (15), (18).")
    for i, (abbr, text) in enumerate((
        ("ML", "Выборочные средние и обратная ковариация."),
        ("PJ", "Средние сжимаются к оценке доходности GMV."),
        ("MP", "Ковариация связана со структурой средних."),
        ("KZ", "Касательный портфель и портфель минимальной дисперсии."),
    )):
        y = 0.71 - i * 0.092
        label(fig, 0.065, y, abbr, 23, BLUE, weight="bold")
        label(fig, 0.145, y, text, 23)
    label(fig, 0.065, 0.23, "Пример при δ = 0,25", 21, BLUE, weight="bold")
    label(fig, 0.065, 0.166, "0,75 × (0,5; 0,5) + 0,25 × (1,2; −0,1) = (0,675; 0,35).  Cash = −2,5%.", 21)
    return fig


def simulations():
    fig = page("Смеси помогают при ошибке оценки", "ЧАСТЬ 2 · СИМУЛЯЦИИ",
               "Tu и Zhou (2011), табл. 1A и 2B. Парные симуляции команды, сноски 8–9.")
    label(fig, 0.065, 0.732, "Опубликованная полезность, годовые п.п. · T = 120, γ = 3", 21, MUTED)
    for i, (head, a, b) in enumerate((
        ("Сценарий", "1A · один фактор", "2B · три фактора"),
        ("1/N", "3,89", "3,85"), ("ML", "−85,72", "−81,09"), ("CKZ", "3,71", "5,09"),
    )):
        y = 0.635 - i * 0.087
        for x, value in zip((0.065, 0.35, 0.64), (head, a, b), strict=True):
            label(fig, x, y, value, 23, BLUE if i == 0 else INK, weight="bold" if i == 0 else "normal")
    label(fig, 0.065, 0.253, "Наша проверка для N = 50", 20, TEAL, weight="bold")
    label(fig, 0.065, 0.157, "CML, CPJ и CKZ выше 1/N с T = 480, CMP — с T = 3000.\n"
          "Для сносок использованы собственные спецификации.", 19)
    return fig


def empirical_design():
    fig = page("Семь выборок, прогноз на месяц вперёд", "ЧАСТЬ 3 · БЭКТЕСТ",
               "Tu и Zhou (2011), §3.3, с. 211–214, уравнение (22).")
    label(fig, 0.065, 0.725, "Пример первого прогноза для industry11 и M = 120", 21, MUTED)
    label(fig, 0.065, 0.56, "Июль 1963 — июнь 1973\nОценка весов", 25, BLUE, weight="bold",
          bbox={"facecolor": LIGHT, "edgecolor": LIGHT, "pad": 14})
    label(fig, 0.485, 0.594, "→", 42, BLUE)
    label(fig, 0.60, 0.56, "Июль 1973\nРеализация доходности", 25, TEAL, weight="bold")
    label(fig, 0.065, 0.36, r"$CER_{\mathrm{год}}=1200\times(\overline{r}_{\mathrm{excess}}-1.5\,s^2)$", 31, INK)
    label(fig, 0.065, 0.22, "Два окна, 120 и 240 месяцев. Годовой CER в процентных пунктах.\n"
          "ML in-sample использует всю панель как ретроспективный ориентир.", 22)
    return fig


def empirical_plot(pivot):
    actual = int((pivot["computed"].ckz > pivot["computed"].ew).sum())
    published = int((pivot["published"].ckz > pivot["published"].ew).sum())
    cases = len(pivot)
    fig = page(f"CKZ выигрывает у 1/N в {actual} из {cases} случаев", "ЧАСТЬ 3 · ТАБЛИЦА 6",
               f"Tu и Zhou (2011), табл. 6A/B. В статье CKZ выше 1/N в {published} из {cases} случаев.")
    label(fig, 0.065, 0.744, "Разница CER между CKZ и 1/N, годовые п.п.", 21, MUTED)
    chart(fig, pivot)
    return fig


def chart(fig, pivot):
    for i, window in enumerate((120, 240)):
        ax = fig.add_axes((0.075 + 0.478 * i, 0.20, 0.402, 0.46))
        part = pivot.xs(window, level="window").reindex(DATASETS)
        x = np.arange(len(DATASETS))
        for j, (method, name, color) in enumerate((("published", "В статье", BLUE),
                                                  ("computed", "Наш расчёт", TEAL))):
            ax.bar(x + (j - 0.5) * 0.35, part[(method, "ckz")] - part[(method, "ew")],
                   0.34, color=color, label=name)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(x, LABELS, fontsize=11)
        ax.set_title(f"M = {window} месяцев", loc="left", pad=12, fontsize=19, color=INK, weight="bold")
        ax.set_ylim(-8, 27)
        ax.set_yticks([-5, 0, 5, 10, 15, 20, 25])
        ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=7)
        ax.set_facecolor(BG)
        ax.grid(axis="y", color="#DDE6EE")
        ax.set_axisbelow(True)
        ax.spines[:].set_visible(False)
        ax.legend(frameon=False, ncol=2, fontsize=12, loc="lower right", bbox_to_anchor=(1, 1.01))


def empirical_limits(matches, total):
    fig = page("140 ячеек и границы совпадения", "ЧАСТЬ 3 · ПРОВЕРКИ",
               "Источники, версии данных и парные проверки — docs/data.md и results/replication.")
    label(fig, 0.065, 0.698, "CMP50 против MP", 21, BLUE, weight="bold")
    label(fig, 0.065, 0.595, "14 / 14", 48, BLUE, weight="bold")
    label(fig, 0.51, 0.698, "CMP50 против 1/N", 21, TEAL, weight="bold")
    label(fig, 0.51, 0.595, "0 / 14", 48, TEAL, weight="bold")
    label(fig, 0.065, 0.465, f"Совпадение с округлением в {matches} из {total} ячеек.", 24, weight="bold")
    label(fig, 0.065, 0.321, "DGU · 10 из 10 совпадений для 1/N, 8 из 10 для rolling ML.\n"
          "French · два набора на FIZ 202412, исходные версии не найдены.", 21)
    label(fig, 0.065, 0.19, "CKZ пересчитан по формуле (13). Для CPJ сохраняется приближение.\n"
          "Проверены смена версий данных, интервалы и вариант CMP с jackknife.", 21)
    return fig


def main():
    table = pd.read_csv("results/replication/table6.csv")
    pivot = table.pivot(index=["case", "window"], columns="rule", values=["published", "computed"])
    empirical = [empirical_design(), empirical_plot(pivot),
                 empirical_limits(table.status.eq("match").sum(), len(table))]
    overview = [intro(), methods(), simulations()]
    for name, pages in (("slides.pdf", overview + empirical), ("part3_slides.pdf", empirical)):
        with PdfPages(Path("defense") / name) as pdf:
            for fig in pages:
                pdf.savefig(fig, facecolor=BG)
    plt.close("all")


if __name__ == "__main__":
    main()
