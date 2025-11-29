import re
from collections import Counter
import matplotlib.pyplot as plt


def analyze_zipf_optimized(filename: str, top_n: int | None = None) -> None:
 
    # 1. Читаем текст
    with open(filename, encoding="utf-8") as f:
        text = f.read().lower()

    # 2. Выделяем слова (русские и английские)
    words = re.findall(r"[а-яa-zё]+", text)

    if not words:
        print("В файле не найдено ни одного слова.")
        return

    # 3. Частоты слов
    freq = Counter(words)

    # 4. Сортируем по убыванию частоты
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        items = items[:top_n]

    # 5. Формируем списки рангов и частот
    ranks = []
    freqs = []

    for rank, (_, f_val) in enumerate(items, start=1):
        ranks.append(rank)
        freqs.append(f_val)

    # 6. Находим оптимальную константу C по формуле МНК:
    #    C* = sum(F_i / R_i) / sum(1 / R_i^2)
    numerator = sum(f / r for f, r in zip(freqs, ranks))
    denominator = sum(1 / (r * r) for r in ranks)
    C_opt = numerator / denominator

    # 7. Считаем теоретические частоты по закону Зипфа с C_opt
    freqs_theor = [C_opt / r for r in ranks]

    # 8. Считаем суммарное квадратичное отклонение
    sse = sum((f_exp - f_th) ** 2 for f_exp, f_th in zip(freqs, freqs_theor))

    # 9. Печатаем таблицу
    print(f"{'R':>4} {'Слово':<15} {'F_эксп':>10} {'F_теор':>10} {'F_эксп-F_теор':>15}")
    print("-" * 60)
    for (word, f_exp), r, f_th in zip(items, ranks, freqs_theor):
        diff = f_exp - f_th
        print(f"{r:4d} {word:<15} {f_exp:10.0f} {f_th:10.1f} {diff:15.1f}")

    print("\nОптимальная константа C (МНК):", C_opt)
    print("Сумма квадратов отклонений (SSE):", sse)

    # 10. Строим график
    plt.figure()
    # экспериментальные точки
    plt.scatter(ranks, freqs, label="экспериментальные данные")
    # теоретическая кривая
    plt.plot(ranks, freqs_theor, label=f"модель Зипфа, C={C_opt:.1f}")
    plt.xlabel("Ранг слова R")
    plt.ylabel("Частота F")
    plt.title("Закон Зипфа: эксперимент и модель")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = "text.txt" 
    top_n_str = input(
        "Введите количество слов для анализа (Enter — все): "
    ).strip()

    top_n = int(top_n_str) if top_n_str else None
    analyze_zipf_optimized(filename, top_n)
