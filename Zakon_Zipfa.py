import os
import sys
import re
from collections import Counter
import matplotlib.pyplot as plt

# Регулярка для слов (русские + латиница)
WORD_RE = re.compile(r"[а-яa-zё]+", re.IGNORECASE)

# Набор русских предлогов и союзов, которые исключаем из анализа
STOPWORDS_RU = {
    # простые союзы
    "и", "а", "но", "или", "либо", "да", "то", "что", "чтоб", "чтобы",
    "как", "если", "когда", "хотя", "однако", "потому", "зато",
    # сложные/составные можно частично учитывать
    "тоесть", "тоесть", "таккак", "потомучто",
    # предлоги
    "в", "во", "на", "с", "со", "к", "ко", "о", "об", "обо",
    "от", "до", "из", "изо", "перед", "передо", "за", "под", "подо",
    "над", "надо", "при", "для", "через", "между", "по", "у", "без",
    "около", "возле", "при", "напод", "про",
    # часто встречающиеся служебные слова (частицы и т.п.), можно при желании убрать
    "же", "ли", "бы", "то", "вот"
}


def load_text(path: str) -> str:
    """
    Чтение текста из файла с попыткой нескольких кодировок.
    Поддерживаем utf-8 / utf-8-sig / cp1251.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1251"]
    last_error = None

    for enc in encodings:
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e
            continue

    # Крайний случай: читаем как utf-8 с заменой битых символов
    print(f"Предупреждение: проблемы с кодировкой файла {path}: {last_error}")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_words(text: str):
    """
    Извлекаем слова и фильтруем предлоги/союзы.
    """
    words = WORD_RE.findall(text.lower())
    # фильтрация стоп-слов
    words = [w for w in words if w not in STOPWORDS_RU]
    return words


def compute_zipf_C_opt(words, top_n=200):
    """
    Считаем частоты, ранги и подбираем C по МНК для модели F = C / R.
    Возвращаем:
      C_opt, ranks, freqs_exp, freqs_theor, sse, items
    где items — список (слово, частота) в порядке убывания частоты.
    """
    freq = Counter(words)
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        items = items[:top_n]

    ranks = []
    freqs_exp = []

    for rank, (_, f_val) in enumerate(items, start=1):
        ranks.append(rank)
        freqs_exp.append(f_val)

    if not ranks:
        return 0.0, [], [], [], 0.0, []

    numerator = sum(f / r for f, r in zip(freqs_exp, ranks))
    denominator = sum(1 / (r * r) for r in ranks)
    C_opt = numerator / denominator if denominator != 0 else 0.0

    freqs_theor = [C_opt / r for r in ranks]
    sse = sum((fe - ft) ** 2 for fe, ft in zip(freqs_exp, freqs_theor))

    return C_opt, ranks, freqs_exp, freqs_theor, sse, items


def print_zipf_table(filename, C_opt, sse, items, ranks, freqs_exp, freqs_theor):
    """
    Печать таблицы в формате, как у тебя на скриншоте.
    """
    print("\nФайл:", filename)
    print(f"Оптимальная константа C (МНК): {C_opt:.4f}")
    print(f"Сумма квадратов отклонений (SSE): {sse:.4f}\n")

    print(f"{'R':>3} {'Слово':<12} {'F_эксп':>8} {'F_теор':>10} {'F_эксп-F_теор':>13}")
    print("-" * 50)

    for (word, _), r, f_exp, f_th in zip(items, ranks, freqs_exp, freqs_theor):
        diff = f_exp - f_th
        print(f"{r:>3} {word:<12} {f_exp:>8.0f} {f_th:>10.1f} {diff:>13.1f}")


def list_text_files(folder="text"):
    """
    Возвращает список путей к .txt файлам в папке folder.
    """
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(".txt"):
            files.append(path)
    files.sort()
    return files


def choose_files_interactively(files):
    """
    Позволяет выбрать несколько файлов по номерам.
    """
    print("Найдены текстовые файлы:")
    for i, f in enumerate(files, start=1):
        print(f"  {i}. {os.path.basename(f)}")

    print("\nВведите номера файлов через пробел (например: 1 3 4)")
    raw = input("Выбор: ").strip()

    if not raw:
        return []

    idxs = []
    for part in raw.split():
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(files):
                idxs.append(idx - 1)

    seen = set()
    selected = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            selected.append(files[i])

    return selected


def compare_texts(paths, top_n=200):
    """
    Обрабатываем несколько текстов, печатаем таблицы
    и возвращаем результаты для сводной таблицы и графика.
    """
    results = []

    for path in paths:
        text = load_text(path)
        words = extract_words(text)

        C_opt, ranks, freqs_exp, freqs_theor, sse, items = compute_zipf_C_opt(
            words, top_n=top_n
        )

        print_zipf_table(os.path.basename(path), C_opt, sse,
                         items, ranks, freqs_exp, freqs_theor)

        results.append({
            "path": path,
            "name": os.path.basename(path),
            "C_opt": C_opt,
            "sse": sse,
            "ranks": ranks,
            "freqs_exp": freqs_exp,
            "freqs_theor": freqs_theor,
            "total_words": len(words),
            "unique_words": len(set(words)),
        })

    # Сводная таблица
    print("\nСравнение текстов по константе Зипфа (после фильтрации предлогов и союзов)")
    print(f"{'Файл':<30} {'Слов':>10} {'Уник':>10} {'top_n':>7} {'C_opt':>12} {'SSE':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30} {r['total_words']:>10} {r['unique_words']:>10} {top_n:>7} "
              f"{r['C_opt']:>12.2f} {r['sse']:>12.2f}")

    return results


def plot_comparison(results):
    """
    Строим график: точки — эксперимент, линии — Zipf для каждого текста.
    """
    if not results:
        return

    plt.figure()
    for r in results:
        if not r["ranks"]:
            continue
        plt.scatter(r["ranks"], r["freqs_exp"], label=f"{r['name']} (exp)", s=20)
        plt.plot(r["ranks"], r["freqs_theor"], label=f"{r['name']} (Zipf)")

    plt.xlabel("Ранг слова R")
    plt.ylabel("Частота F")
    plt.title("Сравнение текстов по закону Зипфа (без предлогов и союзов)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Вариант 1: пути переданы через аргументы командной строки
    if len(sys.argv) > 1:
        paths = [p for p in sys.argv[1:] if os.path.isfile(p)]
        if not paths:
            print("Указанные файлы не найдены.")
            return
    else:
        # Вариант 2: выбираем файлы из папки text/
        folder = "text"
        files = list_text_files(folder)

        if not files:
            print(f"В папке '{folder}' не найдено .txt файлов.")
            print("Создайте папку 'text' и положите в неё тексты в формате .txt")
            return

        paths = choose_files_interactively(files)
        if not paths:
            print("Файлы не выбраны.")
            return

    top_n_str = input("Сколько самых частотных слов брать для оценки C (Enter = 200): ").strip()
    top_n = int(top_n_str) if top_n_str else 200

    results = compare_texts(paths, top_n=top_n)
    plot_comparison(results)


if __name__ == "__main__":
    main()


