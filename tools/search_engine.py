
# tools/search_engine.py
# Класс для удобного поиска и фильтрации по итоговому размеченному датасету ассоциаций.

import pandas as pd
import os
from typing import Optional

class AssociativeSearch:
    """
    Интерфейс для поиска и фильтрации пар "стимул-ассоциация" в финальном датасете с разметкой.

    Предполагается, что данные уже предварительно очищены (приведение к нижнему регистру
    выполнено на этапе подготовки).
    """

    def __init__(self, file_path: str = "data/0_final_dataset.csv") -> None:
        """
        Инициализация и загрузка базы данных.

        Параметры
        ----------
        file_path : str
            Путь к CSV-файлу с итоговым датасетом. Если файл не найден, проверяется
            путь на уровень выше (удобно при импорте из подпапок).
        """
        if not os.path.exists(file_path):
            alt_path = os.path.join("..", file_path)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Файл данных не найден: {file_path}")

        self.df = pd.read_csv(file_path)
        print(f"Загружено {len(self.df)} записей.")

    def get_reactions(self, stimulus: str, sort_by: str = 'частота', top_n: int = 10) -> pd.DataFrame:
        """
        Найти все ассоциации на конкретный стимул.

        Параметры
        ----------
        stimulus : str
            Стимул для поиска (регистр не важен, т.к. данные уже в нижнем регистре).
        sort_by : str
            Колонка для сортировки (по умолчанию 'частота').
        top_n : int
            Количество возвращаемых записей.

        Возвращает
        -------
        pd.DataFrame с ассоциациями, отсортированными по убыванию `sort_by`.
        """
        res = self.df[self.df['стимул'] == stimulus.lower()]
        return res.sort_values(by=sort_by, ascending=False).head(top_n)

    def reverse_search(self, word: str, use_lemma: bool = True) -> pd.DataFrame:
        """
        Обратный поиск: найти все стимулы к указанной реакции.

        Параметры
        ----------
        word : str
            Искомое слово (ассоциация или лемма).
        use_lemma : bool
            Если True, поиск ведётся по колонке 'лемма', иначе – по колонке 'ассоциация'.

        Возвращает
        -------
        pd.DataFrame со стимулами, которые вызвали указанную реакцию.
        """
        col = 'лемма' if use_lemma else 'ассоциация'
        return self.df[self.df[col] == word.lower()]

    def filter_by_relation(self, relation_type: str, min_pmi: Optional[float] = None) -> pd.DataFrame:
        """
        Поиск пар по типу связи (например, 'синтагматический') и порогу PMI.

        Параметры
        ----------
        relation_type : str
            Тип связи (или его часть) для поиска. Регистр не учитывается.
        min_pmi : float, optional
            Минимальное значение PMI. Если None, фильтр не применяется.

        Возвращает
        -------
        pd.DataFrame с отфильтрованными строками.
        """
        mask = self.df['тип связи'].str.contains(relation_type, case=False, na=False)
        if min_pmi is not None:
            mask &= (self.df['PMI'] >= min_pmi)
        return self.df[mask]

    def get_semantic_neighbors(self, tag: str, source: str = 'НКРЯ') -> pd.DataFrame:
        """
        Поиск слов из одной семантической группы (НКРЯ или общая ЛСГ).

        Параметры
        ----------
        tag : str
            Тег семантической группы для поиска (например, 'время').
        source : str
            'НКРЯ' – поиск по колонке 'сем.группа стимула_НКРЯ или сем.группа ассоциации_НКРЯ';
            'ЛСГ' – поиск по колонке 'общая ЛСГ'.

        Возвращает
        -------
        pd.DataFrame с парами, у которых стимул принадлежит указанной группе.
        """
        col = 'сем.группа стимула_НКРЯ' if source == 'НКРЯ' else 'общая ЛСГ'
        return self.df[self.df[col].str.contains(tag, na=False)]

    def get_strong_links(self, threshold: float = 0.7,
                         metric: str = 'косинусное сходство_Sentence‑BERT') -> pd.DataFrame:
        """
        Выборка самых сильных связей по выбранной метрике.

        Параметры
        ----------
        threshold : float
            Минимальное значение метрики (по умолчанию 0.7).
        metric : str
            Название колонки с метрикой. По умолчанию используется косинусное сходство.

        Возвращает
        -------
        pd.DataFrame, отсортированный по убыванию метрики.
        """
        return self.df[self.df[metric] >= threshold].sort_values(by=metric, ascending=False)