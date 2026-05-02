
import pandas as pd
import Levenshtein
from nltk.stem.snowball import RussianStemmer
import pymorphy3
import os
import numpy as np
import re
from collections import defaultdict
from tqdm.auto import tqdm
from functools import lru_cache
from ruwordnet import RuWordNet

"""## I. Однокоренные и фонетически сходные пары стимул-ассоциаций"""

stemmer = RussianStemmer()
morph = pymorphy3.MorphAnalyzer()

def classify_with_flag(word1: str, word2: str):


    if not word1 or not word2:
        return (None, False)
    if word1 == word2:
        return ("идентичные", False)

    p1 = morph.parse(word1)[0]
    p2 = morph.parse(word2)[0]
    lemma1, lemma2 = p1.normal_form, p2.normal_form

    # 1. Грамматическая связь (формы слова)
    if lemma1 == lemma2 and p1.tag.POS == p2.tag.POS:
        return ("грамматическая (формы слова)", False)

    # 2. Эпидигматика (словообразование)
    stem1 = stemmer.stem(word1)
    stem2 = stemmer.stem(word2)

    if stem1 == stem2:
        return ("эпидигматическая", False)

    if len(stem1) >= 3 and len(stem2) >= 3:
        if stem1 in lemma2 or stem2 in lemma1:
            if Levenshtein.distance(lemma1, lemma2) <= 4:
                return ("эпидигматическая", True)   # флаг для визуального контроля

    # 3. Фонетическое сходство
    dist = Levenshtein.distance(word1, word2)
    avg_len = (len(word1) + len(word2)) / 2
    rel_dist = dist / avg_len if avg_len > 0 else 0
    common_start = word1[:2] == word2[:2]
    common_end = word1[-2:] == word2[-2:]

    if dist <= 2 and rel_dist <= 0.25 and (common_start or common_end):
        need_check = False
        if dist == 1:
            need_check = True
        if len(word1) >= 4 and len(word2) >= 4 and word1[:4] == word2[:4]:
            need_check = True
        return ("фонетическая", need_check)

    # 4. Спорные случаи без чёткого тега
    need_check = False
    if dist == 1:
        need_check = True
    elif dist <= 3 and len(word1) >= 4 and len(word2) >= 4 and word1[:4] == word2[:4]:
        need_check = True
    elif dist <= 2 and rel_dist <= 0.25:
        need_check = True

    if need_check:
        return ("спорный_случай", True)

    return (None, False)


def load_and_process(input_path: str = "data/1_base_list.csv") -> pd.DataFrame:


    df = pd.read_csv(input_path)

    # Очистка текстовых полей
    df['стимул'] = df['стимул'].fillna('').astype(str).str.lower().str.strip()
    df['ассоциация'] = df['ассоциация'].fillna('').astype(str).str.lower().str.strip()

    # Классификация пар
    results = [classify_with_flag(w1, w2)
               for w1, w2 in zip(df['стимул'], df['ассоциация'])]
    df['тип_связи'] = [res[0] for res in results]
    df['нужна_проверка'] = [res[1] for res in results]

    return df



input_file = "data/1_base_list.csv"
df_result = load_and_process(input_file)

"""## II. Парадигматика. Разметка по RuWordNet"""

wn = RuWordNet()

@lru_cache(maxsize=50000)
def get_cached_senses(word):
    return wn.get_senses(word) if word else []

def get_wordnet_relations(row):
    # Пропускаем идентичные пары, уже размеченные ранее
    if row.get('эпидигматика и формальная связь') == 'идентичные':
        return None

    st_raw = str(row['стимул']).lower().strip()
    as_lem = str(row['лемма']).lower().strip()

    st_senses = get_cached_senses(st_raw)
    as_senses = get_cached_senses(as_lem)

    if not st_senses or not as_senses:
        return None

    as_synsets = {s.synset for s in as_senses}
    st_synsets = {s.synset for s in st_senses}

    found = []

    if not st_synsets.isdisjoint(as_synsets):
        found.append("синоним")

    rel_map = {
        "гипоним": "hyponyms", "гипероним": "hypernyms",
        "объект домена": "domain_items", "домен": "domains",
        "мероним": "meronyms", "холоним": "holonyms",
        "экземпляр": "instances", "класс": "classes",
        "вывод": "conclusions", "предпосылка": "premises",
        "следствие": "effects", "причина": "causes",
        "межкат_синоним": "pos_synonyms", "антоним": "antonyms",
        "прочая связь": "related"
    }

    for ss in st_synsets:
        for label, attr in rel_map.items():
            related_list = getattr(ss, attr, [])
            if any(rel in as_synsets for rel in related_list):
                found.append(label)

    if found:
        return ", ".join(sorted(set(found)))

    return None


df_result['RuWordNet'] = df_result.progress_apply(get_wordnet_relations, axis=1)

"""## III. Семантическая разметка (НКРЯ)"""

# Путь к папке с файлами семантической разметки НКРЯ
# Пользователь должен предварительно скачать фалы из НКРЯ и поместить в папку 'data/сем_разметка' внутри проекта.
SEM_ROOT = 'data/сем_разметка'

word_weights = {}

if not os.path.isdir(SEM_ROOT):
    print(f"Папка '{SEM_ROOT}' не найдена")
    print("Пожалуйста, скачайте файлы семантической разметки НКРЯ")
    print("и поместите их в эту папку, сохранив структуру подпапок")
    print("После этого перезапустите ячейку")
else:
    print(f"Сканирование папок и загрузка семантических весов из '{SEM_ROOT}'...")
for pos_folder in os.listdir(SEM_ROOT):
    pos_path = os.path.join(SEM_ROOT)

    if os.path.isdir(pos_path):
        csv_files = [f for f in os.listdir(pos_path) if f.endswith('.csv')]

        for filename in tqdm(csv_files, desc=f"Загрузка: {pos_folder}"):
            group_name = filename.replace('.csv', '').lower()
            unique_tag = f"{pos_folder.lower()}: {group_name}"
            file_path = os.path.join(pos_path, filename)

            try:
                temp_df = pd.read_csv(file_path, sep=';', quotechar='"', encoding='utf-8-sig')
                temp_df.columns = [c.strip().replace('"', '') for c in temp_df.columns]

                if 'word_0' in temp_df.columns and 'ipm' in temp_df.columns:
                    for _, row in temp_df.iterrows():
                        word = str(row['word_0']).lower().strip().replace('ё', 'е')
                        try:
                            ipm = float(row['ipm'])
                        except ValueError:
                            ipm = 0.0

                        if word not in word_weights:
                            word_weights[word] = []

                        word_weights[word].append((ipm, unique_tag))
            except Exception as e:
                print(f"Ошибка чтения файла {pos_folder}/{filename}: {e}")

# Функции обработки
def get_weighted_tags(word):
    if pd.isna(word):
        return

    word_clean = str(word).lower().strip().replace('ё', 'е')
    if word_clean not in word_weights:
        return

    # Сортировка по IPM (первый элемент) по убыванию, чтобы частые значения шли первыми
    sorted_data = sorted(word_weights[word_clean], key=lambda x: x[0], reverse=True)
    return ", ".join([item[1] for item in sorted_data])

def find_common_tags(row):

    s_tags = str(row.get('сем.группа стимула_НКРЯ', ''))
    a_tags = str(row.get('сем.группа ассоциации_НКРЯ', ''))

    if not s_tags or not a_tags:
        return None

    set_s = set([t.strip() for t in s_tags.split(',') if t.strip()])
    set_a = set([t.strip() for t in a_tags.split(',') if t.strip()])

    common = set_s.intersection(set_a)
    return ", ".join(list(common)) if common else None


tqdm.pandas(desc="Разметка слов")

df_result['сем.группа стимула_НКРЯ'] = df_result['стимул'].progress_apply(get_weighted_tags)
df_result['сем.группа ассоциации_НКРЯ'] = df_result['лемма'].progress_apply(get_weighted_tags)
df_result['общая ЛСГ'] = df_result.apply(find_common_tags, axis=1)

"""## IV. Фреймовая разметка (FrameBank)"""

# с учетом уже скаченных с репозитория FrameBank файлов

morph = pymorphy3.MorphAnalyzer()

# Маппинг POS
pos_map = {
    'S': 'S', 'A': 'A', 'V': 'V', 'ADV': 'ADV',
    'NUM': 'NUM', 'PR': 'PR', 'SPRO': 'S', 'APRO': 'A'
}

def fast_lemma(text):
    if pd.isna(text) or text == 'N/A' or text == '': return ""
    return morph.parse(str(text).lower().strip())[0].normal_form


path_prefix = "data/framebank/"

items_ex = pd.read_csv(path_prefix + "framebank_anno_ex_items.txt", sep='\t', low_memory=False)
circ_ex = pd.read_csv(path_prefix + "framebank_anno_ex_circ.txt", sep='\t', low_memory=False)
dict_cx = pd.read_csv(path_prefix + "framebank_dict_cx.txt", sep='\t', low_memory=False)
cx_items = pd.read_csv(path_prefix + "framebank_dict_cx_items.txt", sep='\t', low_memory=False)

# 1. Точные примерв
ex_roles = defaultdict(dict)


for _, row in items_ex.iterrows():
    lemma = fast_lemma(row['WordDep'])
    if lemma:
        role = str(row['Role']).strip() if pd.notna(row['Role']) else "-"
        ex_roles[row['ExIndex']][lemma] = role

for _, row in circ_ex.iterrows():
    lemma = fast_lemma(row['Phrase'])
    if lemma:
        role = str(row['Role']).strip() if pd.notna(row['Role']) else "-"
        ex_roles[row['ExIndex']][lemma] = role

# финальный exact_index: пара -> "Роль1 + Роль2 (ID:000)"
exact_index = defaultdict(list)


for ex_id, words_in_ex in ex_roles.items():
    lemmas = list(words_in_ex.keys())
    for i in range(len(lemmas)):
        for j in range(i + 1, len(lemmas)):
            l1, l2 = lemmas[i], lemmas[j]
            pair = tuple(sorted([l1, l2]))
            r1, r2 = words_in_ex[l1], words_in_ex[l2]
            desc = f"{r1} + {r2} (ID:{ex_id})"
            exact_index[pair].append(desc)

exact_index_final = {k: "; ".join(set(v)) for k, v in exact_index.items()}

# 2. Словарь конструкций
cx_names = dict(zip(dict_cx['ConstrIndex'], dict_cx['ConstrName']))
lex_to_cx = defaultdict(list)
for _, row in dict_cx.iterrows():
    cx_id = row['ConstrIndex']
    for lex in str(row['KeyLexemes']).split(','):
        norm_lex = fast_lemma(lex)
        if norm_lex:
            lex_to_cx[norm_lex].append(cx_id)

# 3. Роли конструкций
items_by_cx = defaultdict(list)
for _, row in cx_items.iterrows():
    items_by_cx[row['ConstrIndex']].append({
        'form': str(row['Form']).strip(),
        'role': str(row['Role']).strip()
    })

# Функция для разметки
def annotate_row_v4(row):
    # Исключение пар с тегом "идентичные"
    epi_val = str(row.get('эпидигматика и формальная связь', '')).strip().lower()
    if 'идентичные' in epi_val:
        return pd.Series(["", "", "", "", ""])

    s = str(row.get('стимул', '')).lower().strip()
    l = str(row.get('лемма', '')).lower().strip()

    pos_raw = row.get('часть речи ассоциации') or row.get('часть речи ассоциации') or ""
    pos_l = pos_map.get(str(pos_raw).upper().strip().split(',')[0])

    # 1. Корпус
    exact = exact_index_final.get(tuple(sorted([s, l])), "")

    # 2. Конструкции
    s_ids = set(lex_to_cx.get(s, []))
    l_ids = set(lex_to_cx.get(l, []))

    s_constrs = "; ".join([cx_names.get(i, "") for i in s_ids])
    l_constrs = "; ".join([cx_names.get(i, "") for i in l_ids])

    inter_ids = s_ids.intersection(l_ids)
    inter_names = "; ".join([cx_names.get(i, "") for i in inter_ids])

    # 3. Ролевая связь
    role_matches = []
    for cx_id in s_ids:
        c_name = cx_names.get(cx_id, "")
        for item in items_by_cx.get(cx_id, []):
            form = item['form']
            if l == form.lower() or (pos_l and re.search(rf'\b{pos_l}[a-z]*\b', form)):
                role_matches.append(f"{c_name} [{item['role']}]")

    roles = "; ".join(list(set(role_matches)))

    return pd.Series([exact, s_constrs, l_constrs, inter_names, roles])


tqdm.pandas()

fb_cols = ['FB_точные примеры', 'FB_констр стимула', 'FB_констр леммы', 'FB_пересечение', 'FB_ролевая связь']
df_result[fb_cols] = df_result.progress_apply(annotate_row_v4, axis=1)

"""## V. Синтагматка
* Синтаксическая разметка по НКРЯ
* Поиск устойчивых коллокаций по RuWordNet
"""

morph = pymorphy3.MorphAnalyzer()

STOP_PARADIGM = {
    'гипоним', 'гипероним', 'домен', 'мероним',
    'холоним', 'экземпляр', 'класс', 'межкат_синоним', 'антоним', 'синоним'
}
COPULAS = {'оказываться', 'являться', 'стать', 'казаться', 'оставаться'}

# извлекает первый POS-тег
def get_primary_pos(pos_str):

    if pd.isna(pos_str) or pos_str == '':
        return ''
    return str(pos_str).split('/')[0].strip().upper()

# исключения
def get_syntax_v5_fixed(row):

    epi_val = str(row.get('эпидигматика и формальная связь', '')).strip().lower()
    exclude_tags = ['идентичные', 'формы слова', 'грамматическая', 'фонетическая', 'графическая', 'историческая']

    for tag in exclude_tags:
        if tag in epi_val:
            return f'исключено (формальная: {tag})'


    wordnet_val = str(row.get('RuWordNet', '')).strip().lower()
    found_wn_tags = [p.strip() for p in wordnet_val.split(',') if p.strip() in STOP_PARADIGM]
    if found_wn_tags:
        return f'исключено (WordNet: {found_wn_tags[0]})'


    s_val = row.get('стимул', '')
    a_val = row.get('ассоциация', '')
    if pd.isna(s_val) or pd.isna(a_val):
        return "не выявлено"
    s_word = str(s_val).strip().lower()
    a_word = str(a_val).strip().lower()

    # берём первый POS-тег до слеша
    s_db_pos = get_primary_pos(row.get('часть речи стимула', ''))
    a_db_pos = get_primary_pos(row.get('часть речи ассоциации', ''))

    s_sem = str(row.get('сем. группа стимула_НКРЯ', '')).lower()

    ps = morph.parse(s_word)[0]
    pa = morph.parse(a_word.split()[0])[0]

    # Определение части речи
    s_is_verb = s_db_pos == 'V' or (not s_db_pos and ps.tag.POS in {'VERB', 'INFN'})
    a_is_verb = a_db_pos == 'V' or (not a_db_pos and pa.tag.POS in {'VERB', 'INFN'})
    s_is_noun = s_db_pos == 'S' or (not s_db_pos and ps.tag.POS in {'NOUN', 'NPRO'})
    a_is_noun = a_db_pos == 'S' or (not a_db_pos and pa.tag.POS in {'NOUN', 'NPRO'})
    s_is_adj = s_db_pos == 'A' or (not s_db_pos and ps.tag.POS in {'ADJF', 'PRTF', 'ADJS'})
    a_is_adj = a_db_pos == 'A' or (not a_db_pos and pa.tag.POS in {'ADJF', 'PRTF', 'ADJS'})

    has_prep = 'PR+' in a_db_pos or (' ' in a_word and len(a_word.split()[0]) <= 3)
    s_is_verbal_nature = s_is_verb or 'отглагольное' in s_sem or 'nomina agentis' in s_sem

    # Служебные
    if any(x in {s_word, a_word} for x in {'быть', 'буду', 'стал', 'бы', 'пусть', 'было'}):
        return 'служебное'

    # Атрибутивные
    # а) Обстоятельственное (Глагол + Наречие/Предлог/Творительный)
    if s_is_verb:
        if has_prep or 'ADV' in a_db_pos or pa.tag.POS in {'ADVB', 'GRND'}:
            return 'атрибутивное: обстоятельственное'
        if pa.tag.POS == 'INFN' and 'движение' in s_sem:
            return 'атрибутивное: обстоятельственное'
        if a_is_noun and pa.tag.case == 'ablt':
            return 'атрибутивное: обстоятельственное'

    # б) Определительное (Сущ + Прил)
    if (s_is_noun and a_is_adj) or (s_is_adj and a_is_noun):
        return 'атрибутивное: определительное'

    # в) Количественное
    if 'NUM' in a_db_pos or pa.tag.POS in {'NUMR', 'Anum'}:
        return 'атрибутивное: аппроксимативно-порядковое/количественное'

    # г) Собственно атрибутивное (Сущ + Сущ в косвенном падеже или с предлогом)
    if s_is_noun and a_is_noun:
        if has_prep or pa.tag.case == 'gent':
            return 'атрибутивное'

    # Актнатные
    if (s_is_noun and a_is_verb) or (s_is_verb and a_is_noun) or (s_is_verbal_nature and a_is_noun):
        return 'актантное'

    # Глаголы-связки + Именная часть
    if (s_word in COPULAS and (a_is_noun or a_is_adj)) or (a_word in COPULAS and (s_is_noun or s_is_adj)):
        return 'актантное'


    return 'не выявлено'



df_result['Синтаксическая разметка'] = df_result.apply(get_syntax_v5_fixed, axis=1)

"""### Коллокации"""

wn = RuWordNet()

# Формирование базы коллокаций
rwn_collocations = {
    sense.name.lower().split('(')[0].strip()
    for sense in wn.senses
    if ' ' in sense.name
}


# Функция проверки
def check_sustainability(row):
    # Берем стимул и лемму ассоциации
    s = str(row.get('стимул', '')).strip().lower()
    a = str(row.get('лемма', row.get('ассоциация', ''))).strip().lower()

    if not s or not a:
        return "нет"

    # учет инверсии
    if f"{s} {a}" in rwn_collocations or f"{a} {s}" in rwn_collocations:
        return "коллокация"

    # дополнительная проверка: если ассоциация сама по себе является устойчивым сочетанием
    # (актуально для многословных ассоциаций типа "набор знаний")
    if a in rwn_collocations:
        return "коллокация (вн. структура)"

    return "нет"


df_result['устойчивость'] = df_result.apply(check_sustainability, axis=1)

"""## Метрики"""

# 1.1.Агрегация частот на уровне лемм для каждого стимула
# сколько раз лемма встретилась как реакция на конкретный стимул
lemma_freqs = df_result.groupby(['стимул', 'лемма'])['частота'].sum().reset_index(name='частота_леммы')


def make_conceptual_key(s, l):
    pair = sorted([str(s).lower().strip(), str(l).lower().strip()])
    return " <-> ".join(pair)

lemma_freqs['temp_key'] = [make_conceptual_key(s, l) for s, l in zip(lemma_freqs['стимул'], lemma_freqs['лемма'])]

# 1.2.Концептуальная частота (с учетом инверсии)
conceptual_freqs = lemma_freqs.groupby('temp_key')['частота_леммы'].sum().reset_index(name='Концептуальная частота')

df_result['temp_key'] = [make_conceptual_key(s, l) for s, l in zip(df_result['стимул'], df_result['лемма'])]

# Присоединяем концептуальные частоты
df_result = df_result.merge(conceptual_freqs, on='temp_key', how='left')
df_result = df_result.drop(columns=['temp_key'])

# Расчёт статистических метрик связи и семантической близости по Wu-Palmer
N_total = df_result['частота'].sum()

# Глобальная частота
stimulus_counts = df_result.groupby(df_result['стимул'].astype(str).str.lower().str.strip())['частота'].sum().to_dict()
global_lemma_counts = df_result.groupby(df_result['лемма'].astype(str).str.lower().str.strip())['частота'].sum().to_dict()

wn = RuWordNet()

@lru_cache(maxsize=50000)
def get_synset_info(synset):
    queue = [(synset, 0)]
    visited = {synset}
    max_d = 0
    ancestors = {synset}
    while queue:
        current, d = queue.pop(0)
        max_d = max(max_d, d)
        for hyper in current.hypernyms:
            if hyper not in visited:
                visited.add(hyper)
                ancestors.add(hyper)
                queue.append((hyper, d + 1))
    return max_d, ancestors

@lru_cache(maxsize=50000)
def wup_sim_fixed(ss1, ss2):
    if ss1 == ss2: return 1.0

    d1, anc1 = get_synset_info(ss1)
    d2, anc2 = get_synset_info(ss2)
    common = anc1.intersection(anc2)

    if not common: return 0.0

    lcs_depth = max(get_synset_info(c)[0] for c in common)


    eff_d1 = max(d1, lcs_depth)
    eff_d2 = max(d2, lcs_depth)

    sim = (2.0 * (lcs_depth + 1)) / ((eff_d1 + 1) + (eff_d2 + 1))

    return min(1.0, float(sim))


def calculate_all_metrics(row):
    f_sa = row['Концептуальная частота']
    s_norm = str(row['стимул']).lower().strip()
    a_norm = str(row['лемма']).lower().strip()

    f_s = stimulus_counts.get(s_norm, f_sa)
    f_a = global_lemma_counts.get(a_norm, f_sa)

    # Логическая защита: частота отдельного слова не может быть меньше частоты их совместной пары
    f_s = max(f_s, f_sa)
    f_a = max(f_a, f_sa)

    # Dice Score
    dice = (2 * f_sa) / (f_s + f_a) if (f_s + f_a) > 0 else 0

    # Delta P
    p_a_s = f_sa / f_s if f_s > 0 else 0
    den = (N_total - f_s) if (N_total - f_s) > 0 else 1
    p_a_not_s = (f_a - f_sa) / den
    delta_p = p_a_s - p_a_not_s

    # PMI
    pmi = np.log2((f_sa * N_total) / (f_s * f_a)) if (f_sa > 0 and f_s > 0 and f_a > 0) else 0.0

    # Wu-Palmer
    wup = 0.0
    s_senses = wn.get_senses(s_norm)
    a_senses = wn.get_senses(a_norm)
    if s_senses and a_senses:
        wup = max([wup_sim_fixed(s.synset, a.synset) for s in s_senses for a in a_senses], default=0.0)

    return pd.Series([dice, delta_p, pmi, wup])

cols = ['Dice Score', 'Delta P', 'PMI', 'Wu Palmer']
tqdm.pandas(desc="Вычисление метрик")
df_result[cols] = df_result.progress_apply(calculate_all_metrics, axis=1)

"""### ИТОГ. КЛАССИФИКАЦИЯ ПАР"""

def get_primary_pos(pos_str):
    if pd.isna(pos_str) or pos_str == '':
        return ''
    return str(pos_str).split('/')[0].strip().upper()

def classify_association_type(row):

    # Определение составной ассоциации
    a_pos_raw = str(row.get('часть речи ассоциации', '')).strip()
    is_multiword = '+' in a_pos_raw

    if is_multiword:
        # Проверка: начинается ли с предлога
        # В колонке части речи ассоциации может быть, например, "PR+S"
        parts = a_pos_raw.replace('+', ' ').split()
        if len(parts) >= 2 and parts[0].upper() == 'PR':
            second_pos = parts[1].upper()
            s_pos = get_primary_pos(row.get('часть речи стимула', ''))
            if s_pos == second_pos and s_pos != '':
                return ('синтаксический / составная', False)
            else:
                return ('синтагматический / составная', False)
        # Иначе - тематическая или просто составная
        fb_exact = str(row.get('FB_точные примеры', '')).strip()
        if fb_exact != '':
            return ('тематический / составная', False)
        else:
            return ('составная', False)

    # Проверки для однословных
    s_pos = get_primary_pos(row.get('часть речи стимула', ''))
    a_pos = get_primary_pos(a_pos_raw)

    # Исключения
    epi_val = str(row.get('эпидигматика и формальная связь', '')).strip().lower()

    if 'идентичные' in epi_val:
        return ('идентичная', False)

    formal_keywords = ['фонетическая', 'графическая']
    if any(kw in epi_val for kw in formal_keywords):
        return ('формальная', False)

    if 'грамматическая (формы слова)' in epi_val:
        return ('грамматическая (формы слова)', False)

    # Список тегов
    tags = []
    dispute_flag = False

    # 1.Эпидигматическая связь
    if 'эпидигматическая' in epi_val:
        tags.append('эпидигматический')

    # 2. Парадигматическая связь (RuWordNet)
    wn_val = str(row.get('RuWordNet', '')).strip().lower()
    wn_tags = [t.strip() for t in wn_val.split(',') if t.strip()]

    paradigmatic_wn = {'синоним', 'антоним', 'гипоним', 'гипероним', 'экземпляр', 'класс'}
    if any(t in paradigmatic_wn for t in wn_tags) and s_pos == a_pos and s_pos != '':
        tags.append('парадигматический')

    # 3. Тематическая связь (RuWordNet или FB_точные примеры) ----------
    thematic_wn = {'мероним', 'холоним', 'домен', 'объект домена', 'вывод', 'предпосылка', 'следствие', 'причина'}
    has_thematic_wn = any(t in thematic_wn for t in wn_tags)

    fb_exact = str(row.get('ФБ точные примеры', '')).strip()
    has_fb_exact = (fb_exact != '')

    if has_thematic_wn or has_fb_exact:
        tags.append('тематический')

    # общая ЛСГ
    common_sem_raw = row.get('общая ЛСГ')
    if pd.isna(common_sem_raw):
        common_sem = ''
    else:
        common_sem = str(common_sem_raw).strip()
    if common_sem.lower() in ('', 'nan', 'none'):
        common_sem = ''

    if len(tags) == 0 and common_sem != '':
        tags.append('тематический')

   # 4. Синтаксическая связь
    allowed_syntax = {
        'атрибутивное: определительное',
        'актантное',
        'атрибутивное: обстоятельственное',
        'атрибутивное',
        'атрибутивное: аппроксимативно-порядковое/количественное',
        'служебное'
    }
    syntax_val = str(row.get('Синтаксическая разметка', '')).strip()
    if syntax_val in allowed_syntax:
        if s_pos == a_pos and s_pos != '':
            # Исключение: глагол + инфинитив → синтагматическая связь
            s_gram = str(row.get('граммемы стимула', '')).upper()
            if s_pos == 'V' and a_pos == 'V' and 'INF' in s_gram:
                tags.append('синтагматический')
            else:
                tags.append('синтаксический')
        else:
            tags.append('синтагматический')

    # 5. Коллокативная связь
    stability = str(row.get('устойчивость', '')).strip().lower()
    if 'коллокация' in stability:
        tags.append('коллокативный')


    priority_order = {
        'синтаксический': 0,
        'синтагматический': 1,
        'эпидигматический': 2,
        'парадигматический': 3,
        'коллокативный': 4,
        'тематический': 5,
        'составная': 6
    }
    tags.sort(key=lambda t: priority_order.get(t, 99))

    if len(tags) == 0:
        final_type = 'не определена'
    else:
        final_type = ' / '.join(tags)




tqdm.pandas(desc="Классификация пар")
df_result[['тип связи', 'спорный_случай']] = df_result.progress_apply(
    lambda row: pd.Series(classify_association_type(row)), axis=1
)

df_result.to_csv('0_final_dataset.csv', index=False)

df_result['тип связи'].str.split(' / ').explode().str.strip().value_counts()