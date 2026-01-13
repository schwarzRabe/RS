# Гибридная система рекомендаций книг

Итоговый проект - гибридная рекомендательная система с классическими и нейросетевыми подходами.

---

## Запуск

**Единый notebook** (без модулей) - `ML_book_recommender_classes.ipynb`

### Google Colab (рекомендуется)

**Шаг 1**: Выполните в первой ячейке:
```bash
pip -q uninstall -y scikit-surprise surprise numpy
pip -q install numpy==1.26.4
pip -q install --no-cache-dir scikit-surprise implicit
```

**Шаг 2**:  **ОБЯЗАТЕЛЬНО Restart Runtime** (Runtime → Restart runtime)

**Шаг 3**: Запустите остальные ячейки последовательно

### Локально (Jupyter)
```bash
pip install numpy==1.26.4 pandas scikit-learn scikit-surprise implicit torch
jupyter notebook ML_book_recommender_classes.ipynb
```

**Время выполнения**: ~15-20 минут

---

## Данные и разделение

- **Датасет**: GoodBooks-10k (6M оценок, 53K пользователей, 10K книг)
- **Train/Test**: 80%/20% (стратифицированный split по пользователям)
- **Валидация**: Train → 90% train_fit / 10% val (для подбора гиперпараметров гибрида)
- **Проверка на data leakage**: пары (user_id, book_id) не пересекаются ✅

---

## Модели

1. **PopularityRecommender** - baseline (топ книг по avg_rating + count)
2. **ContentTFIDFRecommender** - TF-IDF по (title + tags)
3. **ItemCFRecommender** - Item-Based CF (косинусная схожесть)
4. **SVDRecommender** - Matrix Factorization (Surprise, explicit ratings)
5. **ALSImplicitRecommender** - Matrix Factorization (implicit library, confidence)
6. **HybridRecommender** - Two-stage (candidate generation + reranking):
   - Генерация кандидатов от всех моделей (по 100 от каждой)
   - Reranking: взвешенное усреднение скоров с сегментацией пользователей
7. **NeuralReranker** - MLP [3→32→32→1] для reranking по признакам

---

## Признаки

**User features** (по train):
- `user_mean` - средний рейтинг пользователя
- `user_cnt` - количество оценок

**Item features** (по train):
- `item_mean` - средний рейтинг книги
- `item_cnt` - популярность (количество оценок)

**Scoring features** (для reranking):
- `als_s` - скор от ALS (normalized 0-1)
- `icf_s` - скор от Item-CF (normalized 0-1)
- `item_cnt` - популярность (normalized 0-1)

---

## Метрики

- **Precision@10** - доля релевантных книг в топ-10
- **Recall@10** - доля найденных релевантных книг от всех релевантных
- **nDCG@10** - ранжирование с учетом позиции
- **HitRate@10** - доля пользователей с ≥1 релевантной книгой в топ-10
- **Coverage** - доля уникальных книг в рекомендациях (diversity)

**Порог релевантности**: rating ≥ 4

---

## Результаты (Test Set)

| Модель | Precision@10 | Recall@10 | nDCG@10 | HitRate@10 | Coverage |
|--------|--------------|-----------|---------|------------|----------|
| Popularity | 0.184 | 0.117 | 0.219 | 0.763 | 0.003 |
| Content | 0.165 | 0.108 | 0.199 | 0.720 | 0.495 |
| Item-CF | 0.194 | 0.127 | 0.233 | 0.791 | 0.623 |
| SVD | 0.209 | 0.137 | 0.250 | 0.823 | 0.574 |
| **ALS** | **0.220** | **0.145** | **0.263** | **0.840** | 0.405 |
| **Hybrid** | **0.221** | **0.145** | **0.264** | **0.842** | **0.463** |
| Hybrid+Neural | 0.221 | 0.146 | 0.265 | 0.842 | 0.463 |

---

## Выводы

**ALS** - лучшая базовая модель по качеству (Precision/Recall/nDCG), но низкий Coverage (0.405).

**Гибрид** улучшает Coverage (+14% vs ALS) через комбинирование моделей, сохраняя качество. Сегментация пользователей (новые → популярные книги, активные → персонализация) решает холодный старт.

**Нейросеть** дала минимальный прирост (+0.1% Precision) - линейная модель уже близка к оптимуму на данных признаках. Для роста нужны дополнительные features (user embeddings, sequential patterns).
