# Гибридная система рекомендаций книг

Итоговый проект - гибридная рекомендательная система с классическими и нейросетевыми подходами.

## Запуск

**Единый notebook** (без модулей) - `ML_book_recommender_classes.ipynb`

### Google Colab 

**1**: Выполните в первой ячейке:
```bash
pip -q uninstall -y scikit-surprise surprise numpy
pip -q install numpy==1.26.4
pip -q install --no-cache-dir scikit-surprise implicit
```

**2**:  **ОБЯЗАТЕЛЬНО Restart Runtime** (Runtime → Restart runtime)

**3**: Запустите остальные ячейки последовательно

### Локально (Jupyter)
```bash
pip install numpy==1.26.4 pandas scikit-learn scikit-surprise implicit torch
jupyter notebook ML_book_recommender_classes.ipynb
```

## Данные и разделение

- **Датасет**: GoodBooks-10k (6M оценок, 53K пользователей, 10K книг)
- **Train/Test**: 80%/20% (стратифицированный split по пользователям)
- **Валидация**: Train → 90% train_fit / 10% val (для подбора гиперпараметров гибрида)
- **Проверка на data leakage**: пары (user_id, book_id) 

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


## Метрики

- **Precision@10** - доля релевантных книг в топ-10
- **Recall@10** - доля найденных релевантных книг от всех релевантных
- **nDCG@10** - ранжирование с учетом позиции
- **HitRate@10** - доля пользователей с ≥1 релевантной книгой в топ-10
- **Coverage** - доля уникальных книг в рекомендациях (diversity)

**Порог релевантности**: rating ≥ 4


