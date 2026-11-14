# Feature Catalog

All features are defined once in `src/features/build_features.py` and served through the `FeatureStore` class, so the same definitions are used for both training and prediction.

## User features

| Feature | Type | Description |
|---------|------|-------------|
| `user_avg_rating` | float | Mean rating the user has given (train period). |
| `user_activity_count` | int | Number of interactions by the user. |
| `user_rating_std` | float | Std-dev of the user's ratings (rating dispersion). |
| `user_distinct_items` | int | Count of distinct items the user interacted with. |
| `user_active_days` | int | Days between the user's first and last interaction. |
| `user_days_since_last` | int | Days from the user's last interaction to the reference time. |
| `user_preferred_category` | category | Genre/category the user interacts with most often. |
| `age` | int | User age (demographic). |
| `gender` | category | User gender (demographic). |
| `occupation` | category | User occupation (demographic). |

## Item features

| Feature | Type | Description |
|---------|------|-------------|
| `item_popularity` | int | Number of interactions the item received. |
| `item_distinct_users` | int | Count of distinct users who interacted with the item. |
| `item_avg_rating` | float | Mean rating received by the item. |
| `item_rating_std` | float | Std-dev of ratings received by the item. |
| `item_popularity_rank` | int | Rank of the item by interaction count (1 = most popular). |
| `item_popularity_score` | float | Damped popularity score blending count and mean rating. |
| `primary_category` | category | Primary genre/category of the item. |
| `item_genre_vector` | int[] | Multi-hot genre flags (used for content/diversity metrics). |

## Interaction features

| Feature | Type | Description |
|---------|------|-------------|
| `days_since_last_user_interaction` | int | Days since the user's previous interaction at event time. |
| `user_item_category_match` | int | 1 if the item's category equals the user's preferred category. |
| `rating` | float | Target variable: the observed rating. |

