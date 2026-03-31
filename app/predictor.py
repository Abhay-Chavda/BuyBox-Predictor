import numpy as np


def rank_among_prices(competitors_price, price):
    total = list(competitors_price) + [price]
    total = sorted(total)
    return total.index(price) + 1


def predict_winner_probability(row, trained_model):
    probability = trained_model.predict_proba(row)[0][1]
    return probability


def find_best_price(
    df,
    trained_model,
    feature_columns,
    seller_id,
    buybox_id,
    min_price,
    max_price,
):
    results = []

    sample_row = df[
        (df["SellerId"] == seller_id) & (df["BuyboxHistoryId"] == buybox_id)
    ]

    if sample_row.empty:
        raise ValueError("No matching row for given SellerId and BuyboxHistoryId.")

    sample_row = sample_row.iloc[[0]].copy()
    competitors_price = df[df["BuyboxHistoryId"] == buybox_id]["SellPrice"].tolist()

    step_size = (max_price - min_price) / 100 if max_price != min_price else 0.01

    shipping_price = float(sample_row["ShippingPrice"].iloc[0])
    total_competitors = int(sample_row["TotalCompetitorsInSnapshot"].iloc[0])
    max_feedback_in_snapshot = float(sample_row["MaxFeedbackInSnapshot"].iloc[0])
    feedback_gap_from_max = float(sample_row["FeedbackGapFromMax"].iloc[0])
    is_fba = int(sample_row["IsFBA"].iloc[0])
    positive_feedback = float(sample_row["PositiveFeedbackPercent"].iloc[0])

    min_price = float(min_price)
    max_price = float(max_price)

    for price in np.arange(min_price, max_price + step_size, step_size):
        row = sample_row.copy()

        row["SellPrice"] = float(price)
        row["TotalPrice"] = round(float(price) + shipping_price, 2)

        min_competitor_price = min(competitors_price)
        row["MinCompetitorPrice"] = min_competitor_price

        min_total_price = round(float(min_competitor_price) + shipping_price, 2)
        row["MinTotalPriceInSnapshot"] = min_total_price

        row["PriceGap"] = round(float(price) - float(min_competitor_price), 2)
        row["TotalPriceGap"] = round(
            float(row["TotalPrice"].iloc[0]) - min_total_price, 2
        )
        row["IsMinSellPrice"] = int(float(price) == float(min_competitor_price))
        row["IsMinTotalPrice"] = int(
            abs(float(row["TotalPrice"].iloc[0]) - min_total_price) < 1e-6
        )

        price_rank = rank_among_prices(competitors_price, float(price))
        row["PriceRank"] = price_rank
        row["PriceGapPercent"] = (
            round(price_rank / float(min_competitor_price), 2)
            if float(min_competitor_price) != 0
            else 0.0
        )
        row["PriceRankNormalized"] = (
            price_rank / total_competitors if total_competitors != 0 else 0.0
        )

        row["TotalCompetitorsInSnapshot"] = total_competitors
        row["PositiveFeedbackPercent"] = positive_feedback
        row["MaxFeedbackInSnapshot"] = max_feedback_in_snapshot
        row["FeedbackGapFromMax"] = feedback_gap_from_max
        row["IsFBA"] = is_fba

        model_input = row.reindex(columns=feature_columns, fill_value=0.0)
        winner_probability = predict_winner_probability(model_input, trained_model)

        normalised_checking = float(winner_probability) * float(price)

        results.append(
            {
                "sell_price": float(row["SellPrice"].iloc[0]),
                "score": normalised_checking,
                "winning_probability": float(winner_probability),
            }
        )

    best_price = max(results, key=lambda x: x["score"])

    return {
        "best_price": best_price["sell_price"],
        "winning_probability": best_price["winning_probability"],
    }
