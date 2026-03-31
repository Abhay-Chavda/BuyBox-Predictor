import numpy as np
import pandas as pd


def finding_range(df, buybox_id, last_range, verbose=False):
    n = (df["BuyboxHistoryId"] == buybox_id).sum()
    if verbose:
        print(last_range, n)

    start_range = last_range + 1
    end_range = start_range + n - 1
    return start_range, int(end_range)


def getting_minimum_value(df, start_range, end_range):
    minimum_value = df.iloc[start_range : end_range + 1]["SellPrice"].min()
    return minimum_value


def getting_price_gap(df, minimum_value, price):
    price_gap = price - minimum_value
    return price_gap


def getting_price_rank(df, price_value, start_range, end_range, verbose=False):
    array = df.iloc[start_range : end_range + 1]["SellPrice"].to_numpy()
    array = np.sort(array)
    indices = np.where(array == price_value)[0]

    if verbose:
        print("array sorted:", array, "indices:", indices)

    if len(indices) == 0:
        return -1

    return int(indices[0] + 1)


def getting_total_competitors(df, start_range, end_range):
    return end_range - start_range + 1


def getting_positive_feedback(df, seller_id, buybox_id):
    value = df[
        (df["BuyboxHistoryId"] == buybox_id) & (df["SellerId"] == seller_id)
    ]["PositiveFeedbackPercent"]
    return value


def getting_max_feedback(df, start_range, end_range):
    array = df.iloc[start_range : end_range + 1]["PositiveFeedbackPercent"].to_numpy()
    return array.max()


def getting_positive_fulfilment_type(df, seller_id, buybox_id):
    value = df[
        (df["BuyboxHistoryId"] == buybox_id) & (df["SellerId"] == seller_id)
    ]["FulfillmentChannel"]
    return value


def check_winner(df, seller_id, buybox_id):
    value = df[
        (df["BuyboxHistoryId"] == buybox_id) & (df["SellerId"] == seller_id)
    ]["IsBuyBoxWinner"]
    return value


def refining_data(df: pd.DataFrame) -> pd.DataFrame:
    featured_columns = [
        "SellPrice",
        "ShippingPrice",
        "TotalPrice",
        "MinCompetitorPrice",
        "MinTotalPriceInSnapshot",
        "PriceGap",
        "TotalPriceGap",
        "PriceGapPercent",
        "PriceRank",
        "PriceRankNormalized",
        "TotalCompetitorsInSnapshot",
        "PositiveFeedbackPercent",
        "MaxFeedbackInSnapshot",
        "FeedbackGapFromMax",
        "IsMinSellPrice",
        "IsMinTotalPrice",
        "IsFBA",
    ]

    df_h = df.sort_values("BuyboxHistoryId").reset_index(drop=True)

    def _scalar(series):
        return series.iloc[0] if len(series) else None

    rows = []
    last_range = -1

    for bid in df_h["BuyboxHistoryId"].unique():
        start_range, end_range = finding_range(df_h, bid, last_range)
        last_range = end_range

        min_price = getting_minimum_value(df_h, start_range, end_range)
        max_feedback = getting_max_feedback(df_h, start_range, end_range)
        n_competitors = getting_total_competitors(df_h, start_range, end_range)
        block = df_h.iloc[start_range : end_range + 1]
        min_total_price = (block["SellPrice"] + block["ShippingPrice"]).min()

        for i in range(start_range, end_range + 1):
            r = df_h.iloc[i]
            seller_id = r["SellerId"]
            price = float(r["SellPrice"])
            ship = float(r["ShippingPrice"])
            total_price = price + ship

            pos_fb = _scalar(getting_positive_feedback(df_h, seller_id, bid))
            channel = _scalar(getting_positive_fulfilment_type(df_h, seller_id, bid))

            fb_gap = (
                round(float(max_feedback) - float(pos_fb), 2)
                if pos_fb is not None
                else None
            )

            is_fba = (
                1
                if channel is not None and str(channel).strip().upper() == "FBA"
                else 0
            )

            price_rank = getting_price_rank(df_h, price, start_range, end_range)

            rows.append(
                {
                    "BuyboxHistoryId": bid,
                    "SellerId": seller_id,
                    "SellPrice": price,
                    "ShippingPrice": ship,
                    "TotalPrice": round(total_price, 2),
                    "MinCompetitorPrice": min_price,
                    "MinTotalPriceInSnapshot": round(float(min_total_price), 2),
                    "PriceGap": round(getting_price_gap(df_h, min_price, price), 2),
                    "TotalPriceGap": round(total_price - float(min_total_price), 2),
                    "IsMinSellPrice": int(price == min_price),
                    "IsMinTotalPrice": int(
                        abs(total_price - float(min_total_price)) < 1e-6
                    ),
                    "PriceRank": price_rank,
                    "TotalCompetitorsInSnapshot": n_competitors,
                    "PriceGapPercent": round(price_rank / min_price, 2)
                    if min_price != 0
                    else 0.0,
                    "PriceRankNormalized": price_rank / n_competitors
                    if n_competitors != 0
                    else 0.0,
                    "PositiveFeedbackPercent": pos_fb,
                    "MaxFeedbackInSnapshot": max_feedback,
                    "FeedbackGapFromMax": fb_gap,
                    "FulfillmentChannel": channel,
                    "IsFBA": is_fba,
                    "IsBuyBoxWinner": _scalar(check_winner(df_h, seller_id, bid)),
                    "CreatedAt": r["CreatedAt"],
                }
            )

    df_features = pd.DataFrame(rows)
    df_features[featured_columns] = df_features[featured_columns].astype(float)
    return df_features
