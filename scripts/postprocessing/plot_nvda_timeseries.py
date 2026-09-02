"""One-off NVDA sentiment time-series validation plot.

Reads outputs/inference/NVDA.US.t1_sentiment_enriched.jsonl, computes monthly
mention-weighted NVDA self-sentiment, and produces a 2-panel matplotlib figure
plus a CSV of the underlying monthly aggregates.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/inference/NVDA.US.t1_sentiment_enriched.jsonl"
OUT_PNG = REPO / "outputs/inference/nvda_sentiment_timeseries.png"
OUT_CSV = REPO / "outputs/inference/nvda_monthly_sentiment.csv"

MIN_ARTICLES_PER_MONTH = 10

EVENTS = [
    ("2022-11-30", "ChatGPT launch"),
    ("2023-05-24", "NVDA blowout earnings"),
    ("2023-11-13", "Hopper cycle"),
    ("2024-03-18", "Blackwell announcement"),
    ("2025-04-02", "Tariff concerns"),
]


def main() -> None:
    # month_key -> list of (sentiment, mentions) for NVDA across all articles
    buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    n_total = 0
    n_with_nvda = 0

    with SRC.open() as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            date = rec.get("date") or rec.get("trade_date")
            if not date:
                continue
            month = date[:7]  # YYYY-MM
            for ts in rec.get("ticker_sentiments", []):
                if ts.get("ticker") != "NVDA":
                    continue
                sent = ts.get("sentiment")
                mentions = ts.get("total_mentions", 0) or 0
                if sent is None or mentions <= 0:
                    continue
                buckets[month].append((float(sent), int(mentions)))
                n_with_nvda += 1
                break  # one NVDA entry per article expected

    print(f"Loaded {n_total} articles, {n_with_nvda} with NVDA self-sentiment")

    rows = []
    for month, vals in sorted(buckets.items()):
        n_articles = len(vals)
        total_mentions = sum(m for _, m in vals)
        if total_mentions == 0:
            continue
        weighted_mean = sum(s * m for s, m in vals) / total_mentions
        # weighted std (population)
        var = sum(m * (s - weighted_mean) ** 2 for s, m in vals) / total_mentions
        weighted_std = math.sqrt(var)
        rows.append(
            {
                "month": month,
                "n_articles": n_articles,
                "total_mentions": total_mentions,
                "weighted_mean_sentiment": round(weighted_mean, 4),
                "weighted_std": round(weighted_std, 4),
            }
        )

    df = pd.DataFrame(rows)
    df["month_dt"] = pd.to_datetime(df["month"] + "-01")
    df_full = df.copy()
    df_plot = df[df["n_articles"] >= MIN_ARTICLES_PER_MONTH].copy()

    df_full.drop(columns=["month_dt"]).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df_full)} months total, {len(df_plot)} after >= {MIN_ARTICLES_PER_MONTH} article filter)")

    # ---- plot ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot(df_plot["month_dt"], df_plot["weighted_mean_sentiment"], marker="o", lw=1.6, color="#1f77b4")
    ax_top.fill_between(
        df_plot["month_dt"],
        df_plot["weighted_mean_sentiment"] - df_plot["weighted_std"],
        df_plot["weighted_mean_sentiment"] + df_plot["weighted_std"],
        alpha=0.12,
        color="#1f77b4",
    )
    ax_top.axhline(0, color="grey", lw=0.7, ls="--")
    ax_top.set_ylabel("Mention-weighted mean NVDA sentiment")
    ax_top.set_title("NVDA self-sentiment over time (monthly, mention-weighted)")
    ax_top.grid(True, alpha=0.3)

    ymin, ymax = ax_top.get_ylim()
    for date_str, label in EVENTS:
        edate = pd.Timestamp(date_str)
        ax_top.axvline(edate, color="crimson", lw=0.8, alpha=0.6)
        ax_top.annotate(
            label,
            xy=(edate, ymax),
            xytext=(2, -10),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="crimson",
        )

    ax_bot.bar(df_full["month_dt"], df_full["n_articles"], width=20, color="#7f7f7f", alpha=0.7)
    ax_bot.axhline(MIN_ARTICLES_PER_MONTH, color="orange", lw=0.7, ls=":")
    ax_bot.set_ylabel("Articles / month")
    ax_bot.set_xlabel("Month")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bot.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")

    # ---- sanity checks ----
    df_plot = df_plot.reset_index(drop=True)
    print("\n=== Sanity-check table ===")

    def month_val(m):
        row = df_plot[df_plot["month"] == m]
        return None if row.empty else float(row["weighted_mean_sentiment"].iloc[0])

    pre_chatgpt = df_plot[df_plot["month"].between("2022-08", "2022-10")]["weighted_mean_sentiment"].mean()
    nov_dec_2022 = df_plot[df_plot["month"].between("2022-11", "2022-12")]["weighted_mean_sentiment"].mean()
    print(f"Aug-Oct 2022 avg sentiment : {pre_chatgpt:.4f}")
    print(f"Nov-Dec 2022 avg sentiment : {nov_dec_2022:.4f}")
    print(f"  -> uptick around ChatGPT? {'YES' if nov_dec_2022 > pre_chatgpt else 'NO'}")

    may_2023 = month_val("2023-05")
    apr_2023 = month_val("2023-04")
    jun_2023 = month_val("2023-06")
    print(f"Apr 2023: {apr_2023}, May 2023: {may_2023}, Jun 2023: {jun_2023}")
    is_local_max = may_2023 is not None and (
        (apr_2023 is None or may_2023 >= apr_2023) and (jun_2023 is None or may_2023 >= jun_2023)
    )
    print(f"  -> May 2023 local max? {'YES' if is_local_max else 'NO'}")

    avg_2020_2022 = df_plot[df_plot["month"].between("2020-01", "2022-10")]["weighted_mean_sentiment"].mean()
    avg_2023_2024 = df_plot[df_plot["month"].between("2023-01", "2024-12")]["weighted_mean_sentiment"].mean()
    print(f"2020-2022 (pre-ChatGPT) avg : {avg_2020_2022:.4f}")
    print(f"2023-2024 (AI boom) avg     : {avg_2023_2024:.4f}")
    print(f"  -> AI-era higher?         {'YES' if avg_2023_2024 > avg_2020_2022 else 'NO'}")

    # extreme outlier detection: months >2 std from the global mean
    mean_g = df_plot["weighted_mean_sentiment"].mean()
    std_g = df_plot["weighted_mean_sentiment"].std()
    df_plot["z"] = (df_plot["weighted_mean_sentiment"] - mean_g) / std_g
    anomalies = df_plot[df_plot["z"].abs() > 2.0].sort_values("z")
    print("\nAnomalous months (|z|>2):")
    if anomalies.empty:
        print("  none")
    else:
        for _, r in anomalies.iterrows():
            print(f"  {r['month']}: sent={r['weighted_mean_sentiment']:.4f} z={r['z']:.2f} n={r['n_articles']}")

    print("\nTop 5 highest-sentiment months:")
    top5 = df_plot.nlargest(5, "weighted_mean_sentiment")[
        ["month", "n_articles", "total_mentions", "weighted_mean_sentiment"]
    ]
    print(top5.to_string(index=False))

    print("\nBottom 5 lowest-sentiment months:")
    bot5 = df_plot.nsmallest(5, "weighted_mean_sentiment")[
        ["month", "n_articles", "total_mentions", "weighted_mean_sentiment"]
    ]
    print(bot5.to_string(index=False))


if __name__ == "__main__":
    main()
