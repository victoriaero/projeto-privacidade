from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/blog_authorship/blogtext.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH, usecols=["id", "gender", "age", "text"])

    posts_per_author = (
        df.groupby("id")
        .size()
        .rename("n_posts")
        .reset_index()
    )

    print("\n=== Posts per author distribution ===")
    print(posts_per_author["n_posts"].describe())

    print("\n=== Percentiles ===")
    percentiles = posts_per_author["n_posts"].quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    print(percentiles)

    print("\n=== Candidate minimum thresholds ===")
    for min_posts in [1, 2, 3, 5, 10, 20]:
        kept = posts_per_author[posts_per_author["n_posts"] >= min_posts]
        print(
            f"min_posts >= {min_posts:>2}: "
            f"authors kept = {len(kept):>6} "
            f"({len(kept) / len(posts_per_author):.1%})"
        )

    print("\n=== Candidate maximum caps ===")
    for max_posts in [5, 10, 20, 50, 100]:
        capped_total_posts = posts_per_author["n_posts"].clip(upper=max_posts).sum()
        original_total_posts = posts_per_author["n_posts"].sum()

        print(
            f"max_posts <= {max_posts:>3}: "
            f"posts processed = {capped_total_posts:>8} "
            f"({capped_total_posts / original_total_posts:.1%} of original posts)"
        )

    print("\n=== Combined candidates ===")
    for min_posts in [1, 3, 5, 10]:
        for max_posts in [10, 20, 50]:
            kept = posts_per_author[posts_per_author["n_posts"] >= min_posts]
            capped_total_posts = kept["n_posts"].clip(upper=max_posts).sum()

            print(
                f"min={min_posts:>2}, max={max_posts:>2}: "
                f"authors={len(kept):>6}, "
                f"posts_to_embed={capped_total_posts:>8}, "
                f"avg_posts_after_cap={capped_total_posts / len(kept):.2f}"
            )


if __name__ == "__main__":
    main()