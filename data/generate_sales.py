"""Generate the fictional sales dataset for the AI Data Analyst project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 2

OUTPUT_PATH = Path(__file__).resolve().parent / "sales.csv"


PRODUCT_INFO: dict[str, tuple[str, float, float]] = {
    "Laptop": ("Electronics", 900.0, 0.76),
    "Phone": ("Electronics", 700.0, 0.74),
    "Monitor": ("Electronics", 300.0, 0.72),
    "Headphones": ("Electronics", 120.0, 0.65),

    "Chair": ("Furniture", 250.0, 0.62),
    "Desk": ("Furniture", 600.0, 0.68),
    "Cabinet": ("Furniture", 500.0, 0.66),
    "Bookshelf": ("Furniture", 400.0, 0.64),

    "Printer": ("Office Supplies", 300.0, 0.62),
    "Paper Box": ("Office Supplies", 75.0, 0.48),
    "Toner": ("Office Supplies", 150.0, 0.56),
    "Stationery Kit": ("Office Supplies", 100.0, 0.50),
}


NORMAL_MONTH_PRODUCT_COUNTS: dict[str, int] = {
    "Laptop": 20,
    "Phone": 18,
    "Monitor": 16,
    "Headphones": 12,

    "Chair": 14,
    "Desk": 14,
    "Cabinet": 13,
    "Bookshelf": 13,

    "Printer": 8,
    "Paper Box": 8,
    "Toner": 7,
    "Stationery Kit": 7,
}


SEPTEMBER_PRODUCT_COUNTS: dict[str, int] = {
    "Laptop": 13,
    "Phone": 14,
    "Monitor": 15,
    "Headphones": 13,

    "Chair": 13,
    "Desk": 14,
    "Cabinet": 12,
    "Bookshelf": 12,

    "Printer": 8,
    "Paper Box": 9,
    "Toner": 8,
    "Stationery Kit": 6,
}


SALESPERSONS = [
    "Ali",
    "Sara",
    "Ahmed",
    "Ayesha",
    "Bilal",
    "Fatima",
]


def calculate_financials(
    product: str,
    quantity: int,
    unit_price: float,
    discount: float,
) -> tuple[float, float, float]:
    """
    Calculate revenue, cost, and profit for one order.

    Revenue:
        quantity * unit_price * (1 - discount)

    Cost:
        quantity * unit_price * product cost ratio

    Profit:
        revenue - cost
    """

    _, _, cost_ratio = PRODUCT_INFO[product]

    revenue = round(
        quantity * unit_price * (1 - discount),
        2,
    )

    cost = round(
        quantity * unit_price * cost_ratio,
        2,
    )

    profit = round(
        revenue - cost,
        2,
    )

    return revenue, cost, profit


def build_product_rows(
    rng: np.random.Generator,
    month: int,
) -> list[dict[str, object]]:
    """
    Build product-level order templates for one month.

    January through August use the normal product distribution.

    September intentionally has a different mix so that the agent
    can discover meaningful sales changes through analysis.
    """

    if month == 9:
        product_counts = SEPTEMBER_PRODUCT_COUNTS
    else:
        product_counts = NORMAL_MONTH_PRODUCT_COUNTS

    quantity_cycle = [2, 3]
    discount_cycle = [0.00, 0.05]

    rows: list[dict[str, object]] = []

    for product, count in product_counts.items():
        category, list_price, _ = PRODUCT_INFO[product]

        for index in range(count):
            quantity = quantity_cycle[
                index % len(quantity_cycle)
            ]

            discount = discount_cycle[
                (index + month) % len(discount_cycle)
            ]

            price_variation = rng.uniform(
                -0.02,
                0.02,
            )

            unit_price = round(
                list_price * (1 + price_variation),
                2,
            )

            rows.append(
                {
                    "product": product,
                    "category": category,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount": discount,
                }
            )

    rng.shuffle(rows)

    return rows


def build_regions(
    rng: np.random.Generator,
    month: int,
) -> list[str]:
    """
    Build the regional order distribution.

    September contains a deliberate regional shift so that the
    agent can investigate where the decline occurred.
    """

    if month == 9:
        regions = (
            ["North"] * 42
            + ["South"] * 46
            + ["East"] * 49
        )
    else:
        regions = (
            ["North"] * 60
            + ["South"] * 50
            + ["East"] * 40
        )

    rng.shuffle(regions)

    return regions


def generate_dataset() -> pd.DataFrame:
    """Generate the complete fictional sales dataset."""

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows: list[dict[str, object]] = []

    order_id = 1001

    for month in range(1, 10):
        product_rows = build_product_rows(
            rng=rng,
            month=month,
        )

        regions = build_regions(
            rng=rng,
            month=month,
        )

        if len(product_rows) != len(regions):
            raise RuntimeError(
                "Product row count and region count do not match."
            )

        for index, product_row in enumerate(
            product_rows
        ):
            product = str(
                product_row["product"]
            )

            category = str(
                product_row["category"]
            )

            quantity = int(
                product_row["quantity"]
            )

            unit_price = float(
                product_row["unit_price"]
            )

            discount = float(
                product_row["discount"]
            )

            revenue, cost, profit = (
                calculate_financials(
                    product=product,
                    quantity=quantity,
                    unit_price=unit_price,
                    discount=discount,
                )
            )

            day = int(
                rng.integers(
                    1,
                    29,
                )
            )

            order_date = pd.Timestamp(
                year=2026,
                month=month,
                day=day,
            )

            customer_id = (
                f"C{int(rng.integers(1, 301)):03d}"
            )

            salesperson = SALESPERSONS[
                int(
                    rng.integers(
                        0,
                        len(SALESPERSONS),
                    )
                )
            ]

            rows.append(
                {
                    "order_id": order_id,
                    "order_date": (
                        order_date.strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "customer_id": customer_id,
                    "product": product,
                    "category": category,
                    "region": regions[index],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount": discount,
                    "revenue": revenue,
                    "cost": cost,
                    "profit": profit,
                    "salesperson": salesperson,
                }
            )

            order_id += 1

    dataframe = pd.DataFrame(rows)

    return dataframe


def add_bulk_order_outliers(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a few intentionally unusual orders.

    These give the agent meaningful anomalies to discover when the
    user asks about unusual sales patterns.
    """

    dataframe = dataframe.copy()

    bulk_orders = {
        155: 10,
        477: 12,
        812: 14,
        1005: 11,
    }

    for row_index, new_quantity in (
        bulk_orders.items()
    ):
        if row_index >= len(dataframe):
            continue

        product = str(
            dataframe.loc[
                row_index,
                "product",
            ]
        )

        unit_price = float(
            dataframe.loc[
                row_index,
                "unit_price",
            ]
        )

        discount = float(
            dataframe.loc[
                row_index,
                "discount",
            ]
        )

        revenue, cost, profit = (
            calculate_financials(
                product=product,
                quantity=new_quantity,
                unit_price=unit_price,
                discount=discount,
            )
        )

        dataframe.loc[
            row_index,
            "quantity",
        ] = new_quantity

        dataframe.loc[
            row_index,
            "revenue",
        ] = revenue

        dataframe.loc[
            row_index,
            "cost",
        ] = cost

        dataframe.loc[
            row_index,
            "profit",
        ] = profit

    return dataframe


def add_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a small number of missing discount values.

    Revenue has already been calculated, so these missing values
    are intentional data-quality issues for the agent to detect.
    """

    dataframe = dataframe.copy()

    rng = np.random.default_rng(
        RANDOM_SEED + 100
    )

    eligible_indices = np.arange(
        0,
        min(
            1100,
            len(dataframe),
        ),
    )

    missing_indices = rng.choice(
        eligible_indices,
        size=12,
        replace=False,
    )

    dataframe.loc[
        missing_indices,
        "discount",
    ] = np.nan

    return dataframe


def validate_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """Perform basic validation before saving the CSV."""

    required_columns = [
        "order_id",
        "order_date",
        "customer_id",
        "product",
        "category",
        "region",
        "quantity",
        "unit_price",
        "discount",
        "revenue",
        "cost",
        "profit",
        "salesperson",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Generated dataset is missing columns: "
            + ", ".join(missing_columns)
        )

    if not 500 <= len(dataframe) <= 2000:
        raise RuntimeError(
            "Dataset row count must be between "
            "500 and 2,000."
        )

    if dataframe["order_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate order IDs were generated."
        )

    if (
        dataframe[
            [
                "quantity",
                "unit_price",
                "revenue",
                "cost",
                "profit",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Unexpected missing numeric values "
            "were generated."
        )


def print_summary(
    dataframe: pd.DataFrame,
) -> None:
    """Print a short validation summary."""

    dates = pd.to_datetime(
        dataframe["order_date"]
    )

    print("=" * 60)
    print("SALES DATASET GENERATED")
    print("=" * 60)

    print(
        f"Rows: {len(dataframe):,}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    print(
        "Date range: "
        f"{dates.min().date()} "
        "to "
        f"{dates.max().date()}"
    )

    print(
        "Missing discount values: "
        f"{dataframe['discount'].isna().sum()}"
    )

    print(
        "Total revenue: "
        f"${dataframe['revenue'].sum():,.2f}"
    )

    print(
        "Total profit: "
        f"${dataframe['profit'].sum():,.2f}"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


def main() -> None:
    """Generate, validate, and save the dataset."""

    dataframe = generate_dataset()

    dataframe = add_bulk_order_outliers(
        dataframe
    )

    dataframe = add_missing_values(
        dataframe
    )

    validate_dataset(
        dataframe
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        dataframe
    )


if __name__ == "__main__":
    main()