"""Project constants, reference data, and documented modeling choices."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_ORDERS_PATH = DATA_DIR / "Nassau_Candy_Distributor.csv"
FACTORY_REFERENCE_PATH = DATA_DIR / "factory_reference.csv"
PRODUCT_MAPPING_PATH = DATA_DIR / "product_factory_mapping.csv"
ELIGIBILITY_PATH = DATA_DIR / "factory_eligibility.csv"
CENTROIDS_PATH = DATA_DIR / "state_centroids.csv"
GEOCODE_TEMPLATE_PATH = DATA_DIR / "templates" / "customer_geocodes.csv"

SEED = 42
MODEL_VERSION = "1.0.0"
DATE_FORMAT = "%d-%m-%Y"
DATE_FORMAT_LABEL = "day-month-year (%d-%m-%Y)"

# Business rule, not a statistical fence. Candy ship-date lead time beyond
# this is operationally implausible and must be reviewed, not silently dropped.
LONG_LEAD_TIME_DAYS = 60
MULTI_YEAR_LEAD_TIME_DAYS = 365
MIN_ROWS_TO_TRAIN = 50
MIN_SAMPLE_WARNING = 30
MIN_SAMPLE_STABLE = 100
MIN_SAMPLE_HIGHER_EVIDENCE = 100
TEST_FRACTION = 0.20
CV_MAE_TOLERANCE_DAYS = 1.0
SPEED_CAP_DAYS = 14.0
PROFIT_CAP_DOLLARS = 500.0
DEFAULT_RISK_WEIGHT = 0.40
DEFAULT_REFERENCE_MONTH = 6
DEFAULT_REFERENCE_DOW = 2  # Wednesday
MATERIAL_SLOW_GAP_DAYS = 1.0
RAW_DISAGREEMENT_DAYS = 5.0
RESIDUAL_DISAGREEMENT_DAYS = 1.0
GP_TOLERANCE = 0.01

ADVISORY_TEXT = (
    "Advisory only. A recommendation is an invitation to investigate, not "
    "authorization to change a factory assignment."
)

REQUIRED_COLUMNS = [
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Customer ID",
    "Country/Region",
    "City",
    "State/Province",
    "Postal Code",
    "Division",
    "Region",
    "Product ID",
    "Product Name",
    "Sales",
    "Units",
    "Gross Profit",
    "Cost",
]

COLUMN_MAP = {
    "Row ID": "row_id",
    "Order ID": "order_id",
    "Order Date": "order_date_raw",
    "Ship Date": "ship_date_raw",
    "Ship Mode": "ship_mode",
    "Customer ID": "customer_id",
    "Country/Region": "country",
    "City": "city",
    "State/Province": "state",
    "Postal Code": "postal_code",
    "Division": "division",
    "Region": "region",
    "Product ID": "product_id",
    "Product Name": "product_name_source",
    "Sales": "sales",
    "Units": "units",
    "Gross Profit": "gross_profit",
    "Cost": "cost",
}

ORIGIN_COLUMN_CANDIDATES = [
    "Origin Factory",
    "Fulfillment Factory",
    "Factory",
    "origin_factory",
    "OriginFactory",
]

# Decision-time features. Ship date, lead time, identifiers, and the order-id
# year cohort are intentionally absent.
CATEGORICAL_FEATURES = [
    "ship_mode",
    "region",
    "division",
    "product_name",
    "state",
    "country",
]
NUMERIC_FEATURES = ["units", "order_month", "order_dow", "order_quarter"]
BASE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

FORBIDDEN_FEATURES = {
    "ship_date",
    "ship_date_raw",
    "lead_time_days",
    "operational_residual_days",
    "descriptive_residual_days",
    "cohort_year",
    "cohort_median_days",
    "order_id",
    "row_id",
    "customer_id",
    "postal_code",
    "order_date",
    "order_date_raw",
    "sales",
    "cost",
    "gross_profit",
    "margin",
}

SHIP_MODE_ORDER = ["Same Day", "First Class", "Second Class", "Standard Class"]
REGION_ORDER = ["Atlantic", "Gulf", "Interior", "Pacific"]

MODEL_SIMPLICITY = {
    "group_median_baseline": 0,
    "linear_regression": 1,
    "random_forest": 2,
    "gradient_boosting": 3,
}

FACTORIES = [
    {"factory": "Lot's O' Nuts", "latitude": 32.881893, "longitude": -111.768036},
    {"factory": "Wicked Choccy's", "latitude": 32.076176, "longitude": -81.088371},
    {"factory": "Sugar Shack", "latitude": 48.119140, "longitude": -96.181150},
    {"factory": "Secret Factory", "latitude": 41.446333, "longitude": -90.565487},
    {"factory": "The Other Factory", "latitude": 35.117500, "longitude": -89.971107},
]

# Supplied current/default assignment. Division here is the mapping-table
# division, which does not always match the order extract.
PRODUCT_MAPPINGS = [
    {"division": "Chocolate", "product_name": "Wonka Bar - Nutty Crunch Surprise", "factory": "Lot's O' Nuts"},
    {"division": "Chocolate", "product_name": "Wonka Bar - Fudge Mallows", "factory": "Lot's O' Nuts"},
    {"division": "Chocolate", "product_name": "Wonka Bar - Scrumdiddlyumptious", "factory": "Lot's O' Nuts"},
    {"division": "Chocolate", "product_name": "Wonka Bar - Milk Chocolate", "factory": "Wicked Choccy's"},
    {"division": "Chocolate", "product_name": "Wonka Bar - Triple Dazzle Caramel", "factory": "Wicked Choccy's"},
    {"division": "Sugar", "product_name": "Laffy Taffy", "factory": "Sugar Shack"},
    {"division": "Sugar", "product_name": "SweeTARTS", "factory": "Sugar Shack"},
    {"division": "Sugar", "product_name": "Nerds", "factory": "Sugar Shack"},
    {"division": "Sugar", "product_name": "Fun Dip", "factory": "Sugar Shack"},
    {"division": "Other", "product_name": "Fizzy Lifting Drinks", "factory": "Sugar Shack"},
    {"division": "Sugar", "product_name": "Everlasting Gobstopper", "factory": "Secret Factory"},
    {"division": "Sugar", "product_name": "Hair Toffee", "factory": "The Other Factory"},
    {"division": "Other", "product_name": "Lickable Wallpaper", "factory": "Secret Factory"},
    {"division": "Other", "product_name": "Wonka Gum", "factory": "Secret Factory"},
    {"division": "Other", "product_name": "Kazookles", "factory": "The Other Factory"},
]

# Approximate geographic centroids for an explicit, opt-in straight-line
# sensitivity. These are not customer coordinates and not road distances.
STATE_CENTROIDS = [
    {"state": "Alabama", "latitude": 32.7794, "longitude": -86.8287, "country": "United States"},
    {"state": "Alaska", "latitude": 64.0685, "longitude": -152.2782, "country": "United States"},
    {"state": "Arizona", "latitude": 34.2744, "longitude": -111.6602, "country": "United States"},
    {"state": "Arkansas", "latitude": 34.8938, "longitude": -92.4426, "country": "United States"},
    {"state": "California", "latitude": 37.1841, "longitude": -119.4696, "country": "United States"},
    {"state": "Colorado", "latitude": 38.9972, "longitude": -105.5478, "country": "United States"},
    {"state": "Connecticut", "latitude": 41.6219, "longitude": -72.7273, "country": "United States"},
    {"state": "Delaware", "latitude": 38.9896, "longitude": -75.5050, "country": "United States"},
    {"state": "District of Columbia", "latitude": 38.9101, "longitude": -77.0147, "country": "United States"},
    {"state": "Florida", "latitude": 28.6305, "longitude": -82.4497, "country": "United States"},
    {"state": "Georgia", "latitude": 32.6415, "longitude": -83.4426, "country": "United States"},
    {"state": "Idaho", "latitude": 44.3509, "longitude": -114.6130, "country": "United States"},
    {"state": "Illinois", "latitude": 40.0417, "longitude": -89.1965, "country": "United States"},
    {"state": "Indiana", "latitude": 39.8942, "longitude": -86.2816, "country": "United States"},
    {"state": "Iowa", "latitude": 42.0751, "longitude": -93.4960, "country": "United States"},
    {"state": "Kansas", "latitude": 38.4937, "longitude": -98.3804, "country": "United States"},
    {"state": "Kentucky", "latitude": 37.5347, "longitude": -85.3021, "country": "United States"},
    {"state": "Louisiana", "latitude": 31.0689, "longitude": -91.9968, "country": "United States"},
    {"state": "Maine", "latitude": 45.3695, "longitude": -69.2428, "country": "United States"},
    {"state": "Maryland", "latitude": 39.0550, "longitude": -76.7909, "country": "United States"},
    {"state": "Massachusetts", "latitude": 42.2596, "longitude": -71.8083, "country": "United States"},
    {"state": "Michigan", "latitude": 44.3467, "longitude": -85.4102, "country": "United States"},
    {"state": "Minnesota", "latitude": 46.2807, "longitude": -94.3053, "country": "United States"},
    {"state": "Mississippi", "latitude": 32.7364, "longitude": -89.6678, "country": "United States"},
    {"state": "Missouri", "latitude": 38.3566, "longitude": -92.4580, "country": "United States"},
    {"state": "Montana", "latitude": 47.0527, "longitude": -109.6333, "country": "United States"},
    {"state": "Nebraska", "latitude": 41.5378, "longitude": -99.7951, "country": "United States"},
    {"state": "Nevada", "latitude": 39.3289, "longitude": -116.6312, "country": "United States"},
    {"state": "New Hampshire", "latitude": 43.6805, "longitude": -71.5811, "country": "United States"},
    {"state": "New Jersey", "latitude": 40.1907, "longitude": -74.6728, "country": "United States"},
    {"state": "New Mexico", "latitude": 34.4071, "longitude": -106.1126, "country": "United States"},
    {"state": "New York", "latitude": 42.9538, "longitude": -75.5268, "country": "United States"},
    {"state": "North Carolina", "latitude": 35.5557, "longitude": -79.3877, "country": "United States"},
    {"state": "North Dakota", "latitude": 47.4501, "longitude": -100.4659, "country": "United States"},
    {"state": "Ohio", "latitude": 40.2862, "longitude": -82.7937, "country": "United States"},
    {"state": "Oklahoma", "latitude": 35.5889, "longitude": -97.4943, "country": "United States"},
    {"state": "Oregon", "latitude": 43.9336, "longitude": -120.5583, "country": "United States"},
    {"state": "Pennsylvania", "latitude": 40.8781, "longitude": -77.7996, "country": "United States"},
    {"state": "Rhode Island", "latitude": 41.6762, "longitude": -71.5562, "country": "United States"},
    {"state": "South Carolina", "latitude": 33.9169, "longitude": -80.8964, "country": "United States"},
    {"state": "South Dakota", "latitude": 44.4443, "longitude": -100.2263, "country": "United States"},
    {"state": "Tennessee", "latitude": 35.8580, "longitude": -86.3505, "country": "United States"},
    {"state": "Texas", "latitude": 31.4757, "longitude": -99.3312, "country": "United States"},
    {"state": "Utah", "latitude": 39.3055, "longitude": -111.6703, "country": "United States"},
    {"state": "Vermont", "latitude": 44.0687, "longitude": -72.6658, "country": "United States"},
    {"state": "Virginia", "latitude": 37.5215, "longitude": -78.8537, "country": "United States"},
    {"state": "Washington", "latitude": 47.3826, "longitude": -120.4472, "country": "United States"},
    {"state": "West Virginia", "latitude": 38.6409, "longitude": -80.6227, "country": "United States"},
    {"state": "Wisconsin", "latitude": 44.6243, "longitude": -89.9941, "country": "United States"},
    {"state": "Wyoming", "latitude": 42.9957, "longitude": -107.5512, "country": "United States"},
    {"state": "Alberta", "latitude": 53.9333, "longitude": -116.5765, "country": "Canada"},
    {"state": "British Columbia", "latitude": 53.7267, "longitude": -127.6476, "country": "Canada"},
    {"state": "Manitoba", "latitude": 53.7609, "longitude": -98.8139, "country": "Canada"},
    {"state": "New Brunswick", "latitude": 46.5653, "longitude": -66.4619, "country": "Canada"},
    {"state": "Newfoundland and Labrador", "latitude": 53.1355, "longitude": -57.6604, "country": "Canada"},
    {"state": "Nova Scotia", "latitude": 44.6820, "longitude": -63.7443, "country": "Canada"},
    {"state": "Ontario", "latitude": 51.2538, "longitude": -85.3232, "country": "Canada"},
    {"state": "Prince Edward Island", "latitude": 46.5107, "longitude": -63.4168, "country": "Canada"},
    {"state": "Quebec", "latitude": 52.9399, "longitude": -73.5491, "country": "Canada"},
    {"state": "Saskatchewan", "latitude": 52.9399, "longitude": -106.4509, "country": "Canada"},
]

CENTROID_NOTE = (
    "Approximate geographic centroid for exploratory straight-line sensitivity. "
    "Not a customer coordinate. Not a road distance."
)

# Illustrative what-if only. Not company costs. The faster factory is more expensive
# so the speed/profit slider can change rank when these assumptions are loaded.
EXAMPLE_DELAY_COST = 6.0
EXAMPLE_HANDLING_DAYS = {
    "Lot's O' Nuts": -2.0,
    "Wicked Choccy's": 0.0,
    "Sugar Shack": 1.0,
    "Secret Factory": 3.0,
    "The Other Factory": -0.5,
}
EXAMPLE_COST_DELTA = {
    "Lot's O' Nuts": 12.0,
    "Wicked Choccy's": 0.0,
    "Sugar Shack": -8.0,
    "Secret Factory": -2.0,
    "The Other Factory": 1.0,
}

REASON_TEXT = {
    "FACTORY_EFFECT_NOT_OBSERVED": "Origin factory is not in the historical orders, so the model cannot learn a factory effect.",
    "ELIGIBILITY_UNVERIFIED": "This factory is not confirmed as eligible to make the product.",
    "INELIGIBLE": "This factory is marked ineligible and is excluded from actionable ranking.",
    "DATE_ANOMALY_DOMINATES_TARGET": "Parsed ship dates are years after order dates, so the raw lead-time target is not a credible ETA.",
    "LOW_SAMPLE": "Fewer than 30 historical orders match this product, region, and ship mode.",
    "LIMITED_SAMPLE": "Sample size is below 100, so the segment estimate is less stable.",
    "PROFIT_NOT_QUANTIFIED": "Freight and factory-cost inputs are missing, so profit impact is not quantified.",
    "UNVALIDATED_ASSUMPTION": "A distance or handling assumption is driving the scenario difference. It is not a historical result.",
    "DISTANCE_PROXY_ASSUMPTION": "Distance uses a straight-line proxy, not a carrier route.",
    "EXTRAPOLATION": "This product-region-ship-mode combination is thin or absent from history.",
    "MODEL_DISAGREEMENT": "The selected model and the comparison model disagree by more than the documented threshold.",
    "NO_LEAD_TIME_GAIN": "Model-supported lead-time change versus the current factory is zero or negative.",
    "PREDICTED_DETERIORATION": "The scenario is slower than the current factory under the stated assumptions.",
    "MISSING_COORDINATES": "No usable destination coordinate was available for the distance scenario.",
    "HIGH_VOLUME_EXPOSURE": "Historical gross profit on this lane is large enough to prioritize data collection.",
    "ADVISORY_ONLY": "A person must review this before any assignment change.",
    "OBSERVED_LEVER_IS_SHIP_MODE": "Within the date anomaly, ship mode is the only stable historical lead-time difference, and it is a few days.",
}
