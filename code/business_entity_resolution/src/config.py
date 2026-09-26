"""Central path configuration for the business entity resolution pipeline.

All paths are resolved relative to the ``student_resource`` root so the code
runs the same way regardless of the current working directory it is invoked
from.
"""

from pathlib import Path

# student_resource/code/business_entity_resolution/src/config.py -> student_resource/
ROOT_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = ROOT_DIR / "output"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Step 2: intermediate parquet data (gitignored, rebuilt by running the src/
# pipeline scripts in order -- never committed).
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DICTS_DIR = DATA_DIR / "dicts"
NORM_DIR = DATA_DIR / "norm"
CAND_DIR = DATA_DIR / "cand"

SOURCE_TSV = {
    ("train", "S1"): TRAIN_SOURCE1,
    ("train", "S2"): TRAIN_SOURCE2,
    ("train", "S3"): TRAIN_SOURCE3,
    ("test", "S1"): TEST_SOURCE1,
    ("test", "S2"): TEST_SOURCE2,
    ("test", "S3"): TEST_SOURCE3,
}
SPLITS = ("train", "test")
SOURCES = ("S1", "S2", "S3")


def raw_parquet_path(split, source):
    """Return the ingested parquet path for one (split, source) pair."""
    return RAW_DIR / f"{split}_{source}.parquet"


def norm_parquet_path(split, source):
    """Return the normalized-fields parquet path for one (split, source) pair."""
    return NORM_DIR / f"{split}_{source}.parquet"


TRAIN_GT_LONG_PARQUET = RAW_DIR / "train_gt_long.parquet"
TRAIN_GT_WIDE_PARQUET = RAW_DIR / "train_gt_wide.parquet"
TRAIN_SAMPLE_S1_PARQUET = DATA_DIR / "train_sample_s1.parquet"
FOLDS_PARQUET = DATA_DIR / "folds.parquet"
TEST_CANDIDATES_PARQUET = CAND_DIR / "test_candidates.parquet"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_LOG_PATH = OUTPUT_DIR / "results_log.csv"
BLOCKING_REPORT_PATH = OUTPUT_DIR / "BLOCKING_REPORT.md"
STEP2_REPORT_PATH = OUTPUT_DIR / "STEP2_REPORT.md"
