#!/usr/bin/env python3
"""Load CIFAR-10 dataset into DerivaML catalog using direct DerivaML API.

This script downloads the CIFAR-10 dataset from Kaggle and loads it into
a Deriva catalog using the DerivaML Python library directly (no MCP). It creates:
- An Image asset table for storing image files
- An Image_Class vocabulary with the 10 CIFAR-10 classes
- An Image_Classification feature linking images to their class labels
- A dataset hierarchy: Complete (all images), Segmented (Training + Testing)

Usage:
    python load_cifar10_direct.py --hostname ml.derivacloud.org --catalog-id 99
    python load_cifar10_direct.py --hostname localhost --create-catalog cifar10_demo

Requirements:
    - Kaggle CLI configured (~/.kaggle/kaggle.json)
    - deriva-ml package installed
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from deriva_ml import DerivaML
from deriva_ml.execution import ExecutionConfiguration
from deriva_ml.schema import create_ml_catalog
from deriva.core.ermrest_model import Schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CIFAR-10 class definitions
CIFAR10_CLASSES = [
    ("airplane", "Fixed-wing aircraft", ["plane", "aeroplane"]),
    ("automobile", "Motor vehicle with four wheels", ["car", "auto"]),
    ("bird", "Feathered flying vertebrate", []),
    ("cat", "Small domestic feline", ["kitten"]),
    ("deer", "Hoofed ruminant mammal", []),
    ("dog", "Domestic canine", ["puppy"]),
    ("frog", "Tailless amphibian", ["toad"]),
    ("horse", "Large domesticated hoofed mammal", ["pony"]),
    ("ship", "Large watercraft", ["boat", "vessel"]),
    ("truck", "Motor vehicle for transporting cargo", ["lorry"]),
]


def verify_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle credentials not found. Please configure ~/.kaggle/kaggle.json\n"
            "See: https://www.kaggle.com/docs/api#authentication"
        )
        return False
    return True


def download_cifar10(temp_dir: Path) -> Path:
    """Download CIFAR-10 dataset from Kaggle.

    Returns:
        Path to the extracted dataset directory
    """
    download_dir = temp_dir / "cifar10"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CIFAR-10 from Kaggle...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "cifar-10", "-p", str(download_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Kaggle download failed: {result.stderr}")
        raise RuntimeError(f"Failed to download CIFAR-10: {result.stderr}")

    # Extract the outer zip file
    zip_files = list(download_dir.glob("*.zip"))
    if zip_files:
        logger.info("Extracting outer zip archive...")
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(download_dir)

    # CIFAR-10 from Kaggle uses 7z archives for train/test data
    seven_z_files = list(download_dir.glob("*.7z"))
    if seven_z_files:
        logger.info("Extracting 7z archives (train.7z, test.7z)...")
        for seven_z_file in seven_z_files:
            result = subprocess.run(
                ["7z", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                result = subprocess.run(
                    ["7za", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to extract {seven_z_file.name}. "
                        "Please install 7-zip: brew install p7zip"
                    )

    return download_dir


def load_train_labels(data_dir: Path) -> dict[str, str]:
    """Load training labels from trainLabels.csv."""
    labels = {}
    labels_file = data_dir / "trainLabels.csv"

    if labels_file.exists():
        with open(labels_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["id"]] = row["label"]
    else:
        logger.warning("trainLabels.csv not found")

    return labels


def iter_images(data_dir: Path, split: str, labels: dict[str, str]):
    """Iterate over images with their class labels."""
    if split == "train":
        train_dir = data_dir / "train"
        if train_dir.exists():
            for img_path in sorted(train_dir.glob("*.png")):
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name:
                    yield img_path, class_name, image_id
    else:
        test_dir = data_dir / "test"
        if test_dir.exists():
            for img_path in sorted(test_dir.glob("*.png")):
                image_id = img_path.stem
                yield img_path, None, image_id


def setup_domain_model(ml: DerivaML) -> dict[str, Any]:
    """Create the domain model for CIFAR-10."""
    results = {}

    # Check existing vocabularies
    vocabs = [v.name for schema in [ml.ml_schema, ml.domain_schema]
              for v in ml.model.schemas[schema].tables.values()
              if ml.model.is_vocabulary(v)]

    # Create Image_Class vocabulary if needed
    if "Image_Class" not in vocabs:
        logger.info("Creating Image_Class vocabulary...")
        ml.create_vocabulary(
            vocabulary_name="Image_Class",
            comment="CIFAR-10 image classification categories",
        )
        results["vocabulary"] = {"status": "created", "name": "Image_Class"}
    else:
        logger.info("Image_Class vocabulary already exists")
        results["vocabulary"] = {"status": "exists", "name": "Image_Class"}

    # Add class terms
    existing_terms = {t["Name"] for t in ml.list_vocabulary_terms("Image_Class")}

    logger.info("Adding CIFAR-10 class terms...")
    for class_name, description, synonyms in CIFAR10_CLASSES:
        if class_name not in existing_terms:
            ml.add_term(
                vocabulary_name="Image_Class",
                term_name=class_name,
                description=description,
                synonyms=synonyms,
            )
            logger.info(f"  Added term: {class_name}")
        else:
            logger.info(f"  Term exists: {class_name}")

    # Check existing tables
    tables = [t.name for t in ml.model.schemas[ml.domain_schema].tables.values()]

    # Create Image asset table if needed
    if "Image" not in tables:
        logger.info("Creating Image asset table...")
        ml.create_asset(
            asset_name="Image",
            column_defs=[],
            comment="CIFAR-10 32x32 RGB images",
        )
        results["asset_table"] = {"status": "created", "table_name": "Image"}
    else:
        logger.info("Image asset table already exists")
        results["asset_table"] = {"status": "exists", "table_name": "Image"}

    # Enable Image as dataset element type
    logger.info("Enabling Image as dataset element type...")
    element_types = ml.list_dataset_element_types()
    if "Image" not in element_types:
        ml.add_dataset_element_type("Image")

    # Create Image_Classification featurewhat
    logger.info("Creating Image_Classification feature...")
    try:
        ml.create_feature(
            table_name="Image",
            feature_name="Image_Classification",
            comment="CIFAR-10 class label for each image",
            terms=["Image_Class"],
        )
        results["feature"] = {"status": "created", "feature_name": "Image_Classification"}
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("Image_Classification feature already exists")
            results["feature"] = {"status": "exists", "feature_name": "Image_Classification"}
        else:
            raise

    return results


def setup_workflow_type(ml: DerivaML) -> None:
    """Ensure Ingest workflow type exists."""
    workflow_types = {w.workflow_type for w in ml.list_workflows()}

    # Check if Ingest exists in Workflow_Type vocabulary
    existing_types = {t["Name"] for t in ml.list_vocabulary_terms("Workflow_Type")}

    if "Ingest" not in existing_types:
        logger.info("Creating Ingest workflow type...")
        ml.add_term(
            vocabulary_name="Workflow_Type",
            term_name="Ingest",
            description="Data ingestion workflow for loading external datasets",
        )


def setup_dataset_types(ml: DerivaML) -> None:
    """Ensure required dataset types exist in Dataset_Type vocabulary."""
    logger.info("Setting up dataset types...")

    required_types = [
        ("Complete", "A complete dataset containing all data", ["complete", "entire"]),
        ("Training", "A dataset subset used for model training", ["training", "train", "Train"]),
        ("Testing", "A dataset subset used for model testing/evaluation", ["test", "Test"]),
        ("Split", "A dataset that contains nested dataset splits", ["split"]),
    ]

    existing_terms = {t["Name"] for t in ml.list_vocabulary_terms("Dataset_Type")}

    for type_name, description, synonyms in required_types:
        if type_name not in existing_terms:
            try:
                ml.add_term(
                    vocabulary_name="Dataset_Type",
                    term_name=type_name,
                    description=description,
                    synonyms=synonyms,
                )
                logger.info(f"  Added dataset type: {type_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not add dataset type {type_name}: {e}")


def create_dataset_hierarchy(ml: DerivaML) -> dict[str, str]:
    """Create the dataset hierarchy."""
    datasets = {}

    logger.info("Creating dataset hierarchy...")

    # Create Complete dataset
    complete_ds = ml.create_dataset(
        description="Complete CIFAR-10 dataset with all labeled images",
        dataset_types=["Complete"],
    )
    datasets["complete"] = complete_ds.dataset_rid
    logger.info(f"  Created Complete dataset: {complete_ds.dataset_rid}")

    # Create Split dataset
    split_ds = ml.create_dataset(
        description="CIFAR-10 dataset split into training and testing subsets",
        dataset_types=["Split"],
    )
    datasets["split"] = split_ds.dataset_rid
    logger.info(f"  Created Split dataset: {split_ds.dataset_rid}")

    # Create Training dataset
    training_ds = ml.create_dataset(
        description="CIFAR-10 training set with 50,000 labeled images",
        dataset_types=["Training"],
    )
    datasets["training"] = training_ds.dataset_rid
    logger.info(f"  Created Training dataset: {training_ds.dataset_rid}")

    # Create Testing dataset
    testing_ds = ml.create_dataset(
        description="CIFAR-10 testing set",
        dataset_types=["Testing"],
    )
    datasets["testing"] = testing_ds.dataset_rid
    logger.info(f"  Created Testing dataset: {testing_ds.dataset_rid}")

    # Add Training and Testing as children of Split
    split_ds.add_nested_dataset(training_ds)
    split_ds.add_nested_dataset(testing_ds)
    logger.info("  Linked Training and Testing to Split dataset")

    return datasets


def load_images(
    ml: DerivaML,
    data_dir: Path,
    datasets: dict[str, str],
    batch_size: int = 500,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Load images into the catalog using execution system."""
    # Ensure Ingest workflow type exists
    setup_workflow_type(ml)

    # Create workflow
    logger.info("Creating execution for data loading...")
    workflow = ml.create_workflow(
        name="CIFAR-10 Data Load",
        workflow_type="Ingest",
        description="Load CIFAR-10 dataset images into DerivaML catalog",
    )

    # Create execution configuration
    config = ExecutionConfiguration(workflow=workflow)

    # Track images for dataset assignment
    train_images = []
    all_images = []

    # Use execution context manager
    with ml.create_execution(config) as exe:
        logger.info(f"  Execution RID: {exe.execution_rid}")

        # Load training labels
        labels = load_train_labels(data_dir)
        logger.info(f"Loaded {len(labels)} training labels")

        # Process training images
        limit_msg = f" (max: {max_images})" if max_images else ""
        logger.info(f"Registering training images for upload...{limit_msg}")
        count = 0
        for img_path, class_name, image_id in iter_images(data_dir, "train", labels):
            if class_name is None:
                continue

            if max_images and count >= max_images:
                logger.info(f"  Reached limit of {max_images} images")
                break

            # Create unique filename with class prefix
            new_filename = f"{class_name}_{image_id}.png"

            # Register file for upload
            exe.asset_file_path(
                asset_name="Image",
                file_name=str(img_path),
                asset_types=["Image"],
                copy_file=True,
                rename_file=new_filename,
            )

            train_images.append(new_filename)
            all_images.append(new_filename)
            count += 1

            if count % 1000 == 0:
                logger.info(f"  Registered {count} images...")

        logger.info(f"  Total training images registered: {count}")

    # Upload outputs (outside context manager)
    logger.info("Uploading images to catalog...")
    upload_result = exe.upload_execution_outputs(clean_folder=True)
    logger.info(f"  Upload complete")

    # Get uploaded image RIDs and assign to datasets
    logger.info("Assigning images to datasets...")
    assets = ml.list_assets("Image")
    logger.info(f"  Found {len(assets)} uploaded images")

    if assets:
        all_rids = [a["RID"] for a in assets]

        # Add all images to Complete dataset
        complete_ds = ml.lookup_dataset(datasets["complete"])
        logger.info("  Adding images to Complete dataset...")
        for i in range(0, len(all_rids), batch_size):
            batch = all_rids[i : i + batch_size]
            complete_ds.add_dataset_members(batch)
        logger.info(f"    Added {len(all_rids)} images")

        # Add training images to Training dataset
        training_ds = ml.lookup_dataset(datasets["training"])
        logger.info("  Adding images to Training dataset...")
        for i in range(0, len(all_rids), batch_size):
            batch = all_rids[i : i + batch_size]
            training_ds.add_dataset_members(batch)
        logger.info(f"    Added {len(all_rids)} images")

    return {
        "total_images": len(all_images),
        "training_images": len(train_images),
        "uploaded_assets": len(assets),
    }


def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    # Either create a new catalog or connect to existing one
    if args.create_catalog:
        logger.info(f"Creating new catalog on {args.hostname} with project name: {args.create_catalog}")

        # Create the catalog
        catalog = create_ml_catalog(args.hostname, args.create_catalog)
        model = catalog.getCatalogModel()

        # Create domain schema
        model.create_schema(Schema.define(args.create_catalog))

        catalog_id = catalog.catalog_id
        domain_schema = args.create_catalog

        print(f"\n{'='*60}")
        print(f"  CREATED NEW CATALOG")
        print(f"  Hostname:    {args.hostname}")
        print(f"  Catalog ID:  {catalog_id}")
        print(f"  Schema:      {domain_schema}")
        print(f"{'='*60}\n")

        # Connect to the newly created catalog
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(catalog_id),
            domain_schema=domain_schema,
            check_auth=True,
        )
    else:
        logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=str(args.catalog_id),
            domain_schema=args.domain_schema,
            check_auth=True,
        )
        catalog_id = args.catalog_id
        domain_schema = ml.domain_schema
        logger.info(f"Connected to catalog, domain schema: {domain_schema}")

    # Set up domain model
    logger.info("Setting up domain model...")
    setup_domain_model(ml)
    logger.info("Domain model setup complete")

    # Apply catalog annotations for Chaise web interface
    logger.info("Applying catalog annotations...")
    project_name = args.create_catalog if args.create_catalog else domain_schema
    ml.apply_catalog_annotations(
        navbar_brand_text=f"CIFAR-10 ({project_name})",
        head_title="CIFAR-10 ML Catalog",
    )

    # Setup dataset types and create dataset hierarchy
    setup_dataset_types(ml)
    datasets = create_dataset_hierarchy(ml)

    load_result = None
    if not args.dry_run:
        # Download CIFAR-10 from Kaggle
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = download_cifar10(temp_path)
            logger.info(f"Downloaded CIFAR-10 to: {data_dir}")

            # Load images
            load_result = load_images(
                ml, data_dir, datasets, args.batch_size, max_images=args.test
            )
            logger.info(f"Loading complete: {load_result}")
    else:
        logger.info("Dry run mode - skipping image download and upload")

    # Get Chaise URLs for datasets if requested
    dataset_urls = {}
    if args.show_urls:
        logger.info("Fetching Chaise URLs for datasets...")
        for name, rid in datasets.items():
            try:
                url = ml.chaise_url(rid)
                dataset_urls[name] = url
                logger.info(f"  {name}: {url}")
            except Exception as e:
                logger.warning(f"  Failed to get URL for {name}: {e}")
                dataset_urls[name] = ""

    # Print summary
    print("\n" + "=" * 60)
    print("  CIFAR-10 LOADING COMPLETE")
    print("=" * 60)
    print(f"  Hostname:      {args.hostname}")
    print(f"  Catalog ID:    {catalog_id}")
    print(f"  Schema:        {domain_schema}")
    print("")
    print("  Datasets created:")
    if args.show_urls and dataset_urls:
        print(f"    - Complete:   {datasets['complete']}")
        print(f"      URL: {dataset_urls.get('complete', 'N/A')}")
        print(f"    - Split:      {datasets['split']}")
        print(f"      URL: {dataset_urls.get('split', 'N/A')}")
        print(f"    - Training:   {datasets['training']}")
        print(f"      URL: {dataset_urls.get('training', 'N/A')}")
        print(f"    - Testing:    {datasets['testing']}")
        print(f"      URL: {dataset_urls.get('testing', 'N/A')}")
    else:
        print(f"    - Complete:   {datasets['complete']}")
        print(f"    - Split:      {datasets['split']}")
        print(f"    - Training:   {datasets['training']}")
        print(f"    - Testing:    {datasets['testing']}")
    if load_result:
        print("")
        print(f"  Images loaded: {load_result['total_images']}")
    if not args.show_urls:
        print("")
        print("  Tip: Use --show-urls to display Chaise URLs for each dataset")
    print("=" * 60 + "\n")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 dataset into DerivaML catalog (direct API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a new catalog and load CIFAR-10
    python load_cifar10_direct.py --hostname localhost --create-catalog cifar10_demo

    # Load into an existing catalog
    python load_cifar10_direct.py --hostname ml.derivacloud.org --catalog-id 99

    # Dry run (create schema/datasets only)
    python load_cifar10_direct.py --hostname localhost --create-catalog test --dry-run

    # Test mode (upload only 10 images)
    python load_cifar10_direct.py --hostname localhost --create-catalog test --test

    # Test mode with custom limit (upload 100 images)
    python load_cifar10_direct.py --hostname localhost --create-catalog test --test 100
        """,
    )
    parser.add_argument(
        "--hostname",
        required=True,
        help="Deriva server hostname (e.g., localhost, ml.derivacloud.org)",
    )

    catalog_group = parser.add_mutually_exclusive_group(required=True)
    catalog_group.add_argument(
        "--catalog-id",
        help="Catalog ID to connect to (for existing catalogs)",
    )
    catalog_group.add_argument(
        "--create-catalog",
        metavar="PROJECT_NAME",
        help="Create a new catalog with this project name",
    )

    parser.add_argument(
        "--domain-schema",
        help="Domain schema name (auto-detected if not provided)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of images to process per batch (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up schema and datasets without downloading/uploading images",
    )
    parser.add_argument(
        "--test",
        nargs="?",
        type=int,
        const=10,
        default=None,
        metavar="N",
        help="Test mode: upload only N images (default: 10 if flag used without value)",
    )
    parser.add_argument(
        "--show-urls",
        action="store_true",
        help="Show Chaise web interface URLs for datasets in the summary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Verify Kaggle credentials
    if not args.dry_run and not verify_kaggle_credentials():
        sys.exit(1)

    sys.exit(main(args))
