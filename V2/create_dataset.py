import csv
import os
import glob
import argparse


RAW_LOGS_DIR = os.path.join(os.path.dirname(__file__), "raw", "logs")
RAW_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "raw", "images")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_FILE = os.path.join(DATASET_DIR, "ground_truth.csv")


def find_csv_files(directory):
    return sorted(glob.glob(os.path.join(directory, "*.csv")))


def load_images_on_disk(images_dir):
    images = set()
    for f in os.listdir(images_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            images.add(f)
    return images


def create_dataset(logs_dir=None, images_dir=None, output_file=None):
    logs_dir = logs_dir or RAW_LOGS_DIR
    images_dir = images_dir or RAW_IMAGES_DIR
    output_file = output_file or OUTPUT_FILE

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    csv_files = find_csv_files(logs_dir)
    if not csv_files:
        print(f"No CSV files found in {logs_dir}")
        return

    images_on_disk = load_images_on_disk(images_dir)
    print(f"Found {len(images_on_disk)} images in {images_dir}")

    all_rows = []

    for csv_path in csv_files:
        print(f"Processing {os.path.basename(csv_path)}...")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                blob_id = row.get("captchaBlobId", "")
                captcha_value = row.get("captchaValue", "")
                if blob_id and blob_id in images_on_disk:
                    all_rows.append({"image": blob_id, "captcha_value": captcha_value})

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "captcha_value"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Matched {len(all_rows)} images with ground truth values")
    print(f"Written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ground truth dataset from raw captcha logs and images")
    parser.add_argument("--logs-dir", default=RAW_LOGS_DIR, help="Directory containing CSV log files")
    parser.add_argument("--images-dir", default=RAW_IMAGES_DIR, help="Directory containing captcha images")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output CSV file path")
    args = parser.parse_args()
    create_dataset(args.logs_dir, args.images_dir, args.output)