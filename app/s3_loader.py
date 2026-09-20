import os
import boto3

BUCKET = "rag-2-alexkhol"
PREFIX = "artifacts"

FILES = [
    "faiss.index",
    "metadata.json"
]


def download():
    s3 = boto3.client("s3")

    os.makedirs("artifacts", exist_ok=True)

    for filename in FILES:
        local_path = f"artifacts/{filename}"
        s3_key = f"{PREFIX}/{filename}"

        s3.download_file(BUCKET, s3_key, local_path)

        print(f"Downloaded: {local_path}")


if __name__ == "__main__":
    download()
