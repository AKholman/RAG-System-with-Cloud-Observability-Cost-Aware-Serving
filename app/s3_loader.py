import boto3, os

BUCKET = "rag-2-alexkhol"
PREFIX = "artifacts"
FILES = ["faiss.index", "metadata.json"]

def download():
    s3 = boto3.client("s3")
    os.makedirs("artifacts", exist_ok=True)
    for f in FILES:
        s3.download_file(BUCKET, f"{PREFIX}/{f}", f"artifacts/{f}")
        print(f"artifacts downloaded {f}")

if __name__ == "__main__":
    download()
