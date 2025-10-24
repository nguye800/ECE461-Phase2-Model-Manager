import sqlite3
import tempfile
import os
import datetime
import boto3
from moto import mock_aws


def create_dummy_sqlite_db(db_path=None):
    """Create a dummy SQLite database with the given schema."""
    if db_path is None:
        db_path = os.path.join(tempfile.gettempdir(), "dummy_test_db.sqlite")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id TEXT,
            model_url TEXT,
            date_time_entered_to_db TEXT,
            likes INTEGER,
            downloads INTEGER,
            license TEXT,
            github_link TEXT,
            github_numContributors INTEGER,
            base_models_modelID TEXT,
            base_model_urls TEXT,
            parameter_number INTEGER,
            gb_size_of_model REAL,
            dataset_link TEXT,
            dataset_name TEXT,
            s3_location_of_model_zip TEXT,
            s3_location_of_cloudwatch_log_for_database_entry TEXT
        );
    """)

    conn.commit()

    # add dummy data for 3 models
    models = [
        ("Qwen2.5-1.5B-Instruct", "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct", "Tongyi Qianwen Research License", "https://github.com/QwenLM/Qwen2", "QwenBase", "https://datasets.alibaba.com/qwen", 1.5e9, 3.4),
        ("Llama-3-8B", "https://huggingface.co/meta-llama/Llama-3-8B", "Meta Llama License", "https://github.com/meta-llama/llama3", "Llama-2", "https://datasets.meta.com/llama", 8e9, 16),
        ("Mistral-7B-v0.2", "https://huggingface.co/mistralai/Mistral-7B-v0.2", "Apache 2.0", "https://github.com/mistralai/mistral", "MistralBase", "https://huggingface.co/datasets/TinyStories", 7e9, 13.2),
    ]

    # Insert sample records with fake CloudWatch/S3 paths
    for name, url, lic, git, base, dataset, params, size in models:
        cursor.execute("""
        INSERT INTO model_registry (
            repo_id, model_url, date_time_entered_to_db, likes, downloads, license,
            github_link, github_numContributors, base_models_modelID, base_model_urls,
            parameter_number, gb_size_of_model, dataset_link, dataset_name,
            s3_location_of_model_zip, s3_location_of_cloudwatch_log_for_database_entry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, url, datetime.datetime.now().isoformat(), 1000, 1000000, lic,
            git, 20, base, f"https://huggingface.co/{base}", params, size,
            dataset, f"{name}-dataset",
            f"s3://dummy-model-bucket/{name}.zip",
            f"s3://dummy-model-bucket/{name}_cloudwatch_log.json"
        ))

    conn.commit()
    return conn


def teardown_dummy_sqlite_db(conn, db_path=None):
    """Completely remove the dummy SQLite database."""
    conn.close()
    if db_path and os.path.exists(db_path):
        os.remove(db_path)

# Moto automatically wipes state after exiting the @mock_s3 scope.
@mock_aws
def create_dummy_s3_environment():
    """Create a dummy S3 environment with fake files using Moto."""
    s3.create_bucket(Bucket="dummy-model-bucket")

    # Create fake model and logs
    s3.put_object(
        Bucket="dummy-model-bucket",
        Key="fake_model_zipfile.zip",
        Body=b"dummy model content"
    )

    s3.put_object(
        Bucket="dummy-model-bucket",
        Key="fake_aws_cloudwatch_systemactivity_logs.json",
        Body=b"{'log': 'this is a fake log entry'}"
    )

    return s3


# Example usage
if __name__ == "__main__":
    print("Setting up dummy SQLite DB...")
    conn = create_dummy_sqlite_db()
    print("Inserting a sample record...")

    # Read data from SQLite example
    print("\nFetching data from SQLite:")
    cursor = conn.cursor()
    cursor.execute("SELECT repo_id, model_url, license FROM model_registry LIMIT 3;")
    for row in cursor.fetchall():
        print(row)

    # setting up s3 and accessing contents example
    """ NOTE: you need to do the boto3.client line to setup the "s3" mock before actually creating a bucket
    boto3 is a aws sdk for python which reaches aws through s3 endpoint
    mock_aws intercepts the boto3 calls and returns fake aws responses """
    print("Setting up dummy S3 environment...")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3 = create_dummy_s3_environment()
        print("Dummy files in S3:")
        print(s3.list_objects_v2(Bucket="dummy-model-bucket")["Contents"])

    print("\nTearing down...")
    db_path = os.path.join(tempfile.gettempdir(), "dummy_test_db.sqlite")
    teardown_dummy_sqlite_db(conn, db_path)
    print("Teardown complete.")