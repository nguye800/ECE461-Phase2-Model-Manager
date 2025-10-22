import sqlite3
import tempfile
import os
import datetime
import boto3
from moto import mock_s3


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
    return conn


def teardown_dummy_sqlite_db(conn, db_path=None):
    """Completely remove the dummy SQLite database."""
    conn.close()
    if db_path and os.path.exists(db_path):
        os.remove(db_path)


@mock_s3
def create_dummy_s3_environment():
    """Create a dummy S3 environment with fake files using Moto."""
    s3 = boto3.client("s3", region_name="us-east-1")
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


def teardown_dummy_s3_environment(s3_client):
    """Teardown the Moto fake S3 environment."""
    # Moto automatically wipes state after exiting the @mock_s3 scope.
    pass


# Example usage
if __name__ == "__main__":
    print("Setting up dummy SQLite DB...")
    conn = create_dummy_sqlite_db()
    print("Inserting a sample record...")

    now = datetime.datetime.now().isoformat()
    conn.execute("""
        INSERT INTO model_registry (
            repo_id, model_url, date_time_entered_to_db, likes, downloads,
            license, github_link, github_numContributors, base_models_modelID,
            base_model_urls, parameter_number, gb_size_of_model, dataset_link,
            dataset_name, s3_location_of_model_zip,
            s3_location_of_cloudwatch_log_for_database_entry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "repo123", "http://example.com/model", now, 120, 3456, "MIT",
        "http://github.com/example/repo", 12, "base123", "http://example.com/base",
        54000000, 1.2, "http://example.com/dataset", "example-dataset",
        "s3://dummy-model-bucket/fake_model_zipfile.zip",
        "s3://dummy-model-bucket/fake_aws_cloudwatch_systemactivity_logs.json"
    ))

    conn.commit()
    print("Sample record inserted.\n")

    print("Setting up dummy S3 environment...")
    s3 = create_dummy_s3_environment()
    print("Dummy files in S3:")
    print(s3.list_objects_v2(Bucket="dummy-model-bucket")["Contents"])

    print("\nTearing down...")
    db_path = os.path.join(tempfile.gettempdir(), "dummy_test_db.sqlite")
    teardown_dummy_sqlite_db(conn, db_path)
    print("Teardown complete.")