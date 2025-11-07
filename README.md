# Contributors
Daniel Brown
Michael Ray
Matthew Weiss
Brian Diaz

for run test:

1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -r dependencies.txt
4. poetry install
5. coverage run -m unittest tests/{file}.py or coverage run -m unittest discover for all tests
6. coverage report -m

## Lambda Containers & Deployment

- All Lambda functions (upload, download, metadata) now share the same container image defined in `dockerfiles/Dockerfile.lambda`. The entry point defaults to `upload.lambda_handler`, but the handler can be overridden per-function when configuring the Lambda in AWS.
- The workflow `.github/workflows/deploy-lambdas.yml` builds the image once, tags it three times (`upload-latest`, `download-latest`, `metadata-latest`), and pushes the tags to ECR (`ece461/function-containers`). Uncomment the final steps if you want the workflow to call `aws lambda update-function-code` automatically after a push.
- To hook a Lambda into API Gateway:
  1. Point the Lambda to the corresponding ECR image tag.
  2. Set the handler (for container images this is the `Command` field) to `upload.lambda_handler`, `download.lambda_handler`, or `metadata.lambda_handler`.
  3. Create/Update the API Gateway routes:
    - `POST /artifacts/{type}` → Upload Lambda
    - `PUT /artifacts/{type}/{id}` → Upload Lambda
    - `GET /artifacts/{type}/{id}` → Metadata Lambda (returns metadata + presigned download URLs)
    - `GET /artifact/{type}/{id}/lineage`, `/cost`, `/audit`, `POST /artifact/model/{id}/license-check` → Metadata Lambda

## Metadata Lambda Configuration

`src/metadata.py` exposes REST-style helpers backed by DynamoDB.

Environment variables:

| Variable | Description |
| --- | --- |
| `ARTIFACTS_DDB_TABLE` | DynamoDB table that stores both artifact metadata (`sk = META`) and audit entries (`sk = AUDIT#...`). Defaults to `ModelArtifacts`. |
| `ARTIFACT_COST_PER_GB` | Cost coefficient used by the `/cost` endpoint (USD per GB). Default `0.12`. |

Endpoints implemented:

| Method & Route | Behavior |
| --- | --- |
| `GET /artifacts/{type}/{id}` | Returns the stored metadata document (`pk = TYPE#ID`). |
| `GET /artifact/{type}/{id}/lineage` | Returns the `base_models` lineage from the metadata record. |
| `GET /artifact/{type}/{id}/cost` | Derives an estimated cost using `gb_size_of_model * ARTIFACT_COST_PER_GB`. |
| `GET /artifact/{type}/{id}/audit` | Queries audit items (`sk` begins with `AUDIT#...`). |
| `POST /artifact/model/{id}/license-check` | Evaluates permissive vs. restricted licenses (simple Apache/MIT/BSD check) and reports fine-tune/inference compatibility. |

The module is covered by `tests/test_metadata_lambda.py`, which stubs the DynamoDB table for deterministic results.
