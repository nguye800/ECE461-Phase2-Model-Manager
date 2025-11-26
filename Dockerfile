FROM public.ecr.aws/lambda/python:3.11
RUN yum install -y git && yum clean all
COPY search-lambda-requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY src ${LAMBDA_TASK_ROOT}/src

# Default runtime configuration for the Lambda container
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src \
    LOCAL_STORAGE_DIR=/tmp/model-manager \
    HF_HOME=/tmp/hf-cache \
    TRANSFORMERS_CACHE=/tmp/hf-cache \
    RATE_LOG_LEVEL=INFO

# Lambda entry point
CMD ["search.lambda_handler"]