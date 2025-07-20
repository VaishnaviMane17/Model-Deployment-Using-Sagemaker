# Sagemaker Orchestration Script
# This script orchestrates the training and deployment of a custom sklearn model on AWS SageMaker.
"""This script is:
not running the model training directly
Instead, it's telling SageMaker:
where the data is (s3://bucket/prefix)
what training script to use (script.py)
what instance type and IAM role to use
It then:
Launches a SageMaker training job
Waits for it to complete
Deploys the trained model to a real-time endpoint"""""


from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime
import sagemaker
from sagemaker import Session
import boto3 
import os


sagemaker_session = Session()

# Constants
FRAMEWORK_VERSION = "0.23-1"
ROLE = "arn:aws:iam::068576877739:user/ThyroidModel"
BUCKET = "thyroidmodeldatat"
REGION = "us-east-2"  # US East (Ohio)
INSTANCE_TYPE_TRAIN = "ml.m5.large"
INSTANCE_TYPE_DEPLOY = "ml.m4.xlarge"

sm_boto3 = boto3.client("sagemaker", region_name=REGION)

def upload_data_to_s3(data_path: str, bucket: str, prefix: str):
    s3 = boto3.client('s3')
    for file_name in os.listdir(data_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_path, file_name)
            s3.upload_file(file_path, bucket, f"{prefix}/{file_name}")
            print(f"Uploaded {file_name} to s3://{bucket}/{prefix}/")
    return f"s3://{bucket}/{prefix}/"

def train_model(entry_point: str, train_path: str, test_path: str):

    sklearn_estimator = SKLearn(
        entry_point=entry_point,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE_TRAIN,
        framework_version=FRAMEWORK_VERSION,
        base_job_name="RF-custom-sklearn",
        hyperparameters={
            "n_estimators": 100,
            "random_state": 0,
        },
        use_spot_instances=True,
        max_wait=7200,
        max_run=3600,
        sagemaker_session=sagemaker_session
    )

    print("Launching training job...")
    sklearn_estimator.fit({"train": train_path, "test": test_path}, wait=True)

    artifact = sm_boto3.describe_training_job(
        TrainingJobName=sklearn_estimator.latest_training_job.name
    )["ModelArtifacts"]["S3ModelArtifacts"]

    print(f"Model artifact saved at: {artifact}")
    return artifact

def deploy_model(model_artifact: str, entry_point: str) -> str:
    """Deploy trained model to a SageMaker endpoint"""
    model_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    model = SKLearnModel(
        name=model_name,
        model_data=model_artifact,
        role=ROLE,
        entry_point=entry_point,
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sagemaker_session
    )

    endpoint_name = model_name
    print(f"Deploying to endpoint: {endpoint_name}")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE_DEPLOY,
        endpoint_name=endpoint_name
    )

    print(f"Endpoint deployed at: {endpoint_name}")
    return endpoint_name


def main():

    entry_point_script = "script.py"  # Your training script with model.save
    train_path = f"s3://{BUCKET}/train-V-1.csv"
    test_path = f"s3://{BUCKET}/test-V-1.csv"

    # Step 1: Train
    model_artifact = train_model(entry_point_script, train_path, test_path)

    # Step 2: Deploy
    endpoint = deploy_model(model_artifact, entry_point_script)

    print(f"Model deployed successfully!\nEndpoint: {endpoint}")

if __name__ == "__main__":
    main()