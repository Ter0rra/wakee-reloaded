"""
Test MLflow après correction
"""

import mlflow
import os

# Configure
MLFLOW_TRACKING_URI = "https://Terorra-wakee-mlflow.hf.space"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"🧪 Testing MLflow: {MLFLOW_TRACKING_URI}")

try:
    # Crée un experiment
    mlflow.set_experiment("test-after-fix")
    
    print("✅ Connected to MLflow")
    
    # Start run
    with mlflow.start_run(run_name="test-artifacts"):
        print("✅ Started run")
        
        # Log params
        mlflow.log_param("test_param", "hello")
        print("✅ Logged param")
        
        # Log metrics
        mlflow.log_metric("test_metric", 42)
        print("✅ Logged metric")
        
        # Log artifact
        with open("/tmp/test_artifact.txt", "w") as f:
            f.write("Test artifact from Python")
        
        mlflow.log_artifact("/tmp/test_artifact.txt")
        print("✅ Logged artifact")
        
        # Log model (simple)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        mlflow.sklearn.log_model(model, "test-model")
        print("✅ Logged model")
    
    print("\n🎉 All tests passed!")
    print(f"Check MLflow UI: {MLFLOW_TRACKING_URI}")
    print("Check R2 bucket: wr-mlflow/mlflow-artifacts/")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
