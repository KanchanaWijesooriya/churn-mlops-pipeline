import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
import joblib
import pandas as pd
from google.cloud import storage

PROJECT_ID = "churn-mlops-pipeline"
DATASET = "churn_data"
SOURCE_TABLE = "cleaned_telco_churn"
DEST_TABLE = "churn_predictions"
BUCKET_NAME = "churn-mlops-pipeline-data-ckw"
TEMP_LOCATION = f"gs://{BUCKET_NAME}/dataflow-temp"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

class ScoreCustomer(beam.DoFn):
    def setup(self):
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("models/model.joblib")
        blob.download_to_filename("/tmp/model.joblib")

        bundle = joblib.load("/tmp/model.joblib")
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.scaler = bundle["scaler"]

    def process(self, row):
        customer_id = row["customerID"]
        input_dict = {k: v for k, v in row.items() if k not in ("customerID", "Churn")}

        # Match training: fill missing TotalCharges (tenure=0 customers) with 0
        # BEFORE encoding, so pandas doesn't mistake it for a categorical column
        if input_dict.get("TotalCharges") is None:
            input_dict["TotalCharges"] = 0.0

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)

        input_encoded[NUMERIC_COLS] = self.scaler.transform(input_encoded[NUMERIC_COLS])

        input_encoded = input_encoded.reindex(columns=self.feature_columns, fill_value=0)

        prediction = self.model.predict(input_encoded)[0]
        probability = self.model.predict_proba(input_encoded)[0][1]

        yield {
            "customerID": customer_id,
            "churn_prediction": bool(prediction),
            "churn_probability": float(round(probability, 4)),
        }

def run():
    options = PipelineOptions()
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT_ID
    google_cloud_options.temp_location = TEMP_LOCATION
    google_cloud_options.region = "us-east4"

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Read from BigQuery" >> beam.io.ReadFromBigQuery(
                query=f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{SOURCE_TABLE}`",
                use_standard_sql=True
              )
            | "Score customers" >> beam.ParDo(ScoreCustomer())
            | "Write to BigQuery" >> beam.io.WriteToBigQuery(
                table=f"{PROJECT_ID}:{DATASET}.{DEST_TABLE}",
                schema="customerID:STRING, churn_prediction:BOOLEAN, churn_probability:FLOAT",
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
              )
        )

if __name__ == "__main__":
    run()