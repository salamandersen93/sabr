# data_lake.py
"""
BioreactorDataLake
- Provides clean interfaces to persist telemetry, anomalies, faults, agent actions, and metadata
- Uses managed Delta tables in the Spark metastore (not DBFS paths)
"""

from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pyspark.sql import functions as F
from pyspark.sql import types as T


def save_data(df, table_name: str, mode: str = "append"):
    """Save PySpark DataFrame to Delta table in the metastore."""
    writer = df.write.format("delta").mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(table_name)


class BioreactorDataLake:
    def __init__(self, schema: str = "biopilot"):
        """
        Args:
            schema: Hive metastore schema (database) for all Delta tables.
        """
        self.schema = schema
        self.tables = {
            "telemetry": f"{schema}.telemetry",
            "anomaly_scores": f"{schema}.anomaly_scores",
            "agent_actions": f"{schema}.agent_actions",
            "run_metadata": f"{schema}.run_metadata",
            "fault_log": f"{schema}.fault_log"
        }

    # ----------------------------------------------------------------
    # Schema/table creation
    # ----------------------------------------------------------------
    def create_schema_and_tables(self, spark):
        """Create schema and Delta tables if they do not exist."""

        # Ensure schema exists
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

        # Telemetry
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['telemetry']} (
            run_id STRING,
            time_h DOUBLE,
            signal_name STRING,
            value DOUBLE,
            is_observed BOOLEAN,
            timestamp TIMESTAMP,
            batch_id INT
        ) USING DELTA
        """)

        # Anomaly Scores
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['anomaly_scores']} (
            run_id STRING,
            time_h DOUBLE,
            signal_name STRING,
            method STRING,
            score DOUBLE,
            is_anomaly BOOLEAN,
            context STRING,
            timestamp TIMESTAMP
        ) USING DELTA
        """)

        # Fault Log
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['fault_log']} (
            run_id STRING,
            fault_id STRING,
            fault_type STRING,
            start_time DOUBLE,
            duration DOUBLE,
            parameters STRING,
            timestamp TIMESTAMP
        ) USING DELTA
        """)

        # Agent Actions
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['agent_actions']} (
            run_id STRING,
            action_id STRING,
            time DOUBLE,
            action_type STRING,
            parameters STRING,
            rationale STRING,
            timestamp TIMESTAMP
        ) USING DELTA
        """)

        # Run Metadata
        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['run_metadata']} (
            run_id STRING,
            config STRING,
            scenario STRING,
            final_titer DOUBLE,
            num_anomalies INT,
            num_actions INT,
            success BOOLEAN,
            score DOUBLE,
            start_time TIMESTAMP,
            end_time TIMESTAMP
        ) USING DELTA
        """)

    # ----------------------------------------------------------------
    # Save methods
    # ----------------------------------------------------------------
    def save_telemetry(
        self, spark, run_id: str, telemetry_data: List[Dict],
        is_observed: bool = True, batch_id: int = 0):

        records = []
        timestamp = datetime.now()
        for snapshot in telemetry_data:
            time_h = float(snapshot.get('time', 0.0))
            for signal in ['X', 'S_glc', 'P', 'DO', 'pH']:
                if signal in snapshot:
                    value = snapshot[signal]
                    if isinstance(value, (np.generic, np.ndarray)):
                        value = float(value)
                    records.append({
                        'run_id': str(run_id),
                        'time_h': float(time_h),
                        'signal_name': str(signal),
                        'value': float(value),
                        'is_observed': bool(is_observed),
                        'timestamp': timestamp,
                        'batch_id': int(batch_id)
                    })
        if records:
            df = spark.createDataFrame(records)
            save_data(df, self.tables["telemetry"])

    def save_anomaly_scores(self, spark, run_id: str, anomaly_scores: List):
        timestamp = datetime.now()
        records = [{
            'run_id': str(run_id),
            'time_h': float(score_obj.time),
            'signal_name': str(getattr(score_obj, "signal_name", getattr(score_obj, "signal", ""))),
            'method': str(score_obj.method),
            'score': float(score_obj.score),
            'is_anomaly': bool(score_obj.is_anomaly),
            'context': json.dumps(score_obj.context) if score_obj.context else None,
            'timestamp': timestamp
        } for score_obj in anomaly_scores]
        if records:
            df = spark.createDataFrame(records)
            save_data(df, self.tables["anomaly_scores"])

    def save_fault_log(self, spark, run_id: str, fault_id: str,
                       fault_type: str, start_time: float, duration: float,
                       parameters: Dict[str, Any]):
        df = spark.createDataFrame([{
            'run_id': run_id,
            'fault_id': fault_id,
            'fault_type': fault_type,
            'start_time': float(start_time),
            'duration': float(duration),
            'parameters': json.dumps(parameters),
            'timestamp': datetime.now()
        }])
        save_data(df, self.tables["fault_log"])

    def save_agent_action(self, spark, run_id: str, action_id: str,
                          time: float, action_type: str, parameters: Dict[str, Any],
                          rationale: str):
        df = spark.createDataFrame([{
            'run_id': run_id,
            'action_id': action_id,
            'time': float(time),
            'action_type': action_type,
            'parameters': json.dumps(parameters),
            'rationale': rationale,
            'timestamp': datetime.now()
        }])
        save_data(df, self.tables["agent_actions"])

    def save_run_metadata(self, spark, run_id: str, config: Dict[str, Any],
                          scenario: str, final_titer: float, num_anomalies: int,
                          num_actions: int, success: bool, score: float,
                          start_time: datetime, end_time: datetime):
        df = spark.createDataFrame([{
            'run_id': run_id,
            'config': json.dumps(config),
            'scenario': scenario,
            'final_titer': float(final_titer),
            'num_anomalies': int(num_anomalies),
            'num_actions': int(num_actions),
            'success': bool(success),
            'score': float(score),
            'start_time': start_time,
            'end_time': end_time
        }])
        save_data(df, self.tables["run_metadata"])

    # ----------------------------------------------------------------
    # Query methods
    # ----------------------------------------------------------------
    def _read(self, spark, name: str):
        return spark.read.table(self.tables[name])

    def get_run_telemetry(self, spark, run_id: str, is_observed: bool = True) -> pd.DataFrame:
        df = self._read(spark, "telemetry").filter(
            (F.col("run_id") == run_id) & (F.col("is_observed") == is_observed)
        )
        return df.toPandas()

    def get_anomalies(self, spark, run_id: str, only_detected: bool = False) -> pd.DataFrame:
        df = self._read(spark, "anomaly_scores").filter(F.col("run_id") == run_id)
        if only_detected:
            df = df.filter(F.col("is_anomaly") == True)
        return df.toPandas()

    def get_agent_actions(self, spark, run_id: str) -> pd.DataFrame:
        df = self._read(spark, "agent_actions").filter(F.col("run_id") == run_id)
        return df.toPandas()

    def get_run_summary(self, spark, run_id: str) -> Dict:
        meta = self._read(spark, "run_metadata").filter(F.col("run_id") == run_id).toPandas()
        if meta.empty:
            return None
        telemetry = self.get_run_telemetry(spark, run_id, True)
        anomalies = self.get_anomalies(spark, run_id, True)
        actions = self.get_agent_actions(spark, run_id)
        return {
            'metadata': meta.to_dict('records')[0],
            'telemetry_points': len(telemetry),
            'anomalies_detected': len(anomalies),
            'actions_taken': len(actions),
            'telemetry_df': telemetry,
            'anomalies_df': anomalies,
            'actions_df': actions
        }

    def list_runs(self, spark, limit: int = 100) -> pd.DataFrame:
        df = self._read(spark, "run_metadata").orderBy("start_time", ascending=False).limit(limit)
        return df.toPandas()

    def delete_run(self, spark, run_id: str):
        for name, _ in self.tables.items():
            df = self._read(spark, name)
            df.filter(F.col("run_id") != run_id).write.format("delta").mode("overwrite").saveAsTable(self.tables[name])