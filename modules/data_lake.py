# data_lake.py
"""
BioreactorDataLake
- Provides clean interfaces to persist telemetry, anomalies, faults, agent actions, and metadata
- Uses Delta tables in the workspace metastore (not DBFS root paths)
"""

from typing import List, Dict, Any
from pyspark.sql import SparkSession
from pyspark.sql import Row
from datetime import datetime
import pandas as pd
import numpy as np
import json

def save_data(df, table_name: str, mode: str = "append"):
    """Save PySpark DataFrame to Delta table."""
    writer = df.write.format("delta").mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(table_name)


class BioreactorDataLake:
    def __init__(self, base_namespace: str = "biopilot"):
        """
        Args:
            base_namespace: schema/database to hold all tables (default: 'biopilot')
        """
        self.tables = {
            "telemetry": "/mnt/biopilot/telemetry",
            "anomaly_scores": "/mnt/biopilot/anomaly_scores",
            "agent_actions": "/mnt/biopilot/agent_actions",
            "run_metadata": "/mnt/biopilot/run_metadata"
        }
        self.base_namespace = base_namespace

    def initialize_schema(self, spark: SparkSession):
        """Ensure schema exists."""
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.base_namespace}")

    def _table(self, name: str) -> str:
        """Full table name with namespace."""
        return f"{self.base_namespace}.{name}"

    def save_telemetry(
        self, spark, run_id: str, telemetry_data: List[Dict], 
        is_observed: bool = True, batch_id: int = 0):
        """
        Save telemetry time-series data.
        """

        records = []
        timestamp = datetime.now()

        for snapshot in telemetry_data:
            time_h = float(snapshot.get('time', 0.0))
            for signal in ['X', 'S_glc', 'P', 'DO', 'pH']:
                if signal in snapshot:
                    value = snapshot[signal]
                    # Convert numpy types to native Python types
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

        df = spark.createDataFrame(records)
        save_data(df, self._table("telemetry"))

    def save_anomaly_scores(self, spark, run_id: str, anomaly_scores: List):
        """
        Save anomaly detection results.
        """
        records = []
        timestamp = datetime.now()

        for score_obj in anomaly_scores:
            records.append({
                'run_id': str(run_id),
                'time_h': float(score_obj.time),
                'signal_name': str(score_obj.signal_name if hasattr(score_obj, "signal_name") else score_obj.signal),
                'method': str(score_obj.method),
                'score': float(score_obj.score),
                'is_anomaly': bool(score_obj.is_anomaly),
                'context': json.dumps(score_obj.context) if score_obj.context else None,
                'timestamp': timestamp
            })

        df = spark.createDataFrame(records)
        save_data(df, self._table("anomalies"))

    def save_fault_log(self, spark: SparkSession, run_id: str, fault_id: str,
                       fault_type: str, start_time: float, duration: float,
                       parameters: Dict[str, Any]):
        df = spark.createDataFrame([
            Row(run_id=run_id,
                fault_id=fault_id,
                fault_type=fault_type,
                start_time=start_time,
                duration=duration,
                parameters=str(parameters))
        ])
        save_data(df, self._table("faults"))

    def save_agent_action(self, spark: SparkSession, run_id: str, action_id: str,
                          time: float, action_type: str, parameters: Dict[str, Any],
                          rationale: str):
        df = spark.createDataFrame([
            Row(run_id=run_id,
                action_id=action_id,
                time=time,
                action_type=action_type,
                parameters=str(parameters),
                rationale=rationale)
        ])
        save_data(df, self._table("agent_actions"))

    def save_run_metadata(self, spark: SparkSession, run_id: str, config: Dict[str, Any],
                          scenario: str, final_titer: float, num_anomalies: int,
                          num_actions: int, success: bool, score: float,
                          start_time: datetime, end_time: datetime):
        df = spark.createDataFrame([
            Row(run_id=run_id,
                config=str(config),
                scenario=scenario,
                final_titer=final_titer,
                num_anomalies=num_anomalies,
                num_actions=num_actions,
                success=success,
                score=score,
                start_time=start_time,
                end_time=end_time)
        ])
        save_data(df, self._table("run_metadata"))

    # ----------------------------
    # Query methods
    # ----------------------------
    def _read(self, spark, table_name: str):
        path = self.tables[table_name]
        return spark.read.format("delta").load(path)

    def get_run_telemetry(self, spark, run_id: str, is_observed: bool = True) -> pd.DataFrame:
        df = self._read(spark, "telemetry").filter(
            f"run_id = '{run_id}' AND is_observed = {str(is_observed).lower()}"
        )
        return df.toPandas()

    def get_anomalies(self, spark, run_id: str, only_detected: bool = False) -> pd.DataFrame:
        df = self._read(spark, "anomaly_scores").filter(f"run_id = '{run_id}'")
        if only_detected:
            df = df.filter("is_anomaly = true")
        return df.toPandas()

    def get_agent_actions(self, spark, run_id: str) -> pd.DataFrame:
        return self._read(spark, "agent_actions").filter(f"run_id = '{run_id}'").toPandas()

    def get_run_summary(self, spark, run_id: str) -> Dict:
        meta = self._read(spark, "run_metadata").filter(f"run_id = '{run_id}'").toPandas()
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
        for name, path in self.tables.items():
            df = self._read(spark, name)
            df.filter(f"run_id != '{run_id}'").write.format("delta").mode("overwrite").save(path)
