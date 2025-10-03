# data_lake_sqlite.py
import sqlite3
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import json

class BioreactorDataLakeSQLite:
    """
    Provides clean interfaces to persist telemetry, anomalies, faults, agent actions, and metadata.
    Uses SQLite instead of Delta tables, fully compatible with Streamlit / local file usage.
    """

    def __init__(self, db_path: str = "sabr_db.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.tables = {
            "telemetry": "telemetry",
            "anomaly_scores": "anomaly_scores",
            "fault_log": "fault_log",
            "run_metadata": "run_metadata"
        }
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['telemetry']} (
            run_id TEXT,
            time_h REAL,
            signal_name TEXT,
            value REAL,
            is_observed INTEGER,
            timestamp TEXT,
            batch_id INTEGER
        )
        """)
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['anomaly_scores']} (
            run_id TEXT,
            time_h REAL,
            signal_name TEXT,
            method TEXT,
            score REAL,
            is_anomaly INTEGER,
            context TEXT,
            timestamp TEXT
        )
        """)
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['fault_log']} (
            run_id TEXT,
            fault_id TEXT,
            fault_type TEXT,
            start_time REAL,
            duration REAL,
            parameters TEXT,
            timestamp TEXT
        )
        """)
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.tables['run_metadata']} (
            run_id TEXT,
            config TEXT,
            scenario TEXT,
            final_titer REAL,
            num_anomalies INTEGER,
            success INTEGER,
            score REAL,
            start_time TEXT,
            end_time TEXT
        )
        """)
        self.conn.commit()

    # ===== Save Methods =====
    def save_telemetry(self, run_id: str, telemetry_data: List[Dict],
                       is_observed: bool = True, batch_id: int = 0):
        records = []
        timestamp = datetime.now().isoformat()
        for snapshot in telemetry_data:
            time_h = float(snapshot.get('time', 0.0))
            for signal in ['X', 'S_glc', 'P', 'DO', 'pH']:
                if signal in snapshot:
                    value = snapshot[signal]
                    if isinstance(value, (list, pd.Series)):
                        value = float(value[0])
                    records.append({
                        'run_id': run_id,
                        'time_h': time_h,
                        'signal_name': signal,
                        'value': float(value),
                        'is_observed': int(is_observed),
                        'timestamp': timestamp,
                        'batch_id': batch_id
                    })
        if records:
            pd.DataFrame(records).to_sql(self.tables['telemetry'], self.conn, if_exists='append', index=False)

    def save_anomaly_scores(self, run_id: str, anomaly_scores: List):
        timestamp = datetime.now().isoformat()
        records = []
        for score_obj in anomaly_scores:
            records.append({
                'run_id': run_id,
                'time_h': float(score_obj.time),
                'signal_name': getattr(score_obj, "signal_name", getattr(score_obj, "signal", "")),
                'method': str(score_obj.method),
                'score': float(score_obj.score),
                'is_anomaly': int(bool(score_obj.is_anomaly)),
                'context': json.dumps(score_obj.context) if score_obj.context else None,
                'timestamp': timestamp
            })
        if records:
            pd.DataFrame(records).to_sql(self.tables['anomaly_scores'], self.conn, if_exists='append', index=False)

    def save_fault_log(self, run_id: str, fault_id: str, fault_type: str,
                       start_time: float, duration: float, parameters: Dict[str, Any]):
        timestamp = datetime.now().isoformat()
        df = pd.DataFrame([{
            'run_id': run_id,
            'fault_id': fault_id,
            'fault_type': fault_type,
            'start_time': start_time,
            'duration': duration,
            'parameters': json.dumps(parameters),
            'timestamp': timestamp
        }])
        df.to_sql(self.tables['fault_log'], self.conn, if_exists='append', index=False)

    def save_run_metadata(self, run_id: str, config: Dict[str, Any],
                          scenario: str, final_titer: float, num_anomalies: int,
                          success: bool, score: float, start_time: datetime, end_time: datetime):
        df = pd.DataFrame([{
            'run_id': run_id,
            'config': json.dumps(config),
            'scenario': scenario,
            'final_titer': final_titer,
            'num_anomalies': num_anomalies,
            'success': int(success),
            'score': score,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }])
        df.to_sql(self.tables['run_metadata'], self.conn, if_exists='append', index=False)

    # ===== Get / Query Methods =====
    def get_run_telemetry(self, run_id: str, is_observed: bool) -> pd.DataFrame:
        query = f"""
        SELECT * FROM {self.tables['telemetry']}
        WHERE run_id = ? AND is_observed = ?
        """
        return pd.read_sql_query(query, self.conn, params=(run_id, int(is_observed)))

    def get_anomalies(self, run_id: str, only_detected: bool = False) -> pd.DataFrame:
        query = f"SELECT * FROM {self.tables['anomaly_scores']} WHERE run_id = ?"
        if only_detected:
            query += " AND is_anomaly = 1"
        return pd.read_sql_query(query, self.conn, params=(run_id,))

    def get_run_summary(self, run_id: str) -> Dict:
        meta = pd.read_sql_query(f"SELECT * FROM {self.tables['run_metadata']} WHERE run_id = ?", self.conn, params=(run_id,))
        if meta.empty:
            return None
        telemetry = self.get_run_telemetry(run_id, True)
        anomalies = self.get_anomalies(run_id, True)
        return {
            'metadata': json.loads(meta['config'].iloc[0]) | meta.iloc[0].to_dict(),
            'telemetry_points': len(telemetry),
            'anomalies_detected': len(anomalies),
            'telemetry_df': telemetry,
            'anomalies_df': anomalies
        }

    def list_runs(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql_query(
            f"SELECT * FROM {self.tables['run_metadata']} ORDER BY start_time DESC LIMIT ?", 
            self.conn, params=(limit,)
        )

    def delete_run(self, run_id: str):
        for table in self.tables.values():
            self.cursor.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        self.conn.commit()