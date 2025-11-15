from db.bmrep_db import BeaconMeasurementDB
from db.bsstm_db import BSSTransitionResponseDB
from db.lmrep_db import LinkMeasurementDB
from db.neighbor_db import NeighborDB
from db.qoe_db import QoEDB
from db.station_db import StationDB
from logger import Logger
import threading
import time
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class Routine:
    def __init__(
        self,
        output_dir: str,
        db_initials: Optional[Dict[str, str]] = None,
        batch_interval: float = 30.0,
        retention_days: Optional[int] = None,
        organize_by: str = "date"  # "date", "hour", or "flat"
    ):
        """
        Enhanced database persistence routine with organized timestamped storage.
        
        Args:
            output_dir: Root directory for database storage
            db_initials: Dictionary mapping db names to filename initials
            batch_interval: How often (seconds) to persist a batch
            retention_days: Auto-delete batches older than this (None = keep all)
            organize_by: How to organize files - "date" (YYYY/MM/DD/), "hour" (YYYY/MM/DD/HH/), or "flat"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = self.output_dir / "_metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.db_initials = db_initials or {
            "bmdb": "BM",
            "lmdb": "LM",
            "qoedb": "QOE",
            "stdb": "ST",
            "neighbordb": "NB",
            "bsstmdb": "BSSTM",
        }
        
        # Initialize databases
        self.databases = {
            "bmdb": BeaconMeasurementDB(),
            "lmdb": LinkMeasurementDB(),
            "qoedb": QoEDB(),
            "stdb": StationDB(),
            "neighbordb": NeighborDB(),
            "bsstmdb": BSSTransitionResponseDB(),
        }
        
        # Configuration
        self.batch_interval = batch_interval
        self.retention_days = retention_days
        self.organize_by = organize_by
        
        # State tracking
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._batch_counter = 0
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_errors = 0
        self._total_saves = 0
        self._session_metadata = {
            "session_id": self._session_id,
            "start_time": datetime.now().isoformat(),
            "databases": list(self.databases.keys()),
            "config": {
                "batch_interval": batch_interval,
                "organize_by": organize_by,
                "retention_days": retention_days
            },
            "saves": []
        }
        
        # Start background thread
        self._thread = threading.Thread(target=self._run, daemon=True, name="RoutineThread")
        self._thread.start()
        Logger.log_info(f"[Routine] Started session {self._session_id}")

    def _get_organized_path(self, timestamp: datetime) -> Path:
        """
        Get organized directory path based on timestamp.
        
        Returns path like:
        - date: output_dir/2025/11/15/
        - hour: output_dir/2025/11/15/14/
        - flat: output_dir/
        """
        if self.organize_by == "date":
            return self.output_dir / timestamp.strftime("%Y/%m/%d")
        elif self.organize_by == "hour":
            return self.output_dir / timestamp.strftime("%Y/%m/%d/%H")
        else:  # flat
            return self.output_dir

    def _generate_filename(self, db_name: str, timestamp: datetime) -> str:
        """
        Generate organized filename with timestamp.
        Format: {initial}_{YYYYMMDD_HHMMSS}_{batch}.json
        Example: BM_20251115_143022_0042.json
        """
        initial = self.db_initials.get(db_name, db_name)
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{initial}_{ts_str}_{self._batch_counter:04d}.json"

    def _get_db_snapshot(self, db_name: str, db_obj: Any, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get a snapshot of a database with full metadata."""
        try:
            if not hasattr(db_obj, "to_dict"):
                Logger.log_warn(f"[Routine] {db_name} has no to_dict() method, skipping")
                return None
            
            with self._lock:
                data = db_obj.to_dict()
            
            # Calculate data statistics
            record_count = None
            data_size = None
            if isinstance(data, dict):
                record_count = len(data)
            elif isinstance(data, list):
                record_count = len(data)
            
            try:
                data_size = len(json.dumps(data))
            except:
                pass
            
            # Build comprehensive snapshot
            return {
                "metadata": {
                    "db_name": db_name,
                    "db_type": type(db_obj).__name__,
                    "session_id": self._session_id,
                    "batch_number": self._batch_counter,
                    "timestamp": timestamp.isoformat(),
                    "unix_timestamp": int(timestamp.timestamp()),
                    "record_count": record_count,
                    "data_size_bytes": data_size,
                    "version": "1.0"
                },
                "data": data
            }
        except Exception as e:
            Logger.log_err(f"[Routine] Error getting snapshot for {db_name}: {e}")
            return None

    def _save_batch(self) -> Dict[str, Any]:
        """Save all database snapshots with metadata tracking."""
        timestamp = datetime.now()
        batch_metadata = {
            "batch_number": self._batch_counter,
            "timestamp": timestamp.isoformat(),
            "unix_timestamp": int(timestamp.timestamp()),
            "session_id": self._session_id,
            "databases": {},
            "summary": {
                "total_databases": 0,
                "successful_saves": 0,
                "failed_saves": 0,
                "total_records": 0,
                "total_size_bytes": 0
            }
        }
        
        try:
            # Get organized directory for this batch
            batch_dir = self._get_organized_path(timestamp)
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            for db_name, db_obj in self.databases.items():
                db_metadata = {
                    "db_name": db_name,
                    "status": "pending",
                    "filepath": None,
                    "error": None
                }
                
                try:
                    snapshot = self._get_db_snapshot(db_name, db_obj, timestamp)
                    if snapshot is None:
                        db_metadata["status"] = "skipped"
                        db_metadata["error"] = "No to_dict() method"
                        batch_metadata["databases"][db_name] = db_metadata
                        continue
                    
                    # Generate filename and path
                    filename = self._generate_filename(db_name, timestamp)
                    filepath = batch_dir / filename
                    
                    # Write atomically
                    temp_filepath = filepath.with_suffix('.tmp')
                    with open(temp_filepath, 'w') as f:
                        json.dump(snapshot, f, indent=2)
                    temp_filepath.rename(filepath)
                    
                    # Update metadata
                    db_metadata["status"] = "success"
                    db_metadata["filepath"] = str(filepath.relative_to(self.output_dir))
                    db_metadata["record_count"] = snapshot["metadata"].get("record_count")
                    db_metadata["data_size_bytes"] = snapshot["metadata"].get("data_size_bytes")
                    
                    batch_metadata["summary"]["successful_saves"] += 1
                    batch_metadata["summary"]["total_records"] += (db_metadata["record_count"] or 0)
                    batch_metadata["summary"]["total_size_bytes"] += (db_metadata["data_size_bytes"] or 0)
                    
                    saved_files.append(filename)
                    
                except Exception as e:
                    Logger.log_err(f"[Routine] Failed to save {db_name}: {e}")
                    db_metadata["status"] = "failed"
                    db_metadata["error"] = str(e)
                    batch_metadata["summary"]["failed_saves"] += 1
                
                batch_metadata["databases"][db_name] = db_metadata
            
            batch_metadata["summary"]["total_databases"] = len(self.databases)
            
            # Save batch metadata
            self._save_batch_metadata(batch_metadata, batch_dir)
            
            # Update session metadata
            self._session_metadata["saves"].append({
                "batch_number": self._batch_counter,
                "timestamp": timestamp.isoformat(),
                "successful_saves": batch_metadata["summary"]["successful_saves"],
                "failed_saves": batch_metadata["summary"]["failed_saves"]
            })
            self._save_session_metadata()
            
            # Log results
            if saved_files:
                Logger.log_info(
                    f"[Routine] Batch #{self._batch_counter} saved: "
                    f"{len(saved_files)}/{len(self.databases)} databases, "
                    f"{batch_metadata['summary']['total_records']} total records"
                )
                self._total_saves += 1
            
            if batch_metadata["summary"]["failed_saves"] > 0:
                self._save_errors += batch_metadata["summary"]["failed_saves"]
            
            return batch_metadata
            
        except Exception as e:
            Logger.log_err(f"[Routine] Critical error during batch save: {e}")
            self._save_errors += 1
            batch_metadata["error"] = str(e)
            return batch_metadata

    def _save_batch_metadata(self, metadata: Dict[str, Any], batch_dir: Path):
        """Save metadata for a specific batch."""
        try:
            metadata_file = batch_dir / f"_batch_{self._batch_counter:04d}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            Logger.log_err(f"[Routine] Failed to save batch metadata: {e}")

    def _save_session_metadata(self):
        """Save overall session metadata."""
        try:
            self._session_metadata["last_updated"] = datetime.now().isoformat()
            self._session_metadata["statistics"] = {
                "total_batches": self._batch_counter + 1,
                "total_saves": self._total_saves,
                "total_errors": self._save_errors
            }
            
            session_file = self.metadata_dir / f"session_{self._session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(self._session_metadata, f, indent=2)
            
            # Also maintain a "latest" symlink or copy
            latest_file = self.metadata_dir / "latest_session.json"
            with open(latest_file, 'w') as f:
                json.dump(self._session_metadata, f, indent=2)
                
        except Exception as e:
            Logger.log_err(f"[Routine] Failed to save session metadata: {e}")

    def _cleanup_old_batches(self):
        """Remove batches older than retention_days."""
        if self.retention_days is None:
            return
        
        try:
            cutoff_time = time.time() - (self.retention_days * 86400)
            deleted_count = 0
            
            # Walk through organized directory structure
            for dirpath, dirnames, filenames in os.walk(self.output_dir):
                dirpath = Path(dirpath)
                
                # Skip metadata directory
                if self.metadata_dir in dirpath.parents or dirpath == self.metadata_dir:
                    continue
                
                # Check files in this directory
                for filename in filenames:
                    if filename.endswith('.json') and not filename.startswith('_'):
                        filepath = dirpath / filename
                        if filepath.stat().st_mtime < cutoff_time:
                            filepath.unlink()
                            deleted_count += 1
            
            if deleted_count > 0:
                Logger.log_info(f"[Routine] Cleaned up {deleted_count} old batch files")
                
        except Exception as e:
            Logger.log_err(f"[Routine] Error during cleanup: {e}")

    def _run(self):
        """Main routine loop."""
        while not self._stop_event.is_set():
            try:
                # Save current batch
                self._save_batch()
                
                # Increment counter
                self._batch_counter += 1
                
                # Periodic cleanup (every 10 batches)
                if self._batch_counter % 10 == 0:
                    self._cleanup_old_batches()
                
            except Exception as e:
                Logger.log_err(f"[Routine] Unexpected error in main loop: {e}")
                self._save_errors += 1
            
            # Wait for next interval
            self._stop_event.wait(self.batch_interval)

    def stop(self, save_final: bool = True):
        """Stop the routine and save final metadata."""
        Logger.log_info("[Routine] Stopping database persistence routine...")
        self._stop_event.set()
        
        if save_final:
            Logger.log_info("[Routine] Saving final batch...")
            self._save_batch()
        
        # Update session end time
        self._session_metadata["end_time"] = datetime.now().isoformat()
        self._save_session_metadata()
        
        self._thread.join(timeout=10.0)
        
        if self._thread.is_alive():
            Logger.log_warn("[Routine] Thread did not stop gracefully")
        else:
            Logger.log_info(
                f"[Routine] Session {self._session_id} ended. "
                f"Total saves: {self._total_saves}, Errors: {self._save_errors}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        return {
            "session_id": self._session_id,
            "batch_counter": self._batch_counter,
            "total_saves": self._total_saves,
            "save_errors": self._save_errors,
            "is_running": self._thread.is_alive(),
            "output_dir": str(self.output_dir),
            "batch_interval": self.batch_interval,
            "organize_by": self.organize_by
        }

    def force_save(self) -> Dict[str, Any]:
        """Manually trigger an immediate save."""
        Logger.log_info("[Routine] Manual save triggered")
        return self._save_batch()

    def get_latest_snapshot(self) -> Dict[str, Any]:
        """Get current in-memory snapshots without saving."""
        timestamp = datetime.now()
        snapshots = {}
        for db_name, db_obj in self.databases.items():
            snapshot = self._get_db_snapshot(db_name, db_obj, timestamp)
            if snapshot:
                snapshots[db_name] = snapshot
        return snapshots

    def list_saved_batches(self, limit: int = 10) -> list:
        """List recent saved batches with metadata."""
        batches = []
        try:
            metadata_files = sorted(
                self.metadata_dir.glob("session_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            for meta_file in metadata_files[:limit]:
                with open(meta_file) as f:
                    batches.append(json.load(f))
                    
        except Exception as e:
            Logger.log_err(f"[Routine] Error listing batches: {e}")
        
        return batches