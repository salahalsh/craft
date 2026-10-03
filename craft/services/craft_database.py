"""
CRAFT Database - Pre-computed SQLite vector store for fast lookup.

Stores CRAFT fingerprints for ~15K ChEMBL human single-protein targets.
Used by similarity search and fast lookup (bypassing API calls).

Schema:
    targets(target_id TEXT PK, uniprot_id, chembl_id, gene_name,
            target_name, target_type, organism, vector_json TEXT,
            mode TEXT, total_on_bits INT, data_completeness REAL,
            created_at TEXT)
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default database location
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'craft_database.sqlite'


class CRAFTDatabase:
    """SQLite-backed pre-computed CRAFT fingerprint store."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH

    def is_available(self) -> bool:
        """Check if the pre-computed database exists and has data."""
        if not self.db_path.exists():
            return False
        try:
            conn = self._connect()
            try:
                count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
                return count > 0
            finally:
                conn.close()
        except Exception:
            return False

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode enabled for safe concurrent reads.

        Callers MUST close the returned connection (use try/finally or 'with').
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def initialize(self):
        """Create the database schema if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS targets (
                    target_id TEXT PRIMARY KEY,
                    uniprot_id TEXT,
                    chembl_id TEXT,
                    gene_name TEXT,
                    target_name TEXT,
                    target_type TEXT,
                    organism TEXT DEFAULT 'Homo sapiens',
                    vector_json TEXT NOT NULL,
                    mode TEXT DEFAULT 'binary',
                    total_on_bits INTEGER DEFAULT 0,
                    data_completeness REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_uniprot ON targets(uniprot_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chembl ON targets(chembl_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gene ON targets(gene_name)")
            conn.commit()
        finally:
            conn.close()

    def store(self, fingerprint) -> bool:
        """Store a CRAFTFingerprint in the database."""
        try:
            self.initialize()
            conn = self._connect()
            try:
                vector_json = json.dumps(fingerprint.vector.tolist())
                conn.execute("""
                    INSERT OR REPLACE INTO targets
                    (target_id, uniprot_id, chembl_id, gene_name, target_name,
                     target_type, organism, vector_json, mode, total_on_bits,
                     data_completeness)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fingerprint.target_id,
                    fingerprint.uniprot_id,
                    fingerprint.chembl_id,
                    fingerprint.gene_name,
                    fingerprint.target_name,
                    fingerprint.target_type,
                    fingerprint.organism,
                    vector_json,
                    fingerprint.mode,
                    fingerprint.total_on_bits,
                    fingerprint.data_completeness,
                ))
                conn.commit()
            finally:
                conn.close()
            return True
        except Exception as e:
            logger.error("Failed to store fingerprint for %s: %s",
                         fingerprint.target_id, e)
            return False

    def lookup(self, identifier: str) -> Optional[Dict]:
        """
        Look up a pre-computed fingerprint by any identifier.

        Searches: target_id, uniprot_id, chembl_id, gene_name
        """
        if not self.is_available():
            return None

        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM targets WHERE target_id = ? OR uniprot_id = ? "
                "OR chembl_id = ? OR gene_name = ?",
                (identifier, identifier, identifier, identifier)
            ).fetchone()
        finally:
            conn.close()

        if row:
            return self._row_to_dict(row)
        return None

    def get_all_vectors(self) -> List[Dict]:
        """Load all target vectors for similarity search."""
        if not self.is_available():
            return []

        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT target_id, gene_name, target_type, vector_json FROM targets"
            ).fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            results.append({
                'target_id': row['target_id'],
                'gene_name': row['gene_name'],
                'target_type': row['target_type'],
                'vector': json.loads(row['vector_json']),
            })
        return results

    def get_stats(self) -> Dict:
        """Get database statistics."""
        if not self.is_available():
            return {'total_targets': 0, 'available': False}

        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            avg_bits = conn.execute(
                "SELECT AVG(total_on_bits) FROM targets"
            ).fetchone()[0] or 0
            avg_comp = conn.execute(
                "SELECT AVG(data_completeness) FROM targets"
            ).fetchone()[0] or 0

            # Type distribution
            types = conn.execute(
                "SELECT target_type, COUNT(*) as cnt FROM targets "
                "GROUP BY target_type ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
        finally:
            conn.close()

        return {
            'available': True,
            'total_targets': total,
            'avg_on_bits': round(avg_bits, 1),
            'avg_completeness': round(avg_comp, 3),
            'top_types': {row[0]: row[1] for row in types},
            'db_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 1),
        }

    def _row_to_dict(self, row) -> Dict:
        """Convert a database row to dict with parsed vector."""
        return {
            'target_id': row['target_id'],
            'uniprot_id': row['uniprot_id'],
            'chembl_id': row['chembl_id'],
            'gene_name': row['gene_name'],
            'target_name': row['target_name'],
            'target_type': row['target_type'],
            'organism': row['organism'],
            'vector': json.loads(row['vector_json']),
            'mode': row['mode'],
            'total_on_bits': row['total_on_bits'],
            'data_completeness': row['data_completeness'],
        }
