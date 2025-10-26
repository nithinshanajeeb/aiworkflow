# db.py
"""PostgreSQL persistence utilities for intake applications."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

DEFAULT_TABLE_NAME = "applications"
DOCUMENTS_TABLE_SUFFIX = "_documents"


@dataclass
class ApplicationRecord:
    """Database representation of an intake application."""

    id: int
    full_name: str
    dob: Optional[date]
    nationality: Optional[str]
    emirate: Optional[str]
    emirates_id: Optional[str]
    employment_status: Optional[str]
    monthly_income_aed: Optional[float]
    credit_score: Optional[int]
    contact_pref: Optional[str]
    best_time: Optional[str]
    notes: Optional[str]
    decision: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ApplicationRecord":
        return cls(**row)  # type: ignore[arg-type]


@dataclass
class ApplicationCreate:
    """Fields accepted when inserting a new application."""

    full_name: str
    dob: Optional[date] = None
    nationality: Optional[str] = None
    emirate: Optional[str] = None
    emirates_id: Optional[str] = None
    employment_status: Optional[str] = None
    monthly_income_aed: Optional[float] = None
    credit_score: Optional[int] = None
    contact_pref: Optional[str] = None
    best_time: Optional[str] = None
    notes: Optional[str] = None
    decision: Optional[Dict[str, Any]] = None


@dataclass
class DocumentMarkdownRecord:
    """Database representation of extracted document content."""

    id: int
    application_id: int
    source_path: Optional[str]
    page_number: int
    content_markdown: str
    created_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "DocumentMarkdownRecord":
        return cls(**row)  # type: ignore[arg-type]


@dataclass
class DocumentMarkdownCreate:
    """Fields accepted when inserting extracted document pages."""

    source_path: Optional[str]
    page_number: int
    content_markdown: str


class ApplicationCRUD:
    """Simple CRUD interface for the applications table."""

    def __init__(
        self,
        dsn: str,
        *,
        table_name: str = DEFAULT_TABLE_NAME,
        auto_initialize: bool = True,
    ) -> None:
        self._dsn = dsn
        self._table_name = table_name
        if auto_initialize:
            self.ensure_schema()

    def ensure_schema(self) -> None:
        """Create the applications table if it does not exist."""

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id SERIAL PRIMARY KEY,
                        full_name TEXT NOT NULL,
                        dob DATE,
                        nationality TEXT,
                        emirate TEXT,
                        emirates_id TEXT,
                        employment_status TEXT,
                        monthly_income_aed NUMERIC(12, 2),
                        credit_score INTEGER,
                        contact_pref TEXT,
                        best_time TEXT,
                        notes TEXT,
                        decision JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                ).format(table=self._table_identifier)
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index}
                    ON {table} (created_at DESC)
                    """
                ).format(
                    index=sql.Identifier(f"{self._table_name}_created_at_idx"),
                    table=self._table_identifier,
                )
            )
            self._ensure_documents_table(cur)

    def _ensure_documents_table(self, cur: psycopg.Cursor[Any]) -> None:
        documents_table = self._documents_table_identifier
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    application_id INTEGER NOT NULL REFERENCES {applications}(id) ON DELETE CASCADE,
                    source_path TEXT,
                    page_number INTEGER NOT NULL,
                    content_markdown TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            ).format(table=documents_table, applications=self._table_identifier)
        )
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON {table} (application_id, page_number)
                """
            ).format(
                index=sql.Identifier(f"{self._documents_table_name}_app_page_idx"),
                table=documents_table,
            )
        )

    def create(self, payload: ApplicationCreate) -> ApplicationRecord:
        """Insert a new application record and return the persisted row."""

        data = self._prepare_create_payload(payload)
        columns_sql = self._columns_sql(data.keys())
        placeholders = self._placeholders(len(data))

        query = sql.SQL(
            """
            INSERT INTO {table} ({columns})
            VALUES ({placeholders})
            RETURNING {returning}
            """
        ).format(
            table=self._table_identifier,
            columns=columns_sql,
            placeholders=placeholders,
            returning=self._returning_sql,
        )

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, list(data.values()))
            row = cur.fetchone()

        if row is None:
            raise RuntimeError("Failed to insert application")
        return ApplicationRecord.from_row(row)

    def get(self, application_id: int) -> Optional[ApplicationRecord]:
        """Retrieve a single application by identifier."""

        query = sql.SQL(
            """
            SELECT {returning}
            FROM {table}
            WHERE id = %s
            """
        ).format(returning=self._returning_sql, table=self._table_identifier)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (application_id,))
            row = cur.fetchone()

        return ApplicationRecord.from_row(row) if row else None

    def list(self, *, limit: int = 50, offset: int = 0) -> List[ApplicationRecord]:
        """Return a window of applications ordered by recency."""

        query = sql.SQL(
            """
            SELECT {returning}
            FROM {table}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """
        ).format(returning=self._returning_sql, table=self._table_identifier)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (limit, offset))
            rows = cur.fetchall()

        return [ApplicationRecord.from_row(row) for row in rows]

    def update(self, application_id: int, updates: Mapping[str, Any]) -> Optional[ApplicationRecord]:
        """Apply partial updates to an application and return the refreshed row."""

        data = self._prepare_update_payload(updates)
        if not data:
            return self.get(application_id)

        assignments = self._assignments_sql(data.keys())
        query = sql.SQL(
            """
            UPDATE {table}
            SET {assignments}, updated_at = NOW()
            WHERE id = %s
            RETURNING {returning}
            """
        ).format(
            table=self._table_identifier,
            assignments=assignments,
            returning=self._returning_sql,
        )

        values: List[Any] = list(data.values()) + [application_id]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, values)
            row = cur.fetchone()

        return ApplicationRecord.from_row(row) if row else None

    def delete(self, application_id: int) -> bool:
        """Remove an application from the table."""

        query = sql.SQL("DELETE FROM {table} WHERE id = %s").format(
            table=self._table_identifier
        )

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (application_id,))
            return cur.rowcount > 0

    def create_document_markdowns(
        self,
        application_id: int,
        documents: Iterable[DocumentMarkdownCreate],
    ) -> List[DocumentMarkdownRecord]:
        """Persist extracted document pages linked to an application."""

        entries = list(documents)
        if not entries:
            return []

        query = sql.SQL(
            """
            INSERT INTO {table} (application_id, source_path, page_number, content_markdown)
            VALUES (%s, %s, %s, %s)
            RETURNING {returning}
            """
        ).format(
            table=self._documents_table_identifier,
            returning=self._documents_returning_sql,
        )

        persisted: List[DocumentMarkdownRecord] = []
        with self._connect() as conn, conn.cursor() as cur:
            for entry in entries:
                cur.execute(
                    query,
                    (
                        application_id,
                        entry.source_path,
                        entry.page_number,
                        entry.content_markdown,
                    ),
                )
                row = cur.fetchone()
                if row:
                    persisted.append(DocumentMarkdownRecord.from_row(row))

        return persisted

    def _prepare_create_payload(self, payload: ApplicationCreate) -> Dict[str, Any]:
        values = asdict(payload)
        return self._normalize_payload(values)

    def _prepare_update_payload(self, updates: Mapping[str, Any]) -> Dict[str, Any]:
        filtered = {k: updates[k] for k in updates if k in self._writeable_columns}
        return self._normalize_payload(filtered)

    def _normalize_payload(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for column in self._writeable_columns:
            if column not in raw:
                continue

            value = raw[column]
            if column == "decision" and value is not None:
                normalized[column] = Json(value)
            else:
                normalized[column] = value

        return normalized

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self._dsn)
        conn.row_factory = dict_row
        return conn

    def _columns_sql(self, columns: Iterable[str]) -> sql.Composed:
        identifiers = [sql.Identifier(col) for col in columns]
        return sql.SQL(", ").join(identifiers)

    def _assignments_sql(self, columns: Iterable[str]) -> sql.Composed:
        assignments = [
            sql.Composed([sql.Identifier(col), sql.SQL(" = "), sql.Placeholder()])
            for col in columns
        ]
        return sql.SQL(", ").join(assignments)

    @staticmethod
    def _placeholders(count: int) -> sql.Composed:
        return sql.SQL(", ").join([sql.Placeholder()] * count)

    @property
    def _table_identifier(self) -> sql.Identifier:
        return sql.Identifier(self._table_name)

    @property
    def _returning_sql(self) -> sql.Composed:
        return self._columns_sql(self._select_columns)

    @property
    def _select_columns(self) -> List[str]:
        return [
            "id",
            "full_name",
            "dob",
            "nationality",
            "emirate",
            "emirates_id",
            "employment_status",
            "monthly_income_aed",
            "credit_score",
            "contact_pref",
            "best_time",
            "notes",
            "decision",
            "created_at",
            "updated_at",
        ]

    @property
    def _writeable_columns(self) -> List[str]:
        return [
            "full_name",
            "dob",
            "nationality",
            "emirate",
            "emirates_id",
            "employment_status",
            "monthly_income_aed",
            "credit_score",
            "contact_pref",
            "best_time",
            "notes",
            "decision",
        ]

    @property
    def _documents_table_name(self) -> str:
        return f"{self._table_name}{DOCUMENTS_TABLE_SUFFIX}"

    @property
    def _documents_table_identifier(self) -> sql.Identifier:
        return sql.Identifier(self._documents_table_name)

    @property
    def _documents_returning_sql(self) -> sql.Composed:
        return self._columns_sql(self._documents_select_columns)

    @property
    def _documents_select_columns(self) -> List[str]:
        return [
            "id",
            "application_id",
            "source_path",
            "page_number",
            "content_markdown",
            "created_at",
        ]
