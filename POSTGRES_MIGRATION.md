# 🐘 Migrating from SQLite to PostgreSQL

This guide helps you switch your SolShield production environment from SQLite to PostgreSQL.

## 1. Prerequisites
- **PostgreSQL Installed**: Ensure PostgreSQL is installed and running.
- **Database Created**: Create a new empty database (e.g., `solshield_db`).

## 2. Install Dependencies
Install the PostgreSQL adapter for Python:

```bash
pip install psycopg2-binary
```

Add it to your `requirements.txt`.

## 3. Update Environment Variables
Modify your `.env` file to point to PostgreSQL instead of SQLite.

**Current (SQLite):**
```env
DATABASE_URL=sqlite:///./smart_contract_analyzer.db
```

**New (PostgreSQL):**
```env
# Format: postgresql://username:password@host:port/database_name
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/solshield_db
```

## 4. Automatic Migration
The application uses SQLAlchemy which will automatically create tables on startup if they don't exist.

1. Stop the backend server.
2. Update the `DATABASE_URL`.
3. Start the backend server: `python -m api.main`

The logs should show:
```
INFO:     Database tables created/verified
```

## 5. Data Migration (Optional)
 If you need to preserve data from SQLite, use a tool like `pgloader` or export/import via CSV.

```bash
# Example using pgloader
pgloader sqlite://./smart_contract_analyzer.db postgresql://user:pass@localhost/solshield_db
```

## 6. Verification
Check if tables are created using `psql`:
```sql
\c solshield_db
\dt
```
You should see tables like `users`, `contracts`, `vulnerabilities`, `analysis_results`.
