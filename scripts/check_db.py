"""
Simple DB check script: reads DATABASE_URL from .env (if present) and lists tables.
Run: python scripts/check_db.py
"""
import os
from sqlalchemy import create_engine, inspect
from pathlib import Path

# Try to read .env if present
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print('DATABASE_URL not set in environment or .env')
    raise SystemExit(1)

print('Using DATABASE_URL:', DATABASE_URL)

try:
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print('Tables found:', tables)
except Exception as e:
    print('Failed to connect or list tables:', e)
    raise
