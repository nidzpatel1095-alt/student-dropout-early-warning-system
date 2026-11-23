# ==============================================================================
# PYTEST CONFTEST: tests/conftest.py
# Configures session-wide test isolation using a temporary database.
# Ensures database/dropout_system.db is NEVER accessed or modified by tests.
# ==============================================================================

import os
import sys
import pytest

# Root directory path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Real database path for verification
REAL_DB_PATH = os.path.join(PROJECT_ROOT, "database", "dropout_system.db")

# Record real DB state before any tests run
REAL_DB_EXISTS = os.path.exists(REAL_DB_PATH)
REAL_DB_MTIME = os.path.getmtime(REAL_DB_PATH) if REAL_DB_EXISTS else None
REAL_DB_SIZE = os.path.getsize(REAL_DB_PATH) if REAL_DB_EXISTS else None


@pytest.fixture(scope="session", autouse=True)
def isolated_test_db(tmp_path_factory):
    """
    Session-scoped fixture that redirects DROPOUT_DB_PATH to a temporary pytest path
    BEFORE app or database modules are imported.
    """
    temp_dir = tmp_path_factory.mktemp("db_temp")
    temp_db_path = str(temp_dir / "test_isolated.db")
    
    # Set environment variable override
    os.environ["DROPOUT_DB_PATH"] = temp_db_path
    
    # Import database init and load functions after setting env var
    from database.init_db import init_db
    from scripts.load_sample_students import load_sample_students
    
    init_db(reset=True)
    load_sample_students()
    
    yield temp_db_path


@pytest.fixture
def client(isolated_test_db):
    """Flask test client fixture using the isolated temporary database."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
