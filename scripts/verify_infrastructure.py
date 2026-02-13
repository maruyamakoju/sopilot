"""Verify VIGIL-RAG infrastructure is running correctly.

This script checks:
1. Postgres connection and tables
2. Qdrant connection and health
3. Alembic migration status

Usage:
    python scripts/verify_infrastructure.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_postgres():
    """Check Postgres connection and tables."""
    print("🔍 Checking Postgres...")

    try:
        import psycopg2
        from psycopg2 import sql  # noqa: F401
    except ImportError:
        print("❌ psycopg2 not installed. Run: pip install -e '.[vigil]'")
        return False

    db_url = os.getenv("VIGIL_POSTGRES_URL", "postgresql://vigil_user:vigil_dev_password@localhost:5432/vigil")

    try:
        # Parse URL
        if db_url.startswith("postgresql://"):
            # Extract components
            parts = db_url.replace("postgresql://", "").split("@")
            user_pass = parts[0].split(":")
            host_db = parts[1].split("/")
            host_port = host_db[0].split(":")

            conn = psycopg2.connect(
                dbname=host_db[1] if len(host_db) > 1 else "vigil",
                user=user_pass[0],
                password=user_pass[1],
                host=host_port[0],
                port=host_port[1] if len(host_port) > 1 else "5432",
            )
        else:
            conn = psycopg2.connect(db_url)

        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = {
            "videos",
            "clips",
            "embeddings",
            "events",
            "queries",
            "ingest_jobs",
            "score_jobs",
            "training_jobs",
            "alembic_version",
        }

        if not expected_tables.issubset(set(tables)):
            missing = expected_tables - set(tables)
            print(f"❌ Missing tables: {missing}")
            print("   Run: alembic upgrade head")
            return False

        print(f"✅ Postgres connected ({len(tables)} tables)")
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Postgres connection failed: {e}")
        print("   Check: docker-compose ps postgres")
        print(f"   URL: {db_url}")
        return False


def check_qdrant():
    """Check Qdrant connection."""
    print("🔍 Checking Qdrant...")

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("❌ qdrant-client not installed. Run: pip install -e '.[vigil]'")
        return False

    host = os.getenv("VIGIL_QDRANT_HOST", "localhost")
    port = int(os.getenv("VIGIL_QDRANT_PORT", "6333"))

    try:
        client = QdrantClient(host=host, port=port)
        collections = client.get_collections()

        print(f"✅ Qdrant connected ({len(collections.collections)} collections)")

        # List collections if any
        if collections.collections:
            for col in collections.collections:
                print(f"   - {col.name}: {col.vectors_count} vectors")

        return True

    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   Check: docker-compose ps qdrant")
        print(f"   URL: http://{host}:{port}")
        return False


def check_alembic():
    """Check Alembic migration status."""
    print("🔍 Checking Alembic migrations...")

    try:
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from sqlalchemy import create_engine
    except ImportError:
        print("❌ alembic not installed. Run: pip install -e '.[vigil]'")
        return False

    db_url = os.getenv("VIGIL_POSTGRES_URL", "postgresql://vigil_user:vigil_dev_password@localhost:5432/vigil")

    try:
        # Load alembic config
        alembic_cfg = Config("alembic.ini")
        script_dir = ScriptDirectory.from_config(alembic_cfg)

        # Get current revision from DB
        engine = create_engine(db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current_rev = context.get_current_revision()

        # Get head revision from scripts
        head_rev = script_dir.get_current_head()

        if current_rev == head_rev:
            print(f"✅ Migrations up-to-date ({current_rev})")
            return True
        else:
            print("⚠️  Migration mismatch:")
            print(f"   Current: {current_rev}")
            print(f"   Head: {head_rev}")
            print("   Run: alembic upgrade head")
            return False

    except Exception as e:
        print(f"❌ Alembic check failed: {e}")
        return False


def main():
    """Run all infrastructure checks."""
    print("=" * 60)
    print("VIGIL-RAG Infrastructure Verification")
    print("=" * 60)
    print()

    results = {
        "Postgres": check_postgres(),
        "Qdrant": check_qdrant(),
        "Alembic": check_alembic(),
    }

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    for service, status in results.items():
        status_symbol = "✅" if status else "❌"
        print(f"{status_symbol} {service}")

    all_good = all(results.values())

    if all_good:
        print()
        print("🎉 All infrastructure services are ready!")
        print()
        print("Next steps:")
        print("  1. Create collections: python scripts/create_qdrant_collections.py")
        print("  2. Run E2E smoke: python scripts/vigil_smoke_e2e.py --video test.mp4")
        return 0
    else:
        print()
        print("⚠️  Some services are not ready. See errors above.")
        print()
        print("Quick fix:")
        print("  docker-compose up -d postgres qdrant")
        print("  alembic upgrade head")
        return 1


if __name__ == "__main__":
    sys.exit(main())
