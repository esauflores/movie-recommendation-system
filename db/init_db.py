from db.database import engine, Base
import db.models  # noqa


def init_db() -> None:
    Base.metadata.create_all(engine)
    print("Database tables created.")


if __name__ == "__main__":
    init_db()
