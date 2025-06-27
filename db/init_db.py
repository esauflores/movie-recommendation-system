import db.models  # noqa
from db.database import Base, engine


def init_db() -> None:
    Base.metadata.create_all(engine)
    print("Database tables created.")


def delete_db() -> None:
    Base.metadata.drop_all(engine)
    print("Database tables deleted.")


if __name__ == "__main__":
    # delete_db()
    init_db()
