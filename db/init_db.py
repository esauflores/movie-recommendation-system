from database import engine, Base
import models  # noqa: F401


def init_db() -> None:
    Base.metadata.create_all(engine)
    print("Database tables created.")


if __name__ == "__main__":
    init_db()
