# 🎬 Movie Recommendation System - MLOps & Modern AI Engineering

Welcome to the **Movie Recommendation System**! This repository provides a **comprehensive, end-to-end ML-powered recommendation system** using the TMDB 5000 movies dataset. It's designed to demonstrate advanced MLOps patterns, vector databases, and modern AI engineering practices in a real-world application.

This project demonstrates advanced ML engineering practices including:

- **🤖 AI-Powered Recommendations**: Using OpenAI embeddings for semantic movie similarity
- **🗄️ Vector Database**: PostgreSQL with pgvector for efficient similarity search
- **🌐 Modern Web API**: FastAPI with async endpoints and interactive documentation
- **� Experiment Tracking**: MLflow for model versioning and experiment management
- **� Containerization**: Docker Compose for development and deployment
- **📈 Data Versioning**: DVC for tracking datasets and model artifacts
- **✨ Code Quality**: Automated linting, formatting, and type checking
- **🧪 Testing**: Comprehensive testing with Pytest
- **🔄 CI/CD**: Automated validation and deployment pipelines
- **📦 Dependency Management**: Modern Python tooling with `uv`

## 🎯 Project Goal

Build a production-ready movie recommendation system that provides personalized movie suggestions based on user preferences. The system uses advanced embedding techniques to understand movie content semantically and delivers recommendations through a modern web interface. The focus is on implementing MLOps best practices including experiment tracking, model versioning, data lineage, and automated deployment.

## 📂 Project Structure

```text
movie-recommendation-system/
├── data/                         # Data pipeline and datasets
│   ├── __init__.py
│   ├── 1-load_data.py            # Download and initial data loading from TMDB
│   ├── 2-preprocess_data.py      # Data cleaning and feature engineering
│   └── 3-update_movie_data.py    # Data updates and maintenance
├── db/                           # Database and recommendation engine
│   ├── __init__.py
│   ├── database.py               # SQLAlchemy database configuration
│   ├── models.py                 # Database models (Movie, Embeddings)
│   ├── init_db.py                # Database initialization
│   ├── 1-load_movies.py          # Load movies into PostgreSQL
│   ├── 2-generate_embeddings.py  # Generate OpenAI embeddings
│   └── recommend.py              # Recommendation algorithms
├── webapp/                       # FastAPI web application
│   ├── __init__.py
│   ├── main.py                   # FastAPI application and routes
│   ├── templates/                # Jinja2 HTML templates
│   │   ├── index.html
│   │   └── movie_detail.html
│   └── static/                   # CSS, JS, and other static assets
├── experiments/                  # ML experimentation and model development
│   ├── __init__.py
│   ├── experiment_openai.py      # OpenAI embedding experiments
│   └── openai_models.py          # OpenAI model configurations
├── docker/                       # Docker configurations
│   └── mlflow/
│       ├── Dockerfile
│       └── requirements.txt
├── tests/                        # Test suites
│   └── test1.py
├── docker-compose.yml            # Multi-container development environment
├── pyproject.toml                # Project dependencies and tool configuration
├── uv.lock                       # Locked dependency versions
├── Makefile                      # Development automation commands
└── README.md                     # This file
```

**Key Architecture Components:**

- **`data/` Directory:** Contains the ETL pipeline for TMDB movie data. Raw data is tracked with DVC for versioning, while processed data feeds into the database.
- **`db/` Directory:** Houses the core recommendation engine using PostgreSQL with pgvector for efficient similarity search. Includes database models, embedding generation, and recommendation algorithms.
- **`webapp/` Directory:** FastAPI-based web application providing both API endpoints and a web interface for movie recommendations.
- **`experiments/` Directory:** ML experimentation workspace for testing different embedding models, recommendation algorithms, and evaluation metrics.
- **`docker/` Directory:** Container configurations for the complete development and production environment, including MLflow tracking server.
- **`pyproject.toml`:** Central configuration for dependencies (FastAPI, SQLAlchemy, OpenAI, pgvector) and development tools.
- **`docker-compose.yml`:** Orchestrates PostgreSQL, MLflow, and application services for seamless development.

## 🛠️ Prerequisites

Ensure you have the following installed on your system:

- **Python 3.11+**: We recommend using the version specified in the `.python-version` file.
- **`uv`**: A fast Python package installer and project manager. See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- **Docker & Docker Compose**: For running PostgreSQL, MLflow, and other services.
- **Git**: For version control and DVC integration.
- **OpenAI API Key**: For generating movie embeddings (sign up at [OpenAI](https://platform.openai.com/)).
- **Make**: (Optional, but highly recommended for convenience).

## 🚀 Quick Start: Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd movie-recommendation-system
   ```

2. **Set Up Environment Variables:**

   Create a `.env` file in the project root:

   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and database credentials
   ```

3. **Set Up Python Environment:**

   ```bash
   # Create a virtual environment using the project's Python version
   uv venv .venv --python 3.11

   # Activate the virtual environment:
   # On macOS and Linux:
   source .venv/bin/activate
   # On Windows (PowerShell):
   # .\.venv\Scripts\Activate.ps1
   ```

4. **Install Dependencies:**

   ```bash
   # Install project dependencies, including development tools:
   uv sync --dev

   # Initialize Pre-commit Hooks (for code quality)
   uv run pre-commit install
   ```

5. **Start Infrastructure Services:**

   ```bash
   # Start PostgreSQL and MLflow using Docker Compose
   docker-compose up -d postgres-db minio mlflow
   ```

6. **Initialize the Database:**

   ```bash
   # Set up database schema and load initial data
   make setup-db
   ```

You're now ready to start!

# ▶️ Running the Movie Recommendation System

The system consists of data loading, embedding generation, and a web application for serving recommendations.

## 🚀 Quick Start - Run the Web Application

**Option 1: Using Docker Compose (Recommended)**

```bash
# Start all services (database, MLflow, web app)
docker-compose up

# Access the application at http://localhost:8000
```

**Option 2: Local Development**

```bash
# Ensure your virtual environment is activated and services are running
source .venv/bin/activate
docker-compose up -d postgres-db minio mlflow

# Run the FastAPI application
uv run python -m webapp.main
```

## 📊 Data Pipeline

Run the data pipeline to load and process movie data:

```bash
# Load raw data from TMDB
python data/1-load_data.py

# Preprocess and clean the data
python data/2-preprocess_data.py

# Load movies into PostgreSQL
python db/1-load_movies.py

# Generate OpenAI embeddings (requires API key)
python db/2-generate_embeddings.py
```

Or run the entire pipeline:

```bash
make run-pipeline
```

## � API Endpoints

The FastAPI application provides several endpoints:

- `GET /`: Web interface for movie recommendations
- `POST /`: Submit a prompt for movie recommendations
- `GET /movie/{movie_id}`: Get detailed information about a specific movie
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation

## 🧪 Experimentation

Use MLflow to track experiments:

```bash
# Access MLflow UI at http://localhost:5000
# Run experiments
python experiments/experiment_openai.py

# Compare different embedding models and recommendation algorithms
```

## 💻 Development Workflow & Tools

This project uses modern development tools for maintaining high code quality:

- **Makefile**: Your primary interface for common tasks. Run `make help` to see all available commands:

  ```bash
  make help
  ```

  This will list targets like `lint`, `format-check`, `test`, `setup-db`, `run-pipeline`, etc.

- **Code Formatting (Ruff)**:

  - Check formatting: `make format-check`
  - Auto-format: `make format`

- **Linting (Ruff)**:

  - Check code quality: `make lint`
  - Auto-fix issues: `make lint-fix`

- **Type Checking (MyPy)**:

  - Verify type annotations: `make mypy`

- **Testing (Pytest)**:

  - Run tests: `make test`
  - View coverage: Open `htmlcov/index.html` in your browser

- **Database Management**:

  - Initialize database: `make setup-db`
  - Reset database: `make reset-db`

- **Docker Services**:

  - Start services: `make up`
  - Stop services: `make down`
  - View logs: `make logs`

- **Pre-commit Hooks**: Automatically run code quality checks on `git commit`. If issues are found, the commit is blocked until they're fixed.

## 🏗️ Architecture Overview

### Tech Stack

- **Backend**: FastAPI (async Python web framework)
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: OpenAI text-embedding-ada-002
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Containerization**: Docker & Docker Compose
- **Code Quality**: Ruff (linting & formatting), MyPy (type checking)

### Recommendation Algorithm

1. **Content-Based Filtering**: Uses movie metadata (genres, overview, keywords) to generate embeddings
2. **Semantic Search**: Leverages OpenAI embeddings for understanding user preferences
3. **Vector Similarity**: Uses cosine similarity in pgvector for efficient similarity search
4. **Hybrid Scoring**: Combines content similarity with popularity and rating metrics

---

**Ready to build intelligent movie recommendations? Let's dive in! 🎬🚀**
