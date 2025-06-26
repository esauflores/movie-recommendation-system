.PHONY: help clean setup data-pipeline load-data preprocess-data init-db load-movies update-posters remove-no-posters generate-embeddings run-experiments

# Default Python interpreter
PYTHON := python

# Help target
help:
	@echo "Movie Recommendation System - Data Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  setup                - Set up the environment"
	@echo "  load-data            - Load raw data"
	@echo "  preprocess-data      - Preprocess the loaded data"
	@echo "  init-db              - Initialize the database"
	@echo "  load-movies          - Load movies into the database"
	@echo "  update-posters       - Update movie posters in the database"
	@echo "  remove-no-posters    - Remove movies without posters"
	@echo "  generate-embeddings  - Generate OpenAI embeddings for movies"
	@echo "  run-experiments      - Run MLflow experiments"
	@echo "  preprocess-data-pipeline - Run the preprocessing data pipeline"
	@echo "  db-data-pipeline     - Run the database data pipeline"
	@echo "  data-pipeline        - Run the full data pipeline"
	@echo "  dev-setup            - Set up the development environment"
	@echo "  mlflow-server        - Start the MLflow server"
	@echo "  webapp               - Start the web application"
	@echo "  db-reset             - Reset the database"
	@echo "  clean                - Clean up generated files"
	@echo "  test                 - Run tests"
	@echo "  check-data           - Check data files"
	@echo "  docker-build         - Build Docker containers"
	@echo "  docker-up            - Start Docker containers"
	@echo "  docker-down          - Stop Docker containers"
	@echo "  logs                 - Show recent logs"
	@echo "  status               - Check the status of the pipeline"

# Setup environment
setup:
	@echo "🔧 Setting up environment..."
	pip install -r requirements.txt || uv sync

# Step 1: Load raw data
load-data:
	@echo "📥 Step 1: Loading raw data..."
	$(PYTHON) data/1-load_data.py

# Step 2: Preprocess data
preprocess-data: load-data
	@echo "🔄 Step 2: Preprocessing data..."
	$(PYTHON) data/2-preprocess_data.py

# Step 3: Initialize database
init-db: preprocess-data
	@echo "🗄️ Step 3: Initializing database..."
	$(PYTHON) db/init_db.py

# Step 4: Load movies into database
load-movies: init-db
	@echo "🎬 Step 4: Loading movies into database..."
	$(PYTHON) db/1-load_movies.py

# Step 5: Update movie posters
update-posters: load-movies
	@echo "🖼️ Step 5: Updating movie posters..."
	$(PYTHON) db/2-update_movies_poster.py

# Step 6: Remove movies without posters
remove-no-posters: update-posters
	@echo "🧹 Step 6: Removing movies without posters..."
	$(PYTHON) db/3-remove_movies_no_poster.py

# Step 7: Generate embeddings
generate-embeddings: remove-no-posters
	@echo "🤖 Step 7: Generating OpenAI embeddings..."
	$(PYTHON) db/4-generate_embeddings.py

# Step 8: Run experiments
run-experiments: generate-embeddings
	@echo "🧪 Step 8: Running MLflow experiments..."
	$(PYTHON) mlflow_experiments/exp_openai.py

# Preprocessing data pipeline
preprocess-data-pipeline: setup load-data preprocess-data	
	@echo "✅ Preprocessing data pipeline completed!"
	@echo "Run 'make db-data-pipeline;' to continue the full pipeline"
	@echor "Or run 'make full-data-pipeline' to start again the full pipeline"

# Database data pipeline
db-data-pipeline: init-db load-movies update-posters remove-no-posters generate-embeddings
	@echo "✅ Data pipeline completed successfully!"


# Full data pipeline
full-data-pipeline: setup load-data preprocess-data init-db load-movies update-posters remove-no-posters generate-embeddings
	@echo "✅ Full data pipeline completed successfully!"

# Development targets
dev-setup: setup
	@echo "🛠️ Setting up development environment..."
	pip install -e . || echo "No setup.py found, skipping editable install"

# Start MLflow server
mlflow-server:
	@echo "🚀 Starting MLflow server..."
	mlflow server --host 0.0.0.0 --port 5000

# Start webapp
webapp: generate-embeddings
	@echo "🌐 Starting web application..."
	$(PYTHON) webapp/main.py

# Database operations
db-reset:
	@echo "🔄 Resetting database..."
	$(PYTHON) db/init_db.py

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	@echo "✅ Cleanup completed!"

# Test targets
test:
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v

# Check data files
check-data:
	@echo "📊 Checking data files..."
	@echo "Raw data:"
	@ls -la data/raw/ 2>/dev/null || echo "  No raw data found"
	@echo "Preprocessed data:"
	@ls -la data/preprocessed/ 2>/dev/null || echo "  No preprocessed data found"

# Docker targets (if using docker)
docker-build:
	@echo "🐳 Building Docker containers..."
	docker-compose build

docker-up:
	@echo "🐳 Starting Docker containers..."
	docker-compose up -d

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down

# Monitoring and logs
logs:
	@echo "📋 Showing recent logs..."
	tail -f *.log 2>/dev/null || echo "No log files found"

# Status check
status:
	@echo "📊 Pipeline Status Check"
	@echo "======================="
	@echo "Raw data files:"
	@ls data/raw/ 2>/dev/null | wc -l | awk '{print "  " $$1 " files"}'
	@echo "Preprocessed data files:"
	@ls data/preprocessed/ 2>/dev/null | wc -l | awk '{print "  " $$1 " files"}'
	@echo "Database status:"
	@$(PYTHON) -c "from db.database import SessionLocal; session = SessionLocal(); from db.models import Movie; count = session.query(Movie).count(); print(f'  {count} movies in database'); session.close()" 2>/dev/null || echo "  Database not accessible"
