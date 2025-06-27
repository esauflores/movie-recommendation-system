from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

# from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from db.recommend import (
    EmbeddingModel,
    ScoreMetricVersion,
    get_movie_by_id,
    get_recommendations,
    get_similar_movies,
)

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")
# app.mount("/static", StaticFiles(directory="webapp/static"), name="static")


EMBEDDING_MODEL = EmbeddingModel.LARGE_3
SCORE_METRIC_VERSION = ScoreMetricVersion.V3


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "movies": [], "prompt": ""})


@app.post("/", response_class=HTMLResponse)
async def recommend(request: Request, prompt: str = Form(...)) -> HTMLResponse:
    movies = get_recommendations(
        prompt,
        page=1,
        per_page=8,
        embedding_model=EMBEDDING_MODEL,
        score_metric_version=SCORE_METRIC_VERSION,
    )
    return templates.TemplateResponse("index.html", {"request": request, "movies": movies, "prompt": prompt})


@app.get("/movie/{movie_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, movie_id: int) -> HTMLResponse:
    # Get the specific movie
    movie = get_movie_by_id(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Get similar movies (optional - you can implement this later)
    similar_movies = get_similar_movies(
        movie_id,
        page=1,
        per_page=8,
        embedding_model=EMBEDDING_MODEL,
        score_metric_version=SCORE_METRIC_VERSION,
    )

    return templates.TemplateResponse(
        "movie_detail.html",
        {"request": request, "movie": movie, "similar_movies": similar_movies},
    )


@app.get("/api/recommendations", response_class=JSONResponse)
async def load_more_recommendations(prompt: str, page: int = 2) -> JSONResponse:
    """API endpoint to load more recommendations via AJAX"""
    movies = get_recommendations(
        prompt,
        page=page,
        per_page=8,
        embedding_model=EMBEDDING_MODEL,
        score_metric_version=SCORE_METRIC_VERSION,
    )

    # Convert movies to dictionaries for JSON response
    movies_data = []
    for movie in movies:
        movies_data.append(
            {
                "movie_id": movie.movie_id,
                "english_title": movie.english_title,
                "poster_path": movie.poster_path,
                "vote_average": movie.vote_average,
                "vote_count": movie.vote_count,
            }
        )

    return JSONResponse(content={"movies": movies_data})
