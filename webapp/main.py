from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from db.recommend import (
    get_recommendations,
    get_movie_by_id,
    get_similar_movies,
)  # your logic here

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")


EMBEDDING_MODEL = "text-embedding-3-large"  # Default embedding model


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "movies": [], "prompt": ""}
    )


@app.post("/", response_class=HTMLResponse)
async def recommend(request: Request, prompt: str = Form(...)):
    movies = get_recommendations(
        prompt, per_page=8, embedding_model=EMBEDDING_MODEL
    )
    return templates.TemplateResponse(
        "index.html", {"request": request, "movies": movies, "prompt": prompt}
    )


@app.get("/movie/{movie_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, movie_id: int):
    # Get the specific movie
    movie = get_movie_by_id(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Get similar movies (optional - you can implement this later)
    similar_movies = get_similar_movies(
        movie_id, limit=8, embedding_model=EMBEDDING_MODEL
    )

    return templates.TemplateResponse(
        "movie_detail.html",
        {"request": request, "movie": movie, "similar_movies": similar_movies},
    )
