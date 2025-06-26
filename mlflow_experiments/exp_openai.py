from builtins import dict
import openai
import json
from db.recommend import get_recommendations
from db.models import Movie
from dotenv import load_dotenv
import mlflow
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("openai_movie_recommendation")


GPT_MODEL = "gpt-4o-mini"  # Use a more capable model for nuanced evaluation

# === Experiment Configuration ===
PROMPTS = {
    "superhero_movies": "Recommend some great superhero movies.",
    "romcom_classics": "What are some iconic romantic comedies?",
    "psychological_horror": "Show me horror movies with deep psychological themes.",
    "dystopian_scifi": "I'm looking for sci-fi movies set in dystopian futures.",
    "female_led_action": "Can you suggest action movies with strong female leads?",
    "animated_family_favorites": "Recommend popular animated movies suitable for families.",
    "mind_bending_thrillers": "Give me thrillers with unexpected twists and turns.",
    "historical_dramas": "What are some compelling historical drama films?",
    "crime_mystery": "Suggest crime or mystery movies with clever plots.",
    "coming_of_age": "What are some coming-of-age movies worth watching?",
}


EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
]


# === Evaluation Prompt ===
def create_judge_prompt(user_prompt: str, movies: list[Movie]) -> str:
    movie_list = "\n".join(
        f"{i + 1}. {m.original_title}" for i, m in enumerate(movies)
    )
    return f"""
    User asked: "{user_prompt}"

    These movies were recommended:

    {movie_list}

    Evaluate this list on:
    1. Relevance to the user prompt
    2. Novelty (not just obvious blockbusters)
    3. Diversity (in genre, tone, release years, etc.)

    Give a score from 1 to 10 for each and a short explanation.

    Respond in JSON like:
    {{
    "relevance": 8,
    "novelty": 6,
    "diversity": 7,
    "comment": "Varied and relevant mix with some fresh picks."
    }}
    """


def call_llm_judge(prompt_text: str) -> dict:
    try:
        client = openai.OpenAI()

        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "developer",
                    "content": (
                        "You are a seasoned movie recommendation expert. "
                        "Your job is to understand nuanced preferences and provide highly relevant, engaging movie suggestions. "
                        "You consider genre, tone, themes, cast, and storytelling style when making recommendations. "
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ],
            response_format={"type": "json_object"},
        )  # pyrefly: ignore

        content = completion.choices[0].message.content
        if content is None:
            print("⚠️ OpenAI returned empty response")
            return {}

        return json.loads(content.strip())

    except Exception as e:
        print("⚠️ Failed to parse LLM response:", e)
        return {}


def evaluate_and_log_prompt(
    prompt_name: str,
    prompt_text: str,
    embedding_model: str,
    number_of_movies: int = 10,
) -> tuple[str, str, dict]:
    """Evaluate a prompt and return the results for logging."""
    print(f"🎬 Evaluating '{prompt_name}' with {embedding_model}...")

    # Step 1: Get recommendations
    recommendations = get_recommendations(
        prompt=prompt_text,
        per_page=number_of_movies,
        embedding_model=embedding_model,
    )

    if not recommendations:
        print(f"⚠️ No recommendations found for prompt '{prompt_name}'")
        return prompt_name, embedding_model, {}

    # Step 2: Generate and send judge prompt to LLM
    judge_prompt = create_judge_prompt(prompt_text, recommendations)
    result = call_llm_judge(judge_prompt)

    if not result:
        print(f"⚠️ Failed to get judge response for prompt '{prompt_name}'")
        return prompt_name, embedding_model, {}

    print(f"✅ {prompt_name} ({embedding_model}): {result}")
    return prompt_name, embedding_model, result


def run_concurrent_evaluation() -> dict[str, dict[str, dict]]:
    """Run all evaluations concurrently and return results in nested dictionary."""
    metrics: dict[str, dict[str, dict]] = {}

    # Create list of all evaluation tasks
    tasks = []
    for embedding_model in EMBEDDING_MODELS:
        for prompt_name, prompt_text in PROMPTS.items():
            tasks.append((prompt_name, prompt_text, embedding_model))

    # Run evaluations concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                evaluate_and_log_prompt,
                prompt_name,
                prompt_text,
                embedding_model,
            )
            for prompt_name, prompt_text, embedding_model in tasks
        ]

        # Collect results as they complete
        for future in futures:
            prompt_name, embedding_model, result = future.result()
            if result:  # Only store non-empty results
                if embedding_model not in metrics:
                    metrics[embedding_model] = {}
                metrics[embedding_model][prompt_name] = result

    return metrics


def print_summary(metrics: dict[str, dict[str, dict]]) -> None:
    """Print a summary of all evaluation results."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    for embedding_model, prompts in metrics.items():
        if not prompts:
            continue

        print(f"\n🤖 Model: {embedding_model}")
        print("-" * 40)

        # Calculate averages for this model
        all_relevance = [
            result.get("relevance", 0) for result in prompts.values()
        ]
        all_novelty = [result.get("novelty", 0) for result in prompts.values()]
        all_diversity = [
            result.get("diversity", 0) for result in prompts.values()
        ]

        if all_relevance:
            avg_relevance = sum(all_relevance) / len(all_relevance)
            avg_novelty = sum(all_novelty) / len(all_novelty)
            avg_diversity = sum(all_diversity) / len(all_diversity)

            print("📈 Average Scores:")
            print(f"   Relevance: {avg_relevance:.1f}")
            print(f"   Novelty: {avg_novelty:.1f}")
            print(f"   Diversity: {avg_diversity:.1f}")
            print(
                f"  Overall: {(avg_relevance + avg_novelty + avg_diversity) / 3:.1f}"
            )

        print(f"✅ Successful evaluations: {len(prompts)}/{len(PROMPTS)}")


def main() -> None:
    """Run all evaluations concurrently and display results."""
    print("🎬 Starting movie recommendation evaluation...")
    print(
        f"📋 Testing {len(PROMPTS)} prompts across {len(EMBEDDING_MODELS)} models"
    )

    # Run all evaluations concurrently
    metrics = run_concurrent_evaluation()

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(
        run_name=f"openai_embedding_evaluation_{current_date}"
    ):
        mlflow.log_param("judge_model", GPT_MODEL)

        for embedding_model, prompts in metrics.items():
            with mlflow.start_run(
                run_name=f"Model: {embedding_model}",
                nested=True,
            ):
                mlflow.log_param("embedding_model", embedding_model)
                mlflow.log_param("number_of_prompts", len(prompts))
                for prompt_name, result in prompts.items():
                    with mlflow.start_run(
                        run_name=f"Prompt: {prompt_name}",
                        nested=True,
                    ):
                        mlflow.log_param("embedding_model", embedding_model)
                        mlflow.log_param("prompt_name", prompt_name)
                        mlflow.log_metric(
                            "relevance", result.get("relevance", 0)
                        )
                        mlflow.log_metric("novelty", result.get("novelty", 0))
                        mlflow.log_metric(
                            "diversity", result.get("diversity", 0)
                        )
                        mlflow.log_text(
                            text=json.dumps(result, indent=2),
                            artifact_file=f"{prompt_name}_{embedding_model}_{current_date}.json",
                        )

                all_relevance = [
                    result.get("relevance", 0) for result in prompts.values()
                ]
                all_novelty = [
                    result.get("novelty", 0) for result in prompts.values()
                ]
                all_diversity = [
                    result.get("diversity", 0) for result in prompts.values()
                ]

                avg_relevance = round(
                    sum(all_relevance) / len(all_relevance), 2
                )
                avg_novelty = round(sum(all_novelty) / len(all_novelty), 2)
                avg_diversity = round(
                    sum(all_diversity) / len(all_diversity), 2
                )

                mlflow.log_metric("avg_relevance", avg_relevance)
                mlflow.log_metric("avg_novelty", avg_novelty)
                mlflow.log_metric("avg_diversity", avg_diversity)
                mlflow.log_metric(
                    "overall_score",
                    (avg_relevance + avg_novelty + avg_diversity) / 3,
                )

    print("✅ All evaluations completed and logged to MLflow.")


if __name__ == "__main__":
    main()
