import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import mlflow
import openai
from dotenv import load_dotenv

from db.models import Movie
from db.recommend import EmbeddingModel, ScoreMetricVersion, get_recommendations
from experiments.openai_models import OpenAIModel

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("openai_movie_recommendation")


GPT_MODEL = OpenAIModel.GPT_4o_MINI

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


# === Evaluation Prompt ===
def create_judge_prompt(user_prompt: str, movies: list[Movie]) -> str:
    movie_list = "\n".join(f"{i + 1}. {m.original_title}" for i, m in enumerate(movies))
    return f"""
    User asked: "{user_prompt}"

    These movies were recommended:

    {movie_list}

    Evaluate this list on:
    1. Relevance to the user prompt
    2. Novelty (not just obvious blockbusters)
    3. Diversity (in genre, tone, release years, etc.)

    Give a score from 1 to 10 with one decimal place for each category.
    Where 1 is poor and 10 is excellent.

    Respond in JSON like:
    {{
    "relevance": 8.0,
    "novelty": 6.5,
    "diversity": 5.5,
    "comment": "Varied and relevant mix with some fresh picks."
    }}
    """


def call_llm_judge(prompt_text: str, judge_model: OpenAIModel = GPT_MODEL) -> dict[str, Any]:
    try:
        client = openai.OpenAI()

        completion = client.chat.completions.create(
            model=judge_model.model_name,
            messages=[
                {
                    "role": "developer",
                    "content": (
                        "You are a seasoned movie recommendation expert."
                        "Your job is to understand nuanced preferences and provide highly relevant"
                        "engaging movie suggestions"
                        "You consider genre, tone, themes, cast, and storytelling style when making recommendations."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ],
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content

        if content is None:
            print("⚠️ OpenAI returned empty response")
            return {}

        parsed_content = json.loads(content.strip())

        if not isinstance(parsed_content, dict):
            print("⚠️ OpenAI response is not in a valid Response format")
            return {}

        return parsed_content

    except Exception as e:
        print("⚠️ Failed to parse LLM response:", e)
        return {}


def evaluate_embedding_score(
    embedding_model: EmbeddingModel,
    score_metric_version: ScoreMetricVersion,
    judge_model: OpenAIModel,
    prompt_name: str,
    prompt_text: str,
    prompt_evaluation_set: int = 10,
) -> dict[str, float]:
    """Evaluate a specific combination of embedding model and score metric."""
    try:
        # Get recommendations using the specified models
        movies = get_recommendations(
            prompt=prompt_text,
            per_page=prompt_evaluation_set,
            embedding_model=embedding_model,
            score_metric_version=score_metric_version,
        )

        if not movies:
            print(f"⚠️ No recommendations returned for {prompt_name}")
            return {}

        # Create judge prompt and get evaluation
        judge_prompt = create_judge_prompt(prompt_text, movies)
        evaluation = call_llm_judge(judge_prompt, judge_model)

        return evaluation

    except Exception as e:
        print(f"⚠️ Error evaluating {prompt_name}: {e}")
        return {}


def run_embedding_score_experiment_all(
    judge_model: OpenAIModel = GPT_MODEL,
    prompt_evaluation_set: int = 10,
) -> None:
    """Run experiments for all combinations of embedding models and score metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get all combinations
    embedding_models = list(EmbeddingModel)
    score_versions = list(ScoreMetricVersion)

    # Create list of all experiment tasks
    experiment_tasks = []
    for embedding_model in embedding_models:
        for score_version in score_versions:
            for prompt_name, prompt_text in PROMPTS.items():
                experiment_tasks.append((embedding_model, score_version, prompt_name, prompt_text))

    results = {}

    with ThreadPoolExecutor() as executor:
        future_to_params = {}
        for task in experiment_tasks:
            future = executor.submit(
                evaluate_embedding_score,
                task[0],  # embedding_model
                task[1],  # score_version
                judge_model,  # judge_model
                task[2],  # prompt_name
                task[3],  # prompt_text
                prompt_evaluation_set,
            )

            future_to_params[future] = (
                task[0],  # embedding_model
                task[1],  # score_version
                task[2],  # prompt_name
            )

        for future in future_to_params:
            embedding_model, score_version, prompt_name = future_to_params[future]
            try:
                result = future.result()
                print(f"✅ Completed task {embedding_model.name}, {score_version.name}, {prompt_name}: {result}")
                results[(embedding_model, score_version, prompt_name)] = result
            except Exception as e:
                print(f"⚠️ Error in task {embedding_model}, {score_version}, {prompt_name}: {e}")

    for embedding_model in embedding_models:
        for score_version in score_versions:
            metrics: dict[str, list[float]] = defaultdict(list)

            with mlflow.start_run(
                run_name=f"{embedding_model.name}_{score_version.name}_judge_{GPT_MODEL.name}_top_{prompt_evaluation_set}",
            ):
                for prompt_name, prompt_text in PROMPTS.items():
                    key = (embedding_model, score_version, prompt_name)
                    evaluation = results.get(key, {})

                    if not evaluation:
                        print(f"⚠️ No evaluation for {key}")
                        continue

                    with mlflow.start_run(
                        run_name=f"Prompt_{prompt_name}",
                        nested=True,
                    ):
                        mlflow.log_param("embedding_model", embedding_model.name)
                        mlflow.log_param("score_version", score_version.name)
                        mlflow.log_param("judge_model", judge_model.name)
                        mlflow.log_param("prompt_name", prompt_name)
                        mlflow.log_param("prompt_text", prompt_text)
                        mlflow.log_param("prompt_evaluation_set", prompt_evaluation_set)
                        mlflow.log_param("timestamp", timestamp)

                        prompt_average = 0.0
                        metric_counter = 0

                        for metric, value in evaluation.items():
                            if isinstance(value, int) or isinstance(value, float):
                                mlflow.log_metric(metric, value)
                                metrics[metric].append(value)

                                prompt_average += value
                                metric_counter += 1
                            elif isinstance(value, str):  # type: ignore
                                mlflow.log_text(
                                    json.dumps(value, indent=2),
                                    f"{metric}_{embedding_model.name}_{score_version.name}_{prompt_name}.json",
                                )

                        if prompt_average > 0:
                            prompt_average /= metric_counter
                            prompt_average = round(prompt_average, 2)
                            mlflow.log_metric("prompt_avg", prompt_average)
                            print(f"✅ Logged local average for {prompt_name}: {prompt_average}")

                        print(f"✅ Logged evaluation for {key}: {evaluation}")

                overall_average = 0.0
                metrics_counter = 0

                # Log average metrics across all prompts
                for metric, values in metrics.items():
                    if values:
                        overall_average += sum(values)
                        metrics_counter += len(values)

                        avg_metric = round(sum(values) / len(values), 2)
                        mlflow.log_metric(f"{metric}_avg", avg_metric)
                        print(f"✅ Logged overall {metric}: {avg_metric}")

                if metrics_counter > 0:
                    overall_average /= metrics_counter
                    overall_average = round(overall_average, 2)
                    mlflow.log_metric("overall_avg", overall_average)
                    print(f"✅ Logged overall average: {overall_average}")


def main() -> None:
    print("Running OpenAI embedding score evaluation experiment...")
    run_embedding_score_experiment_all(judge_model=GPT_MODEL, prompt_evaluation_set=5)
    run_embedding_score_experiment_all(judge_model=GPT_MODEL, prompt_evaluation_set=10)
    run_embedding_score_experiment_all(judge_model=GPT_MODEL, prompt_evaluation_set=20)
    print("Experiment completed!")


if __name__ == "__main__":
    main()
