"""Utility functions for DeepScaler.

This module contains various utility functions for making API calls to LLMs,
implementing RAG functionality, and managing network ports.
"""

import time
from typing import List, Union

# import vertexai
# from vertexai.generative_models import (
#     GenerationConfig,
#     GenerativeModel,
#     HarmBlockThreshold,
#     HarmCategory
# )
# from google.cloud.aiplatform_v1beta1.types.content import SafetySetting

import openai

import torch

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer, util = None, None

from deepscaler.globals import GCP_LOCATION, GCP_PROJECT_ID, GEMINI_MODEL, OAI_RM_MODEL


def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = OAI_RM_MODEL,
    retry_count: int = 1000000000,
) -> Union[str, List[str]]:
    """Call OpenAI API with retry logic.

    Args:
        prompt: The text prompt to send to the model
        system_prompt: System instruction for the model
        n: Number of completions to generate
        temperature: Sampling temperature
        model_id: OpenAI model ID to use
        retry_count: Number of retries on rate limit errors

    Returns:
        Generated text(s) from the model
    """
    client = openai.OpenAI()
    backoff = 1
    retry_count = int(retry_count)

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as exc:
            if "429" in str(exc):
                print("Retry due to rate limit: ", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            print("Exception: ", exc)
            return []

    if n == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]


def call_gemini_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    project_id: str = GCP_PROJECT_ID,
    location: str = GCP_LOCATION,
    model_id: str = GEMINI_MODEL,
    retry_count: int = 1000000000,
) -> Union[str, List[str]]:
    """Call Gemini LLM on Vertex AI with retry logic.

    Args:
        prompt: Text prompt to send to the model
        system_prompt: System instruction for the model
        n: Number of responses to generate
        temperature: Sampling temperature
        project_id: GCP project ID
        location: GCP region
        model_id: Gemini model resource name
        retry_count: Number of retries on rate limit errors

    Returns:
        Generated text(s) from the model

    Raises:
        NotImplementedError: If API access is denied
    """
    vertexai.init(project=project_id, location=location)

    harm_categories = [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]

    model = GenerativeModel(
        model_name=model_id,
        system_instruction=[system_prompt],
    )

    backoff = 1
    retry_count = int(retry_count)
    generation_config = GenerationConfig(
        temperature=temperature,
        candidate_count=n,
    )

    for _ in range(retry_count):
        try:
            response = model.generate_content(
                [prompt],
                generation_config=generation_config,
                safety_settings=[
                    SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE)
                    for h in harm_categories
                ],
            )
            break
        except Exception as exc:
            if "429" in str(exc):
                print("Retry due to rate limit: ", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue
            if "403" in str(exc):
                print("NO ACCESS TO ENDPOINT", exc)
                raise NotImplementedError("API access denied") from exc
            print("Exception: ", exc)
            return []

    try:
        # Keep this to check for errors in indexing.
        [candidate.text for candidate in response.candidates]
        if len(response.candidates) == 1:
            return response.candidates[0].text
        return [candidate.text for candidate in response.candidates]
    except Exception as exc:
        print("Error extracting text from response:", exc)
        return []


class RAG:
    """Retrieval Augmented Generation implementation using sentence transformers."""

    def __init__(
        self, docs: List[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize RAG with documents and model.

        Args:
            docs: List of documents to encode
            model: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model)
        self.docs = docs
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def top_k(self, query: str, k: int = 1) -> List[dict]:
        """Find top-k most similar documents to query.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of dicts containing similarity scores and document texts
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({"score": score, "text": self.docs[int(idx)]})
        return results
