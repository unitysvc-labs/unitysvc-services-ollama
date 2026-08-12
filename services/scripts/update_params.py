#!/usr/bin/env python3
"""Generate Ollama service params for the flat specs/ layout.

The catalog has one service per Ollama model id. BYOE and Ollama Cloud are
channels on that service, not separate services.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterator

import requests
from bs4 import BeautifulSoup

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

PROVIDER_NAME = "ollama"
PROVIDER_DISPLAY_NAME = "Ollama"
OLLAMA_SEARCH_URL = "https://ollama.com/search"
OLLAMA_CLOUD_MODELS_URL = "https://ollama.com/v1/models"
ENV_API_KEY_NAME = "OLLAMA_API_KEY"

SCRIPT_DIR = Path(__file__).parent

EMBEDDING_KEYWORDS = ["embedding", "embed"]
VISION_KEYWORDS = ["vision"]

INSTALLED_BYOE_TAGS: dict[str, str] = {
    "llama3.2": "llama3.2:3b",
    "qwen2.5": "qwen2.5:1.5b",
    "gemma3": "gemma3:1b",
    "nomic-embed-text": "nomic-embed-text",
    "tinyllama": "tinyllama",
}

# Models advertised by the public Ollama Cloud catalog but rejected by the
# staging seller-managed key during submit connectivity tests. Keep them BYOE
# until the managed account can actually route them.
OLLAMA_CLOUD_EXCLUDED_IDS: set[str] = {
    "deepseek-v3.1:671b",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "gemini-3-flash-preview",
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "minimax-m2.7",
    "mistral-large-3:675b",
    "qwen3.5:397b",
    "rnj-1:8b",
}

_FETCHER = ModelDataFetcher()


def _sanitize_description(text: str) -> str:
    cleaned = "".join(ch for ch in text if ord(ch) < 0x10000)
    return " ".join(cleaned.split())


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace(":", "-")


def _display_name(model_id: str) -> str:
    return model_id.replace(":", " ").replace("-", " ").replace("_", " ").title()


def _attach_canonical_metadata(details: dict[str, Any], model_name: str) -> None:
    canonical = ModelDataLookup.get_canonical_metadata(model_name, fetcher=_FETCHER)
    details["context_length"] = canonical["context_length"]
    details["parameter_count"] = canonical["parameter_count"]
    sources = {k: v for k, v in canonical["sources"].items() if v}
    if sources:
        details["metadata_sources"] = sources


def scrape_ollama_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    page = 1

    print(f"Fetching models from {OLLAMA_SEARCH_URL}...")
    while True:
        resp = requests.get(
            OLLAMA_SEARCH_URL,
            params={"q": "", "page": str(page)},
            headers={"HX-Request": "true"},
            timeout=30,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title_els = soup.find_all(attrs={"x-test-search-response-title": True})
        if not title_els:
            break

        for title_el in title_els:
            model_name = title_el.get_text(strip=True)
            container = title_el.find_parent("li") or title_el.find_parent("div")
            if container is None:
                continue

            desc_el = container.find("p", class_=re.compile(r"max-w-lg"))
            cap_els = container.find_all(attrs={"x-test-capability": True})
            size_els = container.find_all(attrs={"x-test-size": True})
            pull_el = container.find(attrs={"x-test-pull-count": True})
            tag_el = container.find(attrs={"x-test-tag-count": True})
            updated_el = container.find(attrs={"x-test-updated": True})
            cloud_els = container.find_all("span", string=re.compile(r"^\s*cloud\s*$", re.I))

            models.append(
                {
                    "model_name": model_name,
                    "description": _sanitize_description(desc_el.get_text(strip=True)) if desc_el else "",
                    "capabilities": [el.get_text(strip=True) for el in cap_els],
                    "sizes": [el.get_text(strip=True) for el in size_els],
                    "pull_count": pull_el.get_text(strip=True) if pull_el else "",
                    "tag_count": tag_el.get_text(strip=True) if tag_el else "",
                    "updated": updated_el.get_text(strip=True) if updated_el else "",
                    "is_cloud": bool(cloud_els),
                }
            )

        print(f"  Page {page}: {len(title_els)} models")
        page += 1

    print(f"Found {len(models)} models total\n")
    return models


def fetch_ollama_cloud_models() -> list[str]:
    print(f"Fetching Ollama Cloud catalog from {OLLAMA_CLOUD_MODELS_URL}...")
    resp = requests.get(OLLAMA_CLOUD_MODELS_URL, timeout=30)
    resp.raise_for_status()
    ids = sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))
    print(f"Found {len(ids)} cloud models\n")
    return ids


def determine_service_type(model_name: str, capabilities: list[str]) -> str:
    name_lower = model_name.lower()
    caps_lower = " ".join(c.lower() for c in capabilities)

    if any(kw in name_lower or kw in caps_lower for kw in EMBEDDING_KEYWORDS):
        return "embedding"
    if any(kw in name_lower or kw in caps_lower for kw in VISION_KEYWORDS):
        return "llm"
    return "llm"


def _details_for(model_name: str, scraped: dict[str, Any], service_type: str) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if scraped.get("sizes"):
        details["available_sizes"] = scraped["sizes"]
    if scraped.get("pull_count"):
        details["pull_count"] = scraped["pull_count"]
    if service_type == "llm":
        _attach_canonical_metadata(details, model_name)
    return details


def _vars_for(
    *,
    service_id: str,
    family: str,
    routing_model: str,
    scraped: dict[str, Any],
    has_cloud: bool,
) -> dict[str, Any]:
    capabilities = scraped.get("capabilities", [])
    service_type = determine_service_type(family, capabilities)
    display_name = _display_name(routing_model)
    installed_tag = INSTALLED_BYOE_TAGS.get(family)
    byoe_model = routing_model if ":" in routing_model else installed_tag or family

    params = {
        "name": f"{PROVIDER_NAME}/{service_id}",
        "offering_name": service_id,
        "display_name": display_name,
        "description": scraped.get("description") or f"{display_name} model via Ollama",
        "service_type": service_type,
        "status": "ready",
        "capabilities": capabilities,
        "details": _details_for(family, scraped, service_type),
        "tags": ["ai", "gateway", "byoe"] + (["managed", "byok"] if has_cloud else []),
        "provider_name": PROVIDER_NAME,
        "provider_display_name": PROVIDER_DISPLAY_NAME,
        "in_ollama_cloud": has_cloud,
        "is_installed": installed_tag is not None,
        "ops_testing_model": byoe_model,
    }
    if has_cloud:
        params["cloud_model"] = routing_model
    return params


def iter_models(models: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    scraped_by_family = {m["model_name"]: m for m in models}
    all_cloud_ids = fetch_ollama_cloud_models()
    cloud_ids = [model_id for model_id in all_cloud_ids if model_id not in OLLAMA_CLOUD_EXCLUDED_IDS]
    excluded_cloud_ids = [model_id for model_id in all_cloud_ids if model_id in OLLAMA_CLOUD_EXCLUDED_IDS]
    cloud_families = {model_id.split(":", 1)[0] for model_id in cloud_ids}
    excluded_cloud_families = {model_id.split(":", 1)[0] for model_id in excluded_cloud_ids}

    for i, cloud_id in enumerate(cloud_ids, 1):
        family = cloud_id.split(":", 1)[0]
        service_id = _slugify_model_id(cloud_id)
        scraped = scraped_by_family.get(family, {})
        print(f"[cloud {i}/{len(cloud_ids)}] {service_id} (routes to {cloud_id!r})")
        yield _vars_for(
            service_id=service_id,
            family=family,
            routing_model=cloud_id,
            scraped=scraped,
            has_cloud=True,
        )

    for i, cloud_id in enumerate(excluded_cloud_ids, 1):
        family = cloud_id.split(":", 1)[0]
        service_id = _slugify_model_id(cloud_id)
        scraped = scraped_by_family.get(family, {})
        print(f"[cloud-excluded {i}/{len(excluded_cloud_ids)}] {service_id} (routes to {cloud_id!r})")
        yield _vars_for(
            service_id=service_id,
            family=family,
            routing_model=cloud_id,
            scraped=scraped,
            has_cloud=False,
        )

    byoe_models = [
        m
        for m in models
        if m["model_name"] not in cloud_families and m["model_name"] not in excluded_cloud_families
    ]
    for i, model in enumerate(byoe_models, 1):
        model_name = model["model_name"]
        print(f"[byoe {i}/{len(byoe_models)}] {model_name}")
        yield _vars_for(
            service_id=model_name,
            family=model_name,
            routing_model=model_name,
            scraped=model,
            has_cloud=False,
        )


def main() -> None:
    models = scrape_ollama_models()
    if not models:
        print("Error: No models found")
        sys.exit(1)

    stats = write_params_from_iterator(
        iterator=iter_models(models),
        output_dir=SCRIPT_DIR.parent / "specs",
        prune_missing=True,
    )
    print(f"\nWrote {stats['written']} params ({stats['errors']} errors)")


if __name__ == "__main__":
    main()
