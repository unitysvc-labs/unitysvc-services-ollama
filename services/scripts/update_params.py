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
# The full library index renders every model on a single page. (The older
# /search?page=N endpoint was abandoned upstream: it now ignores the page param
# — every page returns the same first ~20 models — and dropped the x-test-*
# attributes this scraper used to key on, so it can no longer enumerate the
# catalog.)
OLLAMA_LIBRARY_URL = "https://ollama.com/library"
OLLAMA_CLOUD_MODELS_URL = "https://ollama.com/v1/models"
ENV_API_KEY_NAME = "OLLAMA_API_KEY"
# Some upstream responses vary by client; send a plain browser UA so we get the
# full server-rendered library rather than a trimmed/blocked variant.
_USER_AGENT = "Mozilla/5.0 (compatible; unitysvc-ollama-populator/1.0)"

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


def _parse_card(card: Any, model_name: str) -> dict[str, Any]:
    """Extract one model's metadata from its library-index card.

    Upstream replaced the old ``x-test-*`` hooks with Tailwind-styled badges, so
    each field is now identified by the badge's colour class:
      * capabilities (tools, vision, thinking, audio, ...) -> ``bg-indigo-50``
      * parameter sizes (8b, 70b, e4b, ...)                -> ``text-blue-600``
      * the Ollama Cloud marker                            -> ``bg-cyan-50`` "cloud"
    The pull count / tag count / updated string share one meta line.
    """
    desc_el = card.find("p", class_=re.compile(r"max-w-lg"))

    capabilities: list[str] = []
    sizes: list[str] = []
    is_cloud = False
    for badge in card.select("span[class]"):
        text = badge.get_text(strip=True)
        if not text:
            continue
        classes = badge.get("class", [])
        if "bg-cyan-50" in classes:
            if text.lower() == "cloud":
                is_cloud = True
        elif "bg-indigo-50" in classes:
            capabilities.append(text)
        elif "text-blue-600" in classes:
            sizes.append(text)

    meta_el = card.find("p", class_=re.compile(r"space-x-5"))
    meta_text = meta_el.get_text(" ", strip=True) if meta_el else ""
    tag_m = re.search(r"([\d.,]+)\s+Tags", meta_text)
    updated_m = re.search(r"Updated\s+(.+?)\s*$", meta_text)

    return {
        "model_name": model_name,
        "description": _sanitize_description(desc_el.get_text(strip=True)) if desc_el else "",
        "capabilities": capabilities,
        "sizes": sizes,
        "tag_count": tag_m.group(1).strip() if tag_m else "",
        "updated": updated_m.group(1).strip() if updated_m else "",
        "is_cloud": is_cloud,
    }


def scrape_ollama_models() -> list[dict[str, Any]]:
    print(f"Fetching models from {OLLAMA_LIBRARY_URL}...")
    resp = requests.get(OLLAMA_LIBRARY_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for link in soup.select('a[href^="/library/"]'):
        href = link.get("href", "")
        model_name = href[len("/library/"):].strip("/")
        # The index links straight to each model (``/library/<name>``); skip sub
        # links (``/library/<name>/tags``), empty hrefs, and duplicates.
        if not model_name or "/" in model_name or model_name in seen:
            continue
        seen.add(model_name)
        card = link.find_parent("li") or link
        models.append(_parse_card(card, model_name))

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


#: ``service_type`` -> platform capability vocabulary
#: (unitysvc ``docs/capabilities.yml``).
_SERVICE_TYPE_CAPABILITY = {"llm": "chat", "embedding": "embed"}


def _details_for(model_name: str, scraped: dict[str, Any], service_type: str) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if scraped.get("sizes"):
        details["available_sizes"] = scraped["sizes"]
    # Ollama's own library badges (tools, vision, thinking, audio, embedding).
    # These are ATTRIBUTES — they qualify what may appear in a request, they do
    # not name what the caller gets — so they do not belong in `capabilities`.
    # Kept here because they are scraped, not otherwise recoverable, and
    # `determine_service_type` reads them.
    if scraped.get("capabilities"):
        details["ollama_badges"] = scraped["capabilities"]
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
    badges = scraped.get("capabilities", [])
    service_type = determine_service_type(family, badges)
    # One entry, from the platform vocabulary: what the caller GETS.
    capabilities = [_SERVICE_TYPE_CAPABILITY.get(service_type, "chat")]
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
        "tags": ["ai", "gateway", "byoe"] + (["byok"] if has_cloud else []),
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
    # Cloud models the seller-managed key cannot actually route are kept
    # BYOE-only via per-model <name>.override.json companions
    # ({"parameters": {"in_ollama_cloud": false, ...}}), merged at render
    # time — this script treats the fetched catalog uniformly.
    cloud_ids = fetch_ollama_cloud_models()
    cloud_families = {model_id.split(":", 1)[0] for model_id in cloud_ids}

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

    byoe_models = [
        m
        for m in models
        if m["model_name"] not in cloud_families
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
