#!/usr/bin/env python3
"""
Template-based update_services.py for Ollama.

Scrapes ollama.com/search to discover all available models and yields
template dictionaries for both BYOE (bring-your-own-endpoint) and
cloud services.

Usage: python scripts/update_services.py
"""

import re
import sys
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from unitysvc_services import populate_from_iterator

# Provider Configuration
PROVIDER_NAME = "ollama"
PROVIDER_DISPLAY_NAME = "Ollama"
OLLAMA_SEARCH_URL = "https://ollama.com/search"

SCRIPT_DIR = Path(__file__).parent

# Capabilities that map to service_type
EMBEDDING_KEYWORDS = ["embedding", "embed"]
VISION_KEYWORDS = ["vision"]


def scrape_ollama_models() -> list[dict]:
    """Scrape all models from ollama.com/search with pagination."""
    models = []
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

        # Each model is in an <li> containing x-test-search-response-title
        title_els = soup.find_all(attrs={"x-test-search-response-title": True})
        if not title_els:
            break

        for title_el in title_els:
            model_name = title_el.get_text(strip=True)

            # Navigate to the parent container to find siblings
            container = title_el.find_parent("li") or title_el.find_parent("div")
            if container is None:
                continue

            # Description
            desc_el = container.find("p", class_=re.compile(r"max-w-lg"))
            description = desc_el.get_text(strip=True) if desc_el else ""

            # Capabilities
            cap_els = container.find_all(attrs={"x-test-capability": True})
            capabilities = [el.get_text(strip=True) for el in cap_els]

            # Sizes
            size_els = container.find_all(attrs={"x-test-size": True})
            sizes = [el.get_text(strip=True) for el in size_els]

            # Pull count
            pull_el = container.find(attrs={"x-test-pull-count": True})
            pull_count = pull_el.get_text(strip=True) if pull_el else ""

            # Tag count
            tag_el = container.find(attrs={"x-test-tag-count": True})
            tag_count = tag_el.get_text(strip=True) if tag_el else ""

            # Updated
            updated_el = container.find(attrs={"x-test-updated": True})
            updated = updated_el.get_text(strip=True) if updated_el else ""

            # Check if cloud-only (has cloud badge but no sizes = cloud-only)
            cloud_els = container.find_all(
                "span", string=re.compile(r"^\s*cloud\s*$", re.I)
            )
            is_cloud = len(cloud_els) > 0

            models.append({
                "model_name": model_name,
                "description": description,
                "capabilities": capabilities,
                "sizes": sizes,
                "pull_count": pull_count,
                "tag_count": tag_count,
                "updated": updated,
                "is_cloud": is_cloud,
            })

        print(f"  Page {page}: {len(title_els)} models")
        page += 1

    print(f"Found {len(models)} models total\n")
    return models


def determine_service_type(model_name: str, capabilities: list[str]) -> str:
    """Determine service type from model name and capabilities."""
    name_lower = model_name.lower()
    caps_lower = [c.lower() for c in capabilities]

    if any(kw in name_lower or kw in " ".join(caps_lower) for kw in EMBEDDING_KEYWORDS):
        return "embedding"
    if any(kw in name_lower or kw in " ".join(caps_lower) for kw in VISION_KEYWORDS):
        return "vision_language_model"
    return "llm"


def determine_tags(variant: str) -> list[str]:
    """Return valid tags for the service variant."""
    # TagEnum allows: byok, byoe, ai, gateway, managed
    return ["ai", variant]


def iter_byoe_models(models: list[dict]) -> Iterator[dict]:
    """Yield BYOE (bring-your-own-endpoint) service dicts for all models."""
    for i, model in enumerate(models, 1):
        model_name = model["model_name"]
        print(f"[{i}/{len(models)}] {model_name}-byoe")

        service_type = determine_service_type(model_name, model["capabilities"])
        display_name = model_name.replace("-", " ").replace("_", " ").title()
        tags = determine_tags("byoe")

        details = {}
        if model["sizes"]:
            details["available_sizes"] = model["sizes"]
        if model["pull_count"]:
            details["pull_count"] = model["pull_count"]

        yield {
            "name": f"{model_name}-byoe",
            "offering_name": model_name,
            "display_name": f"{display_name} (BYOE)",
            "description": model["description"] or f"{display_name} model via Ollama",
            "service_type": service_type,
            "status": "ready",
            "capabilities": model["capabilities"],
            "details": details,
            "tags": tags,
            "payout_price": None,
            "list_price": None,
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "service_variant": "byoe",
        }
        print("  OK")


def iter_cloud_models(models: list[dict]) -> Iterator[dict]:
    """Yield cloud (BYOK) service dicts for models that support Ollama cloud."""
    cloud_models = [m for m in models if m["is_cloud"]]
    for i, model in enumerate(cloud_models, 1):
        model_name = model["model_name"]
        print(f"[{i}/{len(cloud_models)}] {model_name}-byok")

        service_type = determine_service_type(model_name, model["capabilities"])
        display_name = model_name.replace("-", " ").replace("_", " ").title()
        tags = determine_tags("byok")

        details = {}
        if model["sizes"]:
            details["available_sizes"] = model["sizes"]
        if model["pull_count"]:
            details["pull_count"] = model["pull_count"]

        yield {
            "name": f"{model_name}-byok",
            "offering_name": model_name,
            "display_name": f"{display_name} (Cloud)",
            "description": model["description"] or f"{display_name} model via Ollama Cloud",
            "service_type": service_type,
            "status": "ready",
            "capabilities": model["capabilities"],
            "details": details,
            "tags": tags,
            "payout_price": None,
            "list_price": None,
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "env_api_key_name": "OLLAMA_API_KEY",
            "service_variant": "byok",
        }
        print("  OK")


def main():
    models = scrape_ollama_models()
    if not models:
        print("Error: No models found")
        sys.exit(1)

    templates_dir = SCRIPT_DIR.parent / "templates"
    output_dir = SCRIPT_DIR.parent / "services"

    # Populate BYOE services (all models)
    print("=== Populating BYOE services ===")
    populate_from_iterator(
        iterator=iter_byoe_models(models),
        templates_dir=templates_dir,
        output_dir=output_dir,
        offering_template="offering-byoe.json.j2",
        listing_template="listing-byoe.json.j2",
        deprecate_missing=False,
    )

    # Populate cloud/BYOK services (cloud-enabled models only)
    print("\n=== Populating cloud (BYOK) services ===")
    populate_from_iterator(
        iterator=iter_cloud_models(models),
        templates_dir=templates_dir,
        output_dir=output_dir,
        offering_template="offering-byok.json.j2",
        listing_template="listing-byok.json.j2",
        deprecate_missing=False,
    )


if __name__ == "__main__":
    main()
