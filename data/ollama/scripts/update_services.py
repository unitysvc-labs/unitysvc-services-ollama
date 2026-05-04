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

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.template_populate import populate_from_iterator

# Shared fetcher — instantiated once so the in-process LRU cache amortises
# across every model yielded by both BYOE and cloud iterators.
_FETCHER = ModelDataFetcher()


def _attach_canonical_metadata(details: dict, model_name: str) -> None:
    """Look up canonical context_length/parameter_count for ``model_name``.

    Always writes both keys (post-PR-#863 the platform validator requires
    presence; ``null`` is the canonical "unknown" marker).  Records
    ``metadata_sources`` provenance only when a fetcher actually returned
    a source for at least one field.
    """
    canonical = ModelDataLookup.get_canonical_metadata(model_name, fetcher=_FETCHER)
    details["context_length"] = canonical["context_length"]
    details["parameter_count"] = canonical["parameter_count"]
    sources = {k: v for k, v in canonical["sources"].items() if v}
    if sources:
        details["metadata_sources"] = sources

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

            # Description.  Sanitised at the source (rather than at the
            # ``yield`` sites) so every consumer of the scraped catalog
            # sees the cleaned form.
            desc_el = container.find("p", class_=re.compile(r"max-w-lg"))
            description = _sanitize_description(desc_el.get_text(strip=True)) if desc_el else ""

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


def _sanitize_description(text: str) -> str:
    """Strip characters that break the upload pipeline.

    The seller upload path encodes the request body to UTF-8 with
    ``ensure_ascii=False``; somewhere along the way (httpx + the
    backend's intake) descriptions containing supplementary-plane
    characters (emoji, U+1F42C 🐬 etc.) trigger a strict UTF-8 encode
    against an unpaired surrogate and the whole batch aborts with::

        'utf-8' codec can't encode characters in position N-N+1:
            surrogates not allowed

    Until the SDK/backend handles non-BMP code points cleanly, drop
    every non-BMP code point from scraped descriptions so the payload
    is safe to ship.  Plain BMP Unicode (accented Latin, CJK, ...)
    stays intact.
    """
    cleaned = "".join(ch for ch in text if ord(ch) < 0x10000)
    # Collapse runs of whitespace left behind by stripped chars and trim
    # the edges so descriptions don't begin with a stray leading space.
    return " ".join(cleaned.split())


def determine_service_type(model_name: str, capabilities: list[str]) -> str:
    """Determine service type from model name and capabilities."""
    name_lower = model_name.lower()
    caps_lower = [c.lower() for c in capabilities]

    if any(kw in name_lower or kw in " ".join(caps_lower) for kw in EMBEDDING_KEYWORDS):
        return "embedding"
    if any(kw in name_lower or kw in " ".join(caps_lower) for kw in VISION_KEYWORDS):
        return "llm"  # vision is a capability, not a service type
    return "llm"


def determine_tags(variant: str) -> list[str]:
    """Return valid tags for the service variant."""
    # TagEnum allows: byok, byoe, ai, gateway, managed
    return ["ai", variant]


# Models pre-pulled on the BYOE ops-test endpoint (ollama.svcmarket.com).
# Each catalog entry maps to the specific Ollama tag we've pulled — Ollama
# would otherwise resolve a bare name to ``:latest``, which for some
# families (qwen2.5, gemma3) defaults to a much larger size than we host.
# The listing template gates code-example test execution on membership in
# this map: members run code examples for real; non-members render the
# examples for users but mark them ``test.status = skip`` so CI only
# probes the connectivity test.  Add a row when a new tag is pulled
# upstream; remove when the model is evicted.
INSTALLED_BYOE_TAGS: dict[str, str] = {
    "llama3.2": "llama3.2:3b",
    "qwen2.5": "qwen2.5:1.5b",
    "gemma3": "gemma3:1b",
    "nomic-embed-text": "nomic-embed-text",
    "tinyllama": "tinyllama",
}


def iter_byoe_models(models: list[dict]) -> Iterator[dict]:
    """Yield BYOE (bring-your-own-endpoint) service dicts for all models.

    The listing template hard-codes the ops-testing endpoint
    (``ollama.svcmarket.com``) and uses the ``api_connectivity`` doc
    preset for the connectivity check, so the script only needs to
    yield the model-specific fields.
    """
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
        if service_type == "llm":
            _attach_canonical_metadata(details, model_name)

        installed_tag = INSTALLED_BYOE_TAGS.get(model_name)
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
            # When set, ops-test infra can exercise this service end-to-end
            # against ollama.svcmarket.com using this concrete tag.  When
            # unset, code-example docs are rendered with ``test.status =
            # skip`` so CI only runs the connectivity probe.
            "is_installed": installed_tag is not None,
            "routing_model": installed_tag or model_name,
        }
        print("  OK")


OLLAMA_CLOUD_MODELS_URL = "https://ollama.com/v1/models"


def fetch_ollama_cloud_models() -> list[str]:
    """Fetch the authoritative Ollama Cloud model id list.

    The cloud catalog is the source of truth for what ``routing_key.model``
    values upstream actually accepts.  ``ollama.com/search`` (the BYOE
    catalog) lists *family* names without size tags; the cloud only
    routes on the full id (e.g. ``gpt-oss:20b``, ``cogito-2.1:671b``).
    Sourcing BYOK services from this endpoint guarantees every routing
    key resolves to a real model and avoids 404s like the one in
    issue #cogito-2.1-byok.

    The endpoint is unauthenticated for catalog reads.
    """
    print(f"Fetching Ollama Cloud catalog from {OLLAMA_CLOUD_MODELS_URL}...")
    resp = requests.get(OLLAMA_CLOUD_MODELS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    ids = sorted(m["id"] for m in data if m.get("id"))
    print(f"Found {len(ids)} cloud models\n")
    return ids


def _slugify_cloud_id(cloud_id: str) -> str:
    """Turn a cloud model id into a filesystem-safe slug.

    ``gemma3:4b``           -> ``gemma3-4b``
    ``cogito-2.1:671b``     -> ``cogito-2.1-671b``
    ``deepseek-v4-pro``     -> ``deepseek-v4-pro`` (untagged)

    The colon is the only character we need to handle: cloud ids
    otherwise stick to ``[a-z0-9._-]``.  The slug becomes the service
    directory name and the offering's stable id; the original
    colon-bearing id stays intact in ``routing_model`` so routing
    targets the upstream model exactly.
    """
    return cloud_id.replace(":", "-")


def iter_cloud_models(models: list[dict]) -> Iterator[dict]:
    """Yield cloud (BYOK) service dicts, one per Ollama Cloud model id.

    Sources the canonical id list from ``ollama.com/v1/models`` rather
    than the search-scraped family names.  The scraped catalog
    (``models``) is consulted by family-name prefix to pull
    description / capabilities / pull_count metadata when available;
    the ids that don't match a scraped entry still ship with a
    stub description.
    """
    # Index the scraped catalog by family name so we can enrich each
    # cloud-listed id with the family's description / capabilities /
    # pull_count metadata when there's a match.
    scraped_by_family = {m["model_name"]: m for m in models if m.get("is_cloud")}

    cloud_ids = fetch_ollama_cloud_models()

    for i, cloud_id in enumerate(cloud_ids, 1):
        slug = _slugify_cloud_id(cloud_id)
        family = cloud_id.split(":", 1)[0]
        scraped = scraped_by_family.get(family, {})

        print(f"[{i}/{len(cloud_ids)}] {slug}-byok (routes to {cloud_id!r})")

        service_type = determine_service_type(family, scraped.get("capabilities", []))
        display_name = cloud_id.replace("-", " ").replace("_", " ").title()
        tags = determine_tags("byok")

        details: dict[str, Any] = {}
        if scraped.get("sizes"):
            details["available_sizes"] = scraped["sizes"]
        if scraped.get("pull_count"):
            details["pull_count"] = scraped["pull_count"]
        if service_type == "llm":
            # Canonical metadata lookup is keyed on family name;
            # specific size variants share the family's
            # context_length / parameter_count entry when present.
            _attach_canonical_metadata(details, family)

        yield {
            "name": f"{slug}-byok",
            # ``offering_name`` is the offering's stable id (no colon
            # for filesystem / URL safety); ``routing_model`` carries
            # the cloud's actual model id (with colon) for the
            # listing's routing_key and the offering's upstream
            # routing_key.
            "offering_name": slug,
            "routing_model": cloud_id,
            "display_name": f"{display_name} (Cloud)",
            "description": (
                scraped.get("description")
                or f"{display_name} model via Ollama Cloud"
            ),
            "service_type": service_type,
            "status": "ready",
            "capabilities": scraped.get("capabilities", []),
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
