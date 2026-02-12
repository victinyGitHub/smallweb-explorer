#!/usr/bin/env python3
"""
smallweb API server - serves graphs and spawns crawls.
Runs alongside the main mains.in.net express server on a different port.
Nginx proxies /smallweb/api/* to this.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from aiohttp import web

# Import from smallweb.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from smallweb import WebGraph, crawl, render_html
from taste import TasteModel
from scoring_config import load_config, save_config, with_overrides, apply_preset, DEFAULTS as CONFIG_DEFAULTS

GRAPHS_DIR = Path("/var/www/arg/mains.in.net/smallweb/graphs")
STATIC_DIR = Path("/var/www/arg/mains.in.net/smallweb")

# Track running crawls
active_crawls = {}  # id -> {status, progress, name, seeds, ...}

# Cache taste models per graph (avoids reloading on every request)
_taste_cache = {}  # graph_id -> TasteModel

# Cache scoring configs per graph
_config_cache = {}  # graph_id -> dict

# Cache loaded graph objects (avoids re-parsing JSON + recomputing PageRank)
_graph_cache = {}  # graph_id -> (mtime, WebGraph)

def _get_taste_model(graph_id: str, graph: WebGraph = None) -> TasteModel:
    """Get or create a TasteModel for a graph."""
    if graph_id not in _taste_cache:
        graph_path = str(GRAPHS_DIR / f"{graph_id}.json")
        _taste_cache[graph_id] = TasteModel(graph=graph, graph_path=graph_path)
    elif graph is not None:
        _taste_cache[graph_id].graph = graph
    return _taste_cache[graph_id]


def _get_graph(graph_id: str) -> WebGraph:
    """Get or load a cached graph. Auto-invalidates if file changed."""
    path = GRAPHS_DIR / f"{graph_id}.json"
    mtime = path.stat().st_mtime
    if graph_id in _graph_cache:
        cached_mtime, cached_graph = _graph_cache[graph_id]
        if cached_mtime == mtime:
            return cached_graph
    graph = WebGraph.load(str(path))
    _graph_cache[graph_id] = (mtime, graph)
    return graph


def _get_config(graph_id: str) -> dict:
    """Get or load scoring config for a graph."""
    if graph_id not in _config_cache:
        graph_path = str(GRAPHS_DIR / f"{graph_id}.json")
        _config_cache[graph_id] = load_config(graph_path)
    return _config_cache[graph_id]


async def handle_get_config(request):
    """Get scoring config for a graph (merged with defaults)."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)
    cfg = _get_config(graph_id)
    return web.json_response(cfg)


async def handle_put_config(request):
    """Save scoring config overrides for a graph."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)
    try:
        overrides = await request.json()
    except:
        return web.json_response({"error": "invalid JSON"}, status=400)

    # Merge overrides onto current config
    cfg = _get_config(graph_id)
    new_cfg = with_overrides(cfg, overrides)
    # Save and update cache
    save_config(str(path), new_cfg)
    _config_cache[graph_id] = new_cfg
    return web.json_response({"ok": True, "config": new_cfg})


async def handle_config_presets(request):
    """List available presets."""
    graph_id = request.match_info["id"]
    cfg = _get_config(graph_id)
    presets = cfg.get("presets", CONFIG_DEFAULTS.get("presets", {}))
    return web.json_response({"presets": presets})


async def handle_apply_preset(request):
    """Apply a named preset to a graph's config."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)
    try:
        body = await request.json()
    except:
        return web.json_response({"error": "invalid JSON"}, status=400)

    preset_name = body.get("preset", "")
    cfg = _get_config(graph_id)
    new_cfg = apply_preset(cfg, preset_name)
    save_config(str(path), new_cfg)
    _config_cache[graph_id] = new_cfg
    return web.json_response({"ok": True, "config": new_cfg})


async def handle_index(request):
    """Serve the main smallweb page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return web.Response(text=index_path.read_text(), content_type="text/html")
    return web.Response(text="smallweb index not found", status=404)


async def handle_list_graphs(request):
    """List all saved graphs."""
    graphs = []
    for f in sorted(GRAPHS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        # Skip taste label files, model files, and config files
        if f.stem.endswith(".taste") or f.stem.endswith(".config") or f.suffix != ".json":
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            meta = data.get("metadata", {})
            n_nodes = len(data.get("nodes", {}))
            n_edges = sum(len(v) for v in data.get("edges", {}).values())
            n_seeds = len(data.get("seeds", []))
            graphs.append({
                "id": f.stem,
                "name": meta.get("name", f.stem),
                "author": meta.get("author", ""),
                "created_at": meta.get("created_at", ""),
                "forked_from": meta.get("forked_from", ""),
                "nodes": n_nodes,
                "edges": n_edges,
                "seeds": n_seeds,
                "file": f.name,
            })
        except Exception as e:
            print(f"error reading {f}: {e}")
    return web.json_response({"graphs": graphs})


async def handle_get_graph(request):
    """Get a specific graph as JSON."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)
    with open(path) as f:
        return web.json_response(json.load(f))


async def handle_graph_html(request):
    """Get a graph rendered as HTML."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)
    graph = WebGraph.load(str(path))
    html = render_html(graph, f"{graph_id}.json", graph_id=graph_id)
    return web.Response(text=html, content_type="text/html")


async def handle_start_crawl(request):
    """Start a new crawl job."""
    try:
        body = await request.json()
    except:
        return web.json_response({"error": "invalid JSON"}, status=400)

    seeds = body.get("seeds", [])
    if isinstance(seeds, str):
        seeds = [s.strip() for s in seeds.split(",") if s.strip()]

    if not seeds:
        return web.json_response({"error": "no seeds provided"}, status=400)

    # Normalize seeds
    seeds = [s if s.startswith("http") else f"https://{s}" for s in seeds]

    name = body.get("name", "").strip()
    if not name:
        # Generate name from first seed domain
        from urllib.parse import urlparse
        name = urlparse(seeds[0]).netloc.replace(".", "-")

    # Sanitize name for filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()[:50]
    if not safe_name:
        safe_name = "crawl"

    # Check for name collision, add suffix if needed
    output_path = GRAPHS_DIR / f"{safe_name}.json"
    if output_path.exists():
        safe_name = f"{safe_name}-{int(time.time()) % 10000}"
        output_path = GRAPHS_DIR / f"{safe_name}.json"

    hops = min(int(body.get("hops", 1)), 3)  # Cap at 3 hops
    max_pages = min(int(body.get("max_pages", 100)), 500)  # Cap at 500 pages
    domain_cap = min(int(body.get("domain_cap", 20)), 50)  # Cap at 50 per domain

    crawl_id = safe_name
    active_crawls[crawl_id] = {
        "status": "running",
        "name": name,
        "safe_name": safe_name,
        "seeds": seeds,
        "hops": hops,
        "max_pages": max_pages,
        "started_at": datetime.now().isoformat(),
        "progress": "starting...",
    }

    # Start crawl in background
    asyncio.create_task(_run_crawl(crawl_id, seeds, hops, max_pages, name, output_path, domain_cap))

    return web.json_response({
        "id": crawl_id,
        "status": "running",
        "message": f"crawling from {len(seeds)} seeds, max {hops} hops, max {max_pages} pages",
    })


async def _run_crawl(crawl_id, seeds, hops, max_pages, name, output_path, domain_cap=20):
    """Run a crawl in the background."""
    try:
        graph = await crawl(seeds, max_hops=hops, max_pages=max_pages, name=name, domain_cap=domain_cap)
        graph.save(str(output_path))

        # Also generate static HTML
        html = render_html(graph, output_path.name)
        html_path = STATIC_DIR / f"{output_path.stem}.html"
        html_path.write_text(html)

        stats = graph.stats()
        active_crawls[crawl_id] = {
            "status": "done",
            "name": name,
            "safe_name": crawl_id,
            "seeds": seeds,
            "hops": hops,
            "max_pages": max_pages,
            "started_at": active_crawls[crawl_id]["started_at"],
            "finished_at": datetime.now().isoformat(),
            "progress": "complete",
            "result": {
                "nodes": stats["nodes"],
                "edges": stats["edges"],
                "seeds": stats["seeds"],
                "domains": stats["domains"],
            }
        }
    except Exception as e:
        active_crawls[crawl_id] = {
            **active_crawls.get(crawl_id, {}),
            "status": "error",
            "error": str(e),
        }
        print(f"crawl error for {crawl_id}: {e}")


async def handle_crawl_status(request):
    """Check status of a crawl."""
    crawl_id = request.match_info["id"]
    if crawl_id in active_crawls:
        return web.json_response(active_crawls[crawl_id])
    return web.json_response({"error": "crawl not found"}, status=404)


async def handle_list_crawls(request):
    """List all active/recent crawls."""
    return web.json_response({"crawls": active_crawls})


async def handle_fork(request):
    """Fork an existing graph with new params and optionally re-crawl."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    try:
        body = await request.json()
    except:
        body = {}

    graph = WebGraph.load(str(path))

    # Fork with optional seed promotion
    name = body.get("name", f"fork-{graph_id}")
    add_seeds = body.get("add_seeds", [])
    promote_top = int(body.get("promote_top", 0))
    damping = float(body.get("damping", 0.95))
    iterations = int(body.get("iterations", 50))

    forked = graph.fork(
        name=name,
        add_seeds=add_seeds if add_seeds else None,
        promote_top_n=promote_top,
    )
    forked.metadata["damping"] = damping
    forked.metadata["iterations"] = iterations

    # Generate safe filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()[:50]
    if not safe_name:
        safe_name = f"fork-{graph_id}"
    output_path = GRAPHS_DIR / f"{safe_name}.json"
    if output_path.exists():
        safe_name = f"{safe_name}-{int(time.time()) % 10000}"
        output_path = GRAPHS_DIR / f"{safe_name}.json"

    recrawl = body.get("recrawl", False)
    if recrawl:
        # Re-crawl from forked seeds
        hops = min(int(body.get("hops", 2)), 3)
        max_pages = min(int(body.get("max_pages", 200)), 500)
        domain_cap = min(int(body.get("domain_cap", 20)), 50)

        crawl_id = safe_name
        active_crawls[crawl_id] = {
            "status": "running",
            "name": name,
            "safe_name": safe_name,
            "seeds": list(forked.seeds),
            "hops": hops,
            "max_pages": max_pages,
            "started_at": datetime.now().isoformat(),
            "progress": "forking + re-crawling...",
            "forked_from": graph_id,
        }
        asyncio.create_task(_run_crawl(crawl_id, list(forked.seeds), hops, max_pages, name, output_path, domain_cap))

        return web.json_response({
            "id": crawl_id,
            "status": "running",
            "forked_from": graph_id,
            "seeds": len(forked.seeds),
            "original_seeds": len(graph.seeds),
            "new_seeds": len(forked.seeds) - len(graph.seeds),
            "message": f"forking {graph_id} with {len(forked.seeds)} seeds, re-crawling...",
        })
    else:
        # Just save the forked graph (no re-crawl)
        forked.save(str(output_path))
        html = render_html(forked, output_path.name)
        html_path = STATIC_DIR / f"{safe_name}.html"
        html_path.write_text(html)

        return web.json_response({
            "id": safe_name,
            "forked_from": graph_id,
            "seeds": len(forked.seeds),
            "original_seeds": len(graph.seeds),
            "new_seeds": len(forked.seeds) - len(graph.seeds),
            "damping": damping,
            "iterations": iterations,
            "message": f"forked {graph_id} → {safe_name}",
        })


async def handle_discoveries(request):
    """Get discoveries for a graph.

    Returns the union of top-N results from each sort category (overall,
    pagerank, quality, smallweb, taste) so the frontend can sort by any
    signal without missing pages that rank high in one dimension but low
    in the default blended sort.
    """
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    try:
        top = int(request.query.get("top", 50))
        damping = float(request.query.get("damping", 0.95))
        iterations = int(request.query.get("iterations", 50))
        personalized = request.query.get("personalized", "true").lower() != "false"
        graph = _get_graph(graph_id)

        # Load scoring config, with optional per-request overrides
        cfg = _get_config(graph_id)
        config_param = request.query.get("config", "")
        if config_param:
            try:
                import urllib.parse
                overrides = json.loads(urllib.parse.unquote(config_param))
                cfg = with_overrides(cfg, overrides)
            except (json.JSONDecodeError, ValueError):
                pass  # bad config param, use saved config

        # Load taste model if trained
        taste = _get_taste_model(graph_id, graph)
        use_taste = taste.is_trained

        # Get a large pool of discoveries so we can pick top-N per category.
        # We request 5x the per-category limit to have enough candidates.
        pool_size = top * 5

        # Run discoveries in a thread to avoid blocking the event loop
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool_exec:
            discoveries = await loop.run_in_executor(
                pool_exec,
                lambda: graph.discoveries(
                    top_n=pool_size, damping=damping, iterations=iterations,
                    personalized=personalized,
                    taste_model=taste if use_taste else None,
                    cfg=cfg,
                )
            )

        # Convert to dicts
        all_items = []
        for url, score, node in discoveries:
            pr_pct = node.get("pagerank_pct", 0)
            q = node.get("quality", 0.5)
            sw = node.get("smallweb_score", 0.5)
            taste = node.get("taste_score", 0.5)
            q_measured = "quality" in node

            # Build the overall score breakdown — reflects current config exponents
            f_cfg = cfg.get("formula", {})
            pr_exp = f_cfg.get("pagerank_exp", 1.0)
            q_exp = f_cfg.get("quality_exp", 1.0)
            sw_exp = f_cfg.get("smallweb_exp", 1.0)
            taste_base = f_cfg.get("taste_base", 0.5)
            taste_weight = f_cfg.get("taste_weight", 0.5)
            fetched_boost_val = f_cfg.get("fetched_boost", 0.2)

            def _exp_str(name, exp):
                if exp == 1.0:
                    return name
                return f"{name}^{exp}"

            formula_parts = [_exp_str("pagerank", pr_exp), _exp_str("quality", q_exp), _exp_str("smallweb", sw_exp)]
            if use_taste:
                formula_parts.append(f"({taste_base} + {taste_weight} × taste)")
            if not q_measured:
                formula_parts.append("fetched_boost")
            formula_str = " × ".join(formula_parts)

            overall_breakdown = {
                "formula": formula_str,
                "components": {
                    "pagerank_pct": {"value": round(pr_pct, 4), "exp": pr_exp, "why": "How well-connected this page is in the link graph, as a percentile rank (0-1). Higher = more pages link to it, especially authoritative ones."},
                    "quality": {"value": round(q, 3), "measured": q_measured, "exp": q_exp, "why": "HTML cleanliness score. Penalizes trackers, ads, bloat. Rewards IndieWeb signals. '?' means we haven't fetched this page yet."},
                    "smallweb": {"value": round(sw, 3), "exp": sw_exp, "why": "Is this domain part of the small web? Combines popularity (too popular = platform) and outlink profile (links to other small sites = good)."},
                },
            }
            if use_taste:
                overall_breakdown["components"]["taste"] = {"value": round(taste, 3), "why": "Neural taste score trained on your upvotes/downvotes. 0.5 = neutral, 1.0 = strong match."}
            if not q_measured:
                overall_breakdown["components"]["fetched_boost"] = {"value": fetched_boost_val, "why": "This page hasn't been crawled yet, so quality is estimated. Fetched pages get priority since we actually measured their HTML."}

            item = {
                "url": url,
                "score": score,
                "score_breakdown": overall_breakdown,
                "pagerank": node.get("pagerank", 0),
                "pagerank_pct": pr_pct,
                "title": node.get("title", ""),
                "description": node.get("description", ""),
                "domain": node.get("domain", ""),
                "quality": q,
                "quality_measured": q_measured,
                "quality_breakdown": node.get("quality_breakdown"),  # detailed HTML analysis (if fetched)
                "anchor_texts": node.get("anchor_texts", []),
                "smallweb_score": sw,
                "inbound_domains": node.get("inbound_domains", 0),
                "outlink_score": node.get("outlink_score", 0.5),
                "popularity_score": node.get("popularity_score", 0.5),
                "taste_score": taste,
            }
            all_items.append(item)

        # Build union of top-N from each sort category
        seen_urls = set()
        results = []

        def add_top(items, key, n):
            sorted_items = sorted(items, key=key, reverse=True)
            for item in sorted_items[:n]:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    results.append(item)

        # Top by overall blended score (already sorted this way)
        add_top(all_items, lambda d: d["score"], top)
        # Top by raw pagerank
        add_top(all_items, lambda d: d["pagerank"], top)
        # Top by quality (only measured pages matter)
        add_top(all_items, lambda d: d["quality"] if d["quality_measured"] else -1, top)
        # Top by smallweb score
        add_top(all_items, lambda d: d["smallweb_score"], top)
        # Top by taste score
        if use_taste:
            add_top(all_items, lambda d: d["taste_score"], top)

        # Sort final results by blended score for default display order
        results.sort(key=lambda d: d["score"], reverse=True)

        return web.json_response({
            "discoveries": results,
            "total": len(results),
            "taste_active": use_taste,
            "config": cfg,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def handle_similar(request):
    """Find sites similar to a target domain via co-citation."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    target = request.query.get("target", "")
    if not target:
        return web.json_response({"error": "target parameter required"}, status=400)

    top = int(request.query.get("top", 20))
    graph = WebGraph.load(str(path))
    results = graph.similar_sites(target, top_n=top)

    return web.json_response({
        "target": target,
        "similar": [
            {"domain": domain, "similarity": round(cosine, 4), "shared_sources": shared}
            for domain, cosine, shared in results
        ],
    })


async def handle_similarities(request):
    """Find all similar domain pairs in a graph."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    min_shared = int(request.query.get("min_shared", 2))
    top = int(request.query.get("top", 50))
    graph = WebGraph.load(str(path))
    pairs = graph.all_similarities(min_shared=min_shared, top_n=top)

    return web.json_response({
        "pairs": [
            {"domain_a": d_a, "domain_b": d_b, "similarity": round(cosine, 4), "shared_sources": shared}
            for d_a, d_b, cosine, shared in pairs
        ],
    })


async def handle_taste_label(request):
    """Add a taste label (positive or negative) for a URL."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    try:
        body = await request.json()
    except:
        return web.json_response({"error": "invalid JSON"}, status=400)

    url = body.get("url", "").strip()
    label = body.get("label", "")  # "positive" or "negative"

    if not url:
        return web.json_response({"error": "url required"}, status=400)
    if label not in ("positive", "negative", "remove"):
        return web.json_response({"error": "label must be 'positive', 'negative', or 'remove'"}, status=400)

    graph = WebGraph.load(str(path))
    taste = _get_taste_model(graph_id, graph)

    if label == "positive":
        taste.add_positive(url)
    elif label == "negative":
        taste.add_negative(url)
    elif label == "remove":
        taste.remove_label(url)

    return web.json_response({
        "url": url,
        "label": label,
        "stats": taste.stats(),
    })


async def handle_taste_train(request):
    """Train the taste model on current labels."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    graph = WebGraph.load(str(path))
    taste = _get_taste_model(graph_id, graph)

    # Train in thread (loads embedding model, CPU-intensive)
    import concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, taste.train)
    return web.json_response(result)


async def handle_taste_status(request):
    """Get taste model status and labels for a graph."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    graph = WebGraph.load(str(path))
    taste = _get_taste_model(graph_id, graph)

    return web.json_response({
        **taste.stats(),
        "positive_urls": taste.positive,
        "negative_urls": taste.negative,
    })


async def handle_taste_batch(request):
    """Add multiple taste labels at once."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    try:
        body = await request.json()
    except:
        return web.json_response({"error": "invalid JSON"}, status=400)

    graph = WebGraph.load(str(path))
    taste = _get_taste_model(graph_id, graph)

    positive = body.get("positive", [])
    negative = body.get("negative", [])
    auto_train = body.get("train", True)

    for url in positive:
        taste.add_positive(url)
    for url in negative:
        taste.add_negative(url)

    result = {"added_positive": len(positive), "added_negative": len(negative)}

    if auto_train and taste.has_labels:
        train_result = taste.train()
        result["train"] = train_result

    result["stats"] = taste.stats()
    return web.json_response(result)


async def handle_domain_graph(request):
    """Return a domain-level aggregated graph for visualization."""
    from urllib.parse import urlparse
    from collections import defaultdict

    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    top = int(request.query.get("top", 80))
    graph = _get_graph(graph_id)

    # Aggregate nodes by domain
    domain_data = defaultdict(lambda: {
        "pages": 0, "pagerank": 0, "quality_sum": 0, "quality_count": 0,
        "smallweb": 0, "is_seed": False, "top_title": ""
    })

    seed_domains = set()
    for seed in graph.seeds:
        try:
            sd = urlparse(seed).netloc.lower()
            seed_domains.add(sd)
        except:
            pass

    for url, node in graph.nodes.items():
        domain = node.get("domain") or urlparse(url).netloc.lower()
        dd = domain_data[domain]
        dd["pages"] += 1
        pr = node.get("pagerank", node.get("pagerank_pct", 0))
        if pr > dd["pagerank"]:
            dd["pagerank"] = pr
            dd["top_title"] = node.get("title", "")
        if "quality" in node:
            dd["quality_sum"] += node["quality"]
            dd["quality_count"] += 1
        dd["smallweb"] = max(dd["smallweb"], node.get("smallweb_score", 0))
        if domain in seed_domains:
            dd["is_seed"] = True

    # Build domain nodes
    domain_nodes = []
    for domain, dd in domain_data.items():
        avg_quality = dd["quality_sum"] / dd["quality_count"] if dd["quality_count"] > 0 else 0.5
        domain_nodes.append({
            "id": domain,
            "pages": dd["pages"],
            "pagerank": dd["pagerank"],
            "quality": round(avg_quality, 3),
            "smallweb": round(dd["smallweb"], 3),
            "is_seed": dd["is_seed"],
            "top_title": dd["top_title"][:60] if dd["top_title"] else ""
        })

    # Sort by pagerank, take top N
    domain_nodes.sort(key=lambda n: n["pagerank"], reverse=True)
    # Always include seeds
    seed_nodes = [n for n in domain_nodes if n["is_seed"]]
    non_seed = [n for n in domain_nodes if not n["is_seed"]]
    remaining = max(0, top - len(seed_nodes))
    domain_nodes = seed_nodes + non_seed[:remaining]
    visible_domains = set(n["id"] for n in domain_nodes)

    # Aggregate edges by domain pair
    domain_links = defaultdict(int)
    for from_url, to_urls in graph.edges.items():
        from_domain = urlparse(from_url).netloc.lower()
        if from_domain not in visible_domains:
            continue
        for to_url in to_urls:
            to_domain = urlparse(to_url).netloc.lower()
            if to_domain != from_domain and to_domain in visible_domains:
                key = (from_domain, to_domain)
                domain_links[key] += 1

    links = [{"source": s, "target": t, "weight": w} for (s, t), w in domain_links.items()]

    return web.json_response({"nodes": domain_nodes, "links": links})


async def handle_audit(request):
    """Trace the discovery chain from seeds to a target URL."""
    from urllib.parse import urlparse
    from collections import defaultdict, deque

    graph_id = request.match_info["id"]
    target_url = request.query.get("url", "")
    if not target_url:
        return web.json_response({"error": "url parameter required"}, status=400)

    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    graph = _get_graph(graph_id)

    # Build URL-level reverse edges
    reverse = defaultdict(set)
    for from_url, to_urls in graph.edges.items():
        for to_url in to_urls:
            reverse[to_url].add(from_url)

    # BFS backwards from target to any seed
    seed_set = set(graph.seeds)
    parent = {target_url: None}
    queue = deque([target_url])
    found_seed = None

    while queue and found_seed is None:
        current = queue.popleft()
        if current in seed_set:
            found_seed = current
            break
        for prev_url in reverse.get(current, []):
            if prev_url not in parent:
                parent[prev_url] = current
                queue.append(prev_url)
                if prev_url in seed_set:
                    found_seed = prev_url
                    break

    # Reconstruct chain
    chain = []
    if found_seed is not None:
        # Walk from seed to target
        path_urls = []
        current = found_seed
        while current is not None:
            path_urls.append(current)
            current = parent.get(current)
        for url in path_urls:
            node = graph.nodes.get(url, {})
            chain.append({
                "url": url,
                "title": node.get("title", ""),
                "domain": node.get("domain", urlparse(url).netloc.lower()),
                "is_seed": url in seed_set,
                "depth": node.get("depth", -1)
            })

    # Count alternative paths (other seeds that can reach the target)
    alt_paths = 0
    if found_seed:
        for seed in seed_set:
            if seed != found_seed and seed in parent:
                alt_paths += 1

    # Direct inbound links to target with anchor texts
    inbound_links = []
    target_node = graph.nodes.get(target_url, {})
    target_anchors = target_node.get("anchor_texts", [])
    for from_url in reverse.get(target_url, set()):
        from_node = graph.nodes.get(from_url, {})
        from_domain = from_node.get("domain", urlparse(from_url).netloc.lower())
        # Try to find matching anchor text
        anchor = ""
        for a in target_anchors:
            if len(a) > 3:
                anchor = a
                break
        inbound_links.append({
            "url": from_url,
            "domain": from_domain,
            "title": from_node.get("title", ""),
            "anchor_text": anchor
        })
    inbound_links = inbound_links[:20]

    # Co-cited domains (share inbound sources)
    target_domain = target_node.get("domain", urlparse(target_url).netloc.lower())
    co_cited = []
    try:
        similar = graph.similar_sites(target_domain, top_n=5)
        co_cited = [d for d, _, _ in similar if d != target_domain]
    except:
        pass

    return web.json_response({
        "target": target_url,
        "chain": chain,
        "alternative_paths": alt_paths,
        "inbound_links": inbound_links,
        "co_cited_with": co_cited
    })


async def warmup_embedding_model(app):
    """Pre-load the sentence-transformer model at startup so first request isn't slow."""
    import concurrent.futures
    loop = asyncio.get_event_loop()
    print("pre-loading embedding model (this takes ~5-10s on first run)...")
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, lambda: __import__('taste')._get_embed_model())
        print("embedding model loaded ✓")
    except Exception as e:
        print(f"warning: failed to pre-load embedding model: {e}")
        print("taste scoring will load on first use instead")


def create_app():
    app = web.Application()
    app.on_startup.append(warmup_embedding_model)

    # API routes
    app.router.add_get("/api/graphs", handle_list_graphs)
    app.router.add_get("/api/graphs/{id}", handle_get_graph)
    app.router.add_get("/api/graphs/{id}/html", handle_graph_html)
    app.router.add_get("/api/graphs/{id}/discoveries", handle_discoveries)
    app.router.add_get("/api/graphs/{id}/config", handle_get_config)
    app.router.add_put("/api/graphs/{id}/config", handle_put_config)
    app.router.add_get("/api/graphs/{id}/config/presets", handle_config_presets)
    app.router.add_post("/api/graphs/{id}/config/preset", handle_apply_preset)
    app.router.add_get("/api/graphs/{id}/similar", handle_similar)
    app.router.add_get("/api/graphs/{id}/similarities", handle_similarities)
    app.router.add_post("/api/graphs/{id}/fork", handle_fork)
    app.router.add_post("/api/graphs/{id}/taste/label", handle_taste_label)
    app.router.add_post("/api/graphs/{id}/taste/train", handle_taste_train)
    app.router.add_get("/api/graphs/{id}/taste", handle_taste_status)
    app.router.add_post("/api/graphs/{id}/taste/batch", handle_taste_batch)
    app.router.add_get("/api/graphs/{id}/domain-graph", handle_domain_graph)
    app.router.add_get("/api/graphs/{id}/audit", handle_audit)
    app.router.add_post("/api/crawl", handle_start_crawl)
    app.router.add_get("/api/crawl/{id}", handle_crawl_status)
    app.router.add_get("/api/crawls", handle_list_crawls)

    # Static files
    app.router.add_get("/", handle_index)
    app.router.add_static("/graphs/", GRAPHS_DIR)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"smallweb server starting on {args.host}:{args.port}")
    print(f"graphs dir: {GRAPHS_DIR}")
    app = create_app()
    web.run_app(app, host=args.host, port=args.port, print=print)
