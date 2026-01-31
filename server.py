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

GRAPHS_DIR = Path("/var/www/arg/mains.in.net/smallweb/graphs")
STATIC_DIR = Path("/var/www/arg/mains.in.net/smallweb")

# Track running crawls
active_crawls = {}  # id -> {status, progress, name, seeds, ...}


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
    """Get discoveries for a graph."""
    graph_id = request.match_info["id"]
    path = GRAPHS_DIR / f"{graph_id}.json"
    if not path.exists():
        return web.json_response({"error": "graph not found"}, status=404)

    top = int(request.query.get("top", 30))
    damping = float(request.query.get("damping", 0.95))
    iterations = int(request.query.get("iterations", 50))
    personalized = request.query.get("personalized", "true").lower() != "false"
    graph = WebGraph.load(str(path))
    discoveries = graph.discoveries(top_n=top, damping=damping, iterations=iterations, personalized=personalized)

    results = []
    for url, score, node in discoveries:
        results.append({
            "url": url,
            "score": score,
            "title": node.get("title", ""),
            "description": node.get("description", ""),
            "domain": node.get("domain", ""),
            "quality": node.get("quality", 1.0),
            "anchor_texts": node.get("anchor_texts", []),
            "smallweb_score": node.get("smallweb_score", 0.5),
            "inbound_domains": node.get("inbound_domains", 0),
            "outlink_score": node.get("outlink_score", 0.5),
            "popularity_score": node.get("popularity_score", 0.5),
        })

    return web.json_response({"discoveries": results, "total": len(results)})


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


def create_app():
    app = web.Application()

    # API routes
    app.router.add_get("/api/graphs", handle_list_graphs)
    app.router.add_get("/api/graphs/{id}", handle_get_graph)
    app.router.add_get("/api/graphs/{id}/html", handle_graph_html)
    app.router.add_get("/api/graphs/{id}/discoveries", handle_discoveries)
    app.router.add_get("/api/graphs/{id}/similar", handle_similar)
    app.router.add_get("/api/graphs/{id}/similarities", handle_similarities)
    app.router.add_post("/api/graphs/{id}/fork", handle_fork)
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
