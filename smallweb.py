#!/usr/bin/env python3
"""
smallweb - a small web discovery engine

Crawl niche corners of the web from seed URLs, build a local link graph,
rank pages by local PageRank, and discover related sites.

Graphs are portable JSON files - fork them, merge them, share them.

Usage:
    smallweb crawl <seed_urls_or_file> [--hops N] [--max-pages N] [--output graph.json]
    smallweb rank <graph.json> [--top N]
    smallweb discover <graph.json> [--top N]  # show pages not in seeds
    smallweb fork <graph.json> [--output forked.json]
    smallweb merge <graph1.json> <graph2.json> [--output merged.json]
    smallweb explore <graph.json>  # interactive exploration
    smallweb serve <graph.json> [--port 8080]  # web UI
    smallweb info <graph.json>  # graph stats
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup


# ── Graph Data Structure ──────────────────────────────────────────────

class WebGraph:
    """
    A local web graph - nodes are URLs, edges are links between them.
    Portable as JSON. Forkable. Mergeable.
    """

    def __init__(self):
        self.nodes: Dict[str, dict] = {}  # url -> {title, description, crawled_at, depth, ...}
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # from_url -> {to_url, ...}
        self.seeds: Set[str] = set()
        self.metadata: dict = {
            "created_at": datetime.now().isoformat(),
            "version": "0.1.0",
            "name": "unnamed",
            "description": "",
            "author": "",
        }

    def add_node(self, url: str, title: str = "", description: str = "", depth: int = 0):
        """Add or update a node."""
        normalized = self._normalize_url(url)
        if normalized not in self.nodes:
            self.nodes[normalized] = {
                "title": title,
                "description": description,
                "crawled_at": datetime.now().isoformat(),
                "depth": depth,
                "domain": urlparse(normalized).netloc,
            }
        else:
            # Update if we have better info
            if title and not self.nodes[normalized].get("title"):
                self.nodes[normalized]["title"] = title
            if description and not self.nodes[normalized].get("description"):
                self.nodes[normalized]["description"] = description

    def add_edge(self, from_url: str, to_url: str):
        """Add a directed edge (link) between two URLs."""
        from_norm = self._normalize_url(from_url)
        to_norm = self._normalize_url(to_url)
        self.edges[from_norm].add(to_norm)

    def add_seed(self, url: str):
        """Mark a URL as a seed."""
        self.seeds.add(self._normalize_url(url))

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove trailing slash, fragments, common tracking params
        path = parsed.path.rstrip("/") or "/"
        # Reconstruct without fragment
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def pagerank(self, damping: float = 0.95, iterations: int = 50,
                 personalized: bool = True) -> Dict[str, float]:
        """
        Compute PageRank over the local graph.

        When personalized=True (default), uses Personalized PageRank:
        the random walker teleports back to SEED nodes instead of
        uniformly to all nodes. This biases rankings toward your
        seeds' neighborhood — sites that your community links to
        rank higher than generically popular sites.

        Returns {url: score} sorted by score descending.
        Caches result — same params on same graph returns cached value.
        """
        # Simple memoization: cache result keyed by (damping, iterations, personalized)
        cache_key = (damping, iterations, personalized)
        if not hasattr(self, '_pagerank_cache'):
            self._pagerank_cache = {}
        if cache_key in self._pagerank_cache:
            return self._pagerank_cache[cache_key]

        all_urls = set(self.nodes.keys())
        if not all_urls:
            return {}

        n = len(all_urls)
        urls = list(all_urls)
        url_to_idx = {url: i for i, url in enumerate(urls)}

        # Build teleportation vector
        # Personalized: teleport to seeds. Standard: teleport uniformly.
        teleport = [0.0] * n
        if personalized and self.seeds:
            seed_indices = [url_to_idx[s] for s in self.seeds if s in url_to_idx]
            if seed_indices:
                seed_weight = 1.0 / len(seed_indices)
                for idx in seed_indices:
                    teleport[idx] = seed_weight
            else:
                # Fallback to uniform if no seeds in graph
                teleport = [1.0 / n] * n
        else:
            teleport = [1.0 / n] * n

        # Initialize uniform
        scores = [1.0 / n] * n

        for _ in range(iterations):
            new_scores = [(1 - damping) * teleport[j] for j in range(n)]

            for from_url, to_urls in self.edges.items():
                if from_url not in url_to_idx:
                    continue
                from_idx = url_to_idx[from_url]
                # Only count edges to nodes we know about
                valid_targets = [u for u in to_urls if u in url_to_idx]
                if not valid_targets:
                    # Dangling node: distribute to teleport targets
                    for j in range(n):
                        new_scores[j] += damping * scores[from_idx] * teleport[j]
                else:
                    share = damping * scores[from_idx] / len(valid_targets)
                    for to_url in valid_targets:
                        new_scores[url_to_idx[to_url]] += share

            scores = new_scores

        result = {urls[i]: scores[i] for i in range(n)}
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        self._pagerank_cache[cache_key] = result
        return result

    def discoveries(self, top_n: int = 20, damping: float = 0.95,
                     iterations: int = 50, use_quality: bool = True,
                     personalized: bool = True,
                     taste_model=None,
                     cfg: dict = None) -> List[Tuple[str, float, dict]]:
        """
        Return top-ranked pages that are NOT seeds.
        These are the things you discovered by crawling outward.

        The final score combines multiple signals:
        - PageRank (link authority, optionally personalized toward seeds)
        - Quality (HTML cleanliness: scripts, trackers, text ratio)
        - Smallweb score (data-driven: popularity bell curve × outlink profile)
        - Taste score (neural classifier trained on user feedback, optional)

        This means: a page needs good link authority AND clean HTML AND
        moderate popularity AND links to other small sites to rank highly.
        If a taste model is trained, it further weights by learned preferences.

        Args:
            top_n:        Number of discoveries to return
            damping:      PageRank damping factor (0.95 = follow links deep,
                          0.5 = stay close to seeds)
            iterations:   PageRank iterations (50 is usually plenty)
            use_quality:  Multiply by quality + smallweb scores
            personalized: Use personalized pagerank (biased toward seeds)
            taste_model:  Optional TasteModel instance for neural taste scoring
        """
        ranks = self.pagerank(damping=damping, iterations=iterations, personalized=personalized)

        # Build set of seed domains to filter out same-domain pages
        seed_domains = set()
        for seed in self.seeds:
            try:
                seed_domains.add(urlparse(seed).netloc.lower())
            except Exception:
                pass

        # Compute data-driven smallweb scores per domain
        sw_scores = self._smallweb_scores(cfg=cfg) if use_quality else {}

        # Normalize pagerank to percentiles (0-1) so it doesn't dominate
        # quality/smallweb signals. Raw pagerank spans ~4 orders of magnitude
        # while quality spans ~10x, so linear multiplication lets pagerank
        # completely drown out quality. Percentile normalization gives each
        # signal roughly equal weight in the final blend.
        if use_quality:
            sorted_pr = sorted(ranks.values())
            pr_rank_map = {}
            n_total = len(sorted_pr)
            for i, v in enumerate(sorted_pr):
                pr_rank_map[v] = (i + 1) / n_total  # percentile 0-1

        results = []
        for url, score in ranks.items():
            if url in self.seeds:
                continue
            if url not in self.nodes:
                continue
            try:
                domain = urlparse(url).netloc.lower()
                # Skip pages on the same domain as any seed (internal navigation)
                if domain in seed_domains:
                    continue
            except Exception:
                domain = ""

            # Always store raw pagerank for the API
            self.nodes[url]["pagerank"] = score

            if use_quality:
                # Percentile-normalized pagerank (0-1 range)
                pr_norm = pr_rank_map.get(score, 0.5)
                self.nodes[url]["pagerank_pct"] = round(pr_norm, 4)

                # Only fetched pages have a "quality" key (set during crawl).
                # With 35:1 unfetched-to-fetched ratio, unfetched pages flood
                # the rankings even at 0.5 quality. We apply a fetched bonus
                # (1.0 for measured pages, 0.2 for unmeasured) so that pages
                # we actually crawled and scored get priority.
                if "quality" in self.nodes[url]:
                    q = self.nodes[url]["quality"]
                    fetched_boost = 1.0
                else:
                    q = 0.5
                    f_cfg = (cfg or {}).get("formula", {})
                    fetched_boost = f_cfg.get("fetched_boost", 0.2)
                sw = sw_scores.get(domain, {}).get("smallweb_score", 0.5)

                # Blend with exponents from config.
                # Higher exponent = that signal matters more (amplifies differences).
                # Default 1.0 = linear (original behavior).
                f_cfg = (cfg or {}).get("formula", {})
                pr_exp = f_cfg.get("pagerank_exp", 1.0)
                q_exp = f_cfg.get("quality_exp", 1.0)
                sw_exp = f_cfg.get("smallweb_exp", 1.0)
                final_score = (pr_norm ** pr_exp) * (q ** q_exp) * (sw ** sw_exp) * fetched_boost

                # Store metadata on the node for API/frontend access
                sw_data = sw_scores.get(domain, {})
                self.nodes[url]["smallweb_score"] = sw
                self.nodes[url]["inbound_domains"] = sw_data.get("inbound_domains", 0)
                self.nodes[url]["outlink_score"] = sw_data.get("outlink_score", 0.5)
                self.nodes[url]["popularity_score"] = sw_data.get("popularity_score", 0.5)
                self.nodes[url]["taste_score"] = 0.5  # default neutral
            else:
                final_score = score

            results.append((url, final_score, self.nodes[url]))

        # First pass sort by graph signals
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply taste scoring to top candidates only (expensive: runs embeddings)
        # Only score 2x the requested results to keep it fast
        if taste_model and taste_model.is_trained:
            candidates = results[:top_n * 2]
            candidate_urls = [url for url, _, _ in candidates]
            taste_scores = taste_model.score_all(candidate_urls)

            # Re-score with taste
            f_cfg = (cfg or {}).get("formula", {})
            taste_base = f_cfg.get("taste_base", 0.5)
            taste_weight = f_cfg.get("taste_weight", 0.5)
            rescored = []
            for url, score, node in candidates:
                taste = taste_scores.get(url, 0.5)
                node["taste_score"] = taste
                new_score = score * (taste_base + taste_weight * taste)
                rescored.append((url, new_score, node))

            rescored.sort(key=lambda x: x[1], reverse=True)
            return rescored[:top_n]

        return results[:top_n]

    def _build_inbound_index(self) -> Dict[str, Set[str]]:
        """
        Build reverse index: domain -> set of domains that link TO it.
        Only counts cross-domain links (self-links within a domain don't count).
        """
        inbound: Dict[str, Set[str]] = defaultdict(set)
        for from_url, to_urls in self.edges.items():
            from_domain = urlparse(from_url).netloc.lower()
            for to_url in to_urls:
                to_domain = urlparse(to_url).netloc.lower()
                if from_domain != to_domain:  # external links only
                    inbound[to_domain].add(from_domain)
        return inbound

    def _build_outlink_profiles(self, graph_domains: Set[str] = None) -> Dict[str, dict]:
        """
        Pre-compute raw outlink data for ALL domains in one pass over edges.

        Returns dict of domain -> {
            ecosystem_fraction: float,   # fraction of outlinks pointing into graph
            non_platform_fraction: float, # fraction NOT pointing to platforms
            has_data: bool,              # whether domain has any outlinks
        }

        This is the expensive part (scans all edges once). The cheap part —
        applying config weights — happens in _outlink_score().
        """
        # One pass: build per-domain external target lists
        domain_targets = {}  # domain -> list of target domains
        for url, targets in self.edges.items():
            src_domain = urlparse(url).netloc.lower()
            if src_domain not in domain_targets:
                domain_targets[src_domain] = []
            for target in targets:
                td = urlparse(target).netloc.lower()
                if td != src_domain:
                    domain_targets[src_domain].append(td)

        # Compute fractions per domain
        profiles = {}
        for domain, ext_targets in domain_targets.items():
            if not ext_targets:
                profiles[domain] = {"ecosystem_fraction": 0.5, "non_platform_fraction": 0.5, "has_data": False}
                continue

            total = len(ext_targets)

            if graph_domains:
                in_graph = sum(1 for td in ext_targets if td in graph_domains)
                eco_frac = in_graph / total
            else:
                eco_frac = 0.5

            plat_count = sum(1 for td in ext_targets if is_platform_domain(td))
            non_plat_frac = 1.0 - (plat_count / total)

            profiles[domain] = {
                "ecosystem_fraction": round(eco_frac, 4),
                "non_platform_fraction": round(non_plat_frac, 4),
                "has_data": True,
            }

        return profiles

    def _outlink_score(self, domain: str, cfg: dict = None) -> float:
        """
        Get outlink profile score for a domain using cached raw data + config weights.
        Cheap: just a weighted sum of two pre-computed fractions.
        """
        if not hasattr(self, '_outlink_cache'):
            return 0.5  # cache not built yet
        profile = self._outlink_cache.get(domain, {"ecosystem_fraction": 0.5, "non_platform_fraction": 0.5, "has_data": False})
        if not profile["has_data"]:
            return 0.5

        sw_cfg = (cfg or {}).get("smallweb", {})
        eco_w = sw_cfg.get("ecosystem_weight", 0.7)
        plat_w = sw_cfg.get("platform_weight", 0.3)
        score = eco_w * profile["ecosystem_fraction"] + plat_w * profile["non_platform_fraction"]
        return round(score, 3)

    def _smallweb_scores(self, cfg: dict = None) -> Dict[str, dict]:
        """
        Compute per-domain "smallweb-ness" scores using data-driven signals.

        Returns dict of domain -> {
            inbound_domains: int,   # how many domains link to this one
            popularity_score: float, # bell curve peaking at moderate popularity
            outlink_score: float,    # how much it links into our ecosystem
            smallweb_score: float,   # combined score 0.0-1.0
        }

        Caches the inbound index and outlink profiles (expensive to compute).
        Config-dependent values (popularity curve, weights) are applied fresh.
        """
        # Cache the expensive parts: inbound index and raw outlink profiles
        if not hasattr(self, '_inbound_cache'):
            self._inbound_cache = self._build_inbound_index()
        inbound = self._inbound_cache

        # Build set of all domains in graph for ecosystem scoring
        scores = {}
        all_domains = set()
        for url in self.nodes:
            all_domains.add(urlparse(url).netloc.lower())

        if not all_domains:
            return {}

        # Cache outlink profiles (one-time full scan of edges)
        if not hasattr(self, '_outlink_cache'):
            self._outlink_cache = self._build_outlink_profiles(graph_domains=all_domains)

        for domain in all_domains:
            n_inbound = len(inbound.get(domain, set()))
            outlink = self._outlink_score(domain, cfg=cfg)

            # Popularity curve from config (bell curve peaking at moderate popularity)
            from scoring_config import popularity_score as _pop_score
            pop_score = _pop_score(cfg or {}, n_inbound)

            # Combined: popularity × outlink profile
            # Both signals matter: a site should be moderately popular
            # AND link to other small sites in our ecosystem
            sw_cfg = (cfg or {}).get("smallweb", {})
            mod = sw_cfg.get("outlink_moderation", 0.5)
            combined = pop_score * (mod + mod * outlink)  # outlink moderates, doesn't dominate

            # Safety net: if domain is a known platform, cap the score
            # This isn't the primary filter (data-driven signals are), but prevents
            # edge cases where a platform has just 3 inbound links in a small graph
            platform_cap = sw_cfg.get("platform_cap", 0.15)
            if is_platform_domain(domain):
                combined = min(combined, platform_cap)

            scores[domain] = {
                "inbound_domains": n_inbound,
                "popularity_score": round(pop_score, 3),
                "outlink_score": round(outlink, 3),
                "smallweb_score": round(combined, 3),
            }

        return scores

    def similar_sites(self, target: str, top_n: int = 20) -> List[Tuple[str, float, int]]:
        """
        Find sites similar to target using co-citation analysis.

        Two domains are similar if they're linked FROM the same sources.
        Uses cosine similarity on binary inbound-link vectors.

        Args:
            target: URL or domain to find similar sites for
            top_n:  Number of results to return

        Returns:
            List of (domain, cosine_similarity, shared_sources_count)
        """
        # Accept either URL or bare domain
        target_domain = urlparse(target).netloc.lower() if "://" in target else target.lower()

        if not hasattr(self, '_inbound_cache'):
            self._inbound_cache = self._build_inbound_index()
        inbound = self._inbound_cache
        target_sources = inbound.get(target_domain, set())
        if not target_sources:
            return []

        similarities = []
        for domain, sources in inbound.items():
            if domain == target_domain:
                continue
            intersection = len(target_sources & sources)
            if intersection == 0:
                continue
            cosine = intersection / (len(target_sources) ** 0.5 * len(sources) ** 0.5)
            similarities.append((domain, cosine, intersection))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    def all_similarities(self, min_shared: int = 2, top_n: int = 50) -> List[Tuple[str, str, float, int]]:
        """
        Find all pairs of similar domains in the graph.

        Args:
            min_shared: Minimum shared sources to include a pair
            top_n:      Maximum pairs to return

        Returns:
            List of (domain_a, domain_b, cosine_similarity, shared_count)
        """
        inbound = self._build_inbound_index()
        # Only consider domains with at least min_shared inbound sources
        domains = {d: s for d, s in inbound.items() if len(s) >= min_shared}
        domain_list = list(domains.keys())

        pairs = []
        for i in range(len(domain_list)):
            for j in range(i + 1, len(domain_list)):
                d_a, d_b = domain_list[i], domain_list[j]
                intersection = len(domains[d_a] & domains[d_b])
                if intersection < min_shared:
                    continue
                cosine = intersection / (len(domains[d_a]) ** 0.5 * len(domains[d_b]) ** 0.5)
                pairs.append((d_a, d_b, cosine, intersection))

        return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]

    def domains(self) -> Dict[str, int]:
        """Count pages per domain."""
        counts = defaultdict(int)
        for url in self.nodes:
            domain = urlparse(url).netloc
            counts[domain] += 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def stats(self) -> dict:
        """Return graph statistics."""
        total_edges = sum(len(targets) for targets in self.edges.values())
        return {
            "nodes": len(self.nodes),
            "edges": total_edges,
            "seeds": len(self.seeds),
            "domains": len(self.domains()),
            "avg_outlinks": total_edges / max(len(self.edges), 1),
            "created_at": self.metadata.get("created_at", "unknown"),
            "name": self.metadata.get("name", "unnamed"),
        }

    # ── Serialization ──

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "metadata": self.metadata,
            "seeds": list(self.seeds),
            "nodes": self.nodes,
            "edges": {k: list(v) for k, v in self.edges.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "WebGraph":
        """Deserialize from JSON dict."""
        graph = cls()
        graph.metadata = data.get("metadata", graph.metadata)
        graph.seeds = set(data.get("seeds", []))
        graph.nodes = data.get("nodes", {})
        graph.edges = defaultdict(set, {k: set(v) for k, v in data.get("edges", {}).items()})
        return graph

    def save(self, path: str):
        """Save graph to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        print(f"saved graph to {path} ({len(self.nodes)} nodes, {sum(len(v) for v in self.edges.values())} edges)")

    @classmethod
    def load(cls, path: str) -> "WebGraph":
        """Load graph from JSON file."""
        with open(path) as f:
            return cls.from_json(json.load(f))

    # ── Fork & Merge ──

    def fork(self, name: str = "", author: str = "",
             add_seeds: Optional[List[str]] = None,
             promote_top_n: int = 0) -> "WebGraph":
        """
        Create a copy of this graph (fork it), optionally with new seeds.

        Args:
            name:           Name for the forked graph
            author:         Author of the fork
            add_seeds:      Additional URLs to add as seeds in the fork.
                            Use this to "promote" discovered URLs into seeds
                            for a deeper re-crawl.
            promote_top_n:  Auto-promote the top N discoveries to seeds.
                            e.g. promote_top_n=10 adds the 10 highest-ranked
                            non-seed pages as seeds in the fork.
        """
        data = self.to_json()
        forked = WebGraph.from_json(data)
        forked.metadata["forked_from"] = self.metadata.get("name", "unknown")
        forked.metadata["forked_at"] = datetime.now().isoformat()
        forked.metadata["created_at"] = datetime.now().isoformat()
        if name:
            forked.metadata["name"] = name
        if author:
            forked.metadata["author"] = author

        # Add explicit seed URLs
        if add_seeds:
            for url in add_seeds:
                if not url.startswith("http"):
                    url = "https://" + url
                forked.add_seed(url)
            forked.metadata["seeds_added"] = add_seeds

        # Auto-promote top discoveries to seeds
        if promote_top_n > 0:
            discoveries = self.discoveries(top_n=promote_top_n)
            promoted = []
            for url, score, node in discoveries:
                forked.add_seed(url)
                promoted.append(url)
            forked.metadata["seeds_promoted"] = promoted
            forked.metadata["seeds_promoted_count"] = len(promoted)

        return forked

    @staticmethod
    def merge(graph_a: "WebGraph", graph_b: "WebGraph", name: str = "") -> "WebGraph":
        """Merge two graphs. Union of nodes and edges."""
        merged = WebGraph()
        merged.metadata["name"] = name or f"merge of {graph_a.metadata.get('name', 'a')} + {graph_b.metadata.get('name', 'b')}"
        merged.metadata["merged_from"] = [
            graph_a.metadata.get("name", "unknown"),
            graph_b.metadata.get("name", "unknown"),
        ]
        merged.metadata["created_at"] = datetime.now().isoformat()

        # Union nodes
        for url, data in graph_a.nodes.items():
            merged.nodes[url] = data.copy()
        for url, data in graph_b.nodes.items():
            if url not in merged.nodes:
                merged.nodes[url] = data.copy()

        # Union edges
        for from_url, to_urls in graph_a.edges.items():
            merged.edges[from_url] |= to_urls
        for from_url, to_urls in graph_b.edges.items():
            merged.edges[from_url] |= to_urls

        # Union seeds
        merged.seeds = graph_a.seeds | graph_b.seeds

        return merged


# ── Crawler ───────────────────────────────────────────────────────────

# Domains/paths to skip
SKIP_DOMAINS = {
    "google.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "linkedin.com", "amazon.com", "apple.com", "microsoft.com",
    "reddit.com", "tiktok.com", "pinterest.com", "tumblr.com",
    "fonts.googleapis.com", "cdn.jsdelivr.net", "unpkg.com",
    "w3.org", "schema.org", "creativecommons.org",
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".zip", ".tar", ".gz", ".mp3", ".mp4", ".avi",
    ".xml", ".rss", ".atom",
}


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Skip non-http
        if parsed.scheme not in ("http", "https"):
            return True

        # Skip big tech / CDNs
        for skip in SKIP_DOMAINS:
            if domain == skip or domain.endswith("." + skip):
                return True

        # Skip file extensions
        path_lower = parsed.path.lower()
        for ext in SKIP_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        return False
    except:
        return True


async def fetch_page(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> Optional[Tuple[str, str]]:
    """Fetch a page and return (html, final_url) or None."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout),
                               allow_redirects=True,
                               headers={"User-Agent": "smallweb/0.1 (niche web discovery)"}) as resp:
            if resp.status != 200:
                return None
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
            html = await resp.text(errors="replace")
            return html, str(resp.url)
    except Exception:
        return None


def extract_links_and_meta(html: str, base_url: str) -> Tuple[str, str, List[str], Dict[str, List[str]]]:
    """
    Extract title, description, outgoing links, and anchor texts from HTML.

    Returns:
        (title, description, links, anchor_texts)
        where anchor_texts is {target_url: [list of anchor text strings]}
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()[:200]

    # Description
    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()[:300]

    # Links + anchor text
    links = []
    anchor_texts: Dict[str, List[str]] = defaultdict(list)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        try:
            absolute = urljoin(base_url, href)
            if not should_skip_url(absolute):
                links.append(absolute)
                # Capture anchor text — what this page calls the linked page
                text = a.get_text(strip=True)[:100]
                if text and text.lower() not in ("click here", "here", "link", "read more", "more", "→", "»"):
                    anchor_texts[absolute].append(text)
        except:
            continue

    return title, description, links, anchor_texts


def quality_score(html: str, soup: BeautifulSoup, cfg: dict = None) -> float:
    """
    Compute a quality score for a page (0.0 to 1.0).

    Ported from Marginalia's DocumentValuator + FeatureExtractor approach.
    Uses an additive penalty/bonus system on a log text-to-html ratio base.

    Returns a float between 0.0 (spam) and 1.0 (clean content).
    Also stores a detailed breakdown on the function attribute `last_breakdown`.
    """
    breakdown = quality_breakdown(html, soup, cfg=cfg)
    quality_score.last_breakdown = breakdown
    return breakdown["score"]


def quality_breakdown(html: str, soup: BeautifulSoup, cfg: dict = None) -> dict:
    """
    Compute quality score with full breakdown of every signal that fired.

    Returns a dict with:
      - score: final 0-1 score
      - base: text ratio base score
      - penalties: list of {signal, value, detail, why}
      - bonuses: list of {signal, value, detail, why}
      - text_ratio: raw text/html ratio

    cfg: scoring config dict (from scoring_config.py). Uses defaults if None.
    """
    import math
    if cfg is None:
        from scoring_config import DEFAULTS
        cfg = DEFAULTS
    qp = cfg.get("quality_penalties", {})
    qb = cfg.get("quality_bonuses", {})

    text = soup.get_text()
    text_len = len(text)
    html_len = max(len(html), 1)
    html_lower = html.lower()

    # ── Base score: text-to-html ratio (Marginalia's core signal) ──
    text_ratio = text_len / html_len
    base = (math.log(text_ratio + 0.001) + 5) / 7
    base = max(0.1, min(1.0, base))

    penalties = []
    bonuses = []

    # --- Script analysis ---
    scripts = soup.find_all("script")
    inline_scripts = 0
    external_scripts = 0
    dynamic_injection = 0

    TRACKER_DOMAINS = {
        "google-analytics.com", "googletagmanager.com", "googlesyndication.com",
        "googleadservices.com", "google.com/adsense",
        "facebook.net", "facebook.com/tr", "connect.facebook.net",
        "doubleclick.net", "amazon-adsystem.com", "adnxs.com",
        "hotjar.com", "segment.com", "mixpanel.com", "amplitude.com",
        "optimizely.com", "crazyegg.com", "hubspot.com", "pardot.com",
        "marketo.net", "newrelic.com", "nr-data.net",
        "quantserve.com", "scorecardresearch.com", "chartbeat.com",
        "omtrdc.net", "demdex.net",
        "adsrvr.org", "criteo.com", "taboola.com", "outbrain.com",
        "sharethrough.com", "pubmatic.com", "rubiconproject.com",
        "openx.net", "bidswitch.net",
    }

    ADTECH_KEYWORDS = {"adsystem", "adsrvr", "adnxs", "doubleclick",
                       "syndication", "criteo", "taboola", "outbrain",
                       "pubmatic", "rubicon", "openx", "bidswitch"}

    tracker_count = 0
    adtech_count = 0
    trackers_found = []
    adtech_found = []

    for s in scripts:
        src = s.get("src", "")
        if src:
            external_scripts += 1
            src_lower = src.lower()
            for tracker in TRACKER_DOMAINS:
                if tracker in src_lower:
                    if any(ad in tracker for ad in ADTECH_KEYWORDS):
                        adtech_count += 1
                        adtech_found.append(tracker)
                    else:
                        tracker_count += 1
                        trackers_found.append(tracker)
                    break
        else:
            inline_scripts += 1
            script_text = s.string or ""
            if "createElement(" in script_text or "createElement (" in script_text:
                dynamic_injection += 1

    if external_scripts:
        penalties.append({
            "signal": "external_scripts",
            "value": round(external_scripts * qp.get("external_scripts", 0.08), 3),
            "detail": f"{external_scripts} external scripts",
            "why": "Each external script adds page weight, slows loading, and often loads third-party code. Clean personal sites rarely need more than 1-2."
        })
    if inline_scripts:
        penalties.append({
            "signal": "inline_scripts",
            "value": round(inline_scripts * qp.get("inline_scripts", 0.02), 3),
            "detail": f"{inline_scripts} inline scripts",
            "why": "Small penalty per inline script. Some are fine (progressive enhancement), but dozens suggest framework overhead."
        })
    if dynamic_injection:
        penalties.append({
            "signal": "dynamic_injection",
            "value": round(dynamic_injection * qp.get("dynamic_injection", 0.1), 3),
            "detail": f"{dynamic_injection} createElement() calls",
            "why": "Dynamically injecting scripts usually means loading trackers or ads that try to evade content blockers."
        })
    if tracker_count:
        penalties.append({
            "signal": "trackers",
            "value": round(tracker_count * qp.get("trackers", 0.15), 3),
            "detail": f"{tracker_count} trackers ({', '.join(trackers_found[:3])})",
            "why": "Analytics and tracking scripts (Google Analytics, Hotjar, etc.) indicate a site that monitors visitors. Independent sites tend to skip these."
        })
    if adtech_count:
        penalties.append({
            "signal": "adtech",
            "value": round(adtech_count * qp.get("adtech", 0.2), 3),
            "detail": f"{adtech_count} ad-tech ({', '.join(adtech_found[:3])})",
            "why": "Ad networks (DoubleClick, Criteo, Taboola) are the strongest commercial signal. Sites running programmatic ads are almost never small/indie web."
        })

    # --- Cookie consent / GDPR banners ---
    CONSENT_SIGNALS = [
        "cookielaw", "onetrust", "cookieconsent", "cookie-consent",
        "gdpr", "cookie-banner", "cookiebanner", "didomi",
        "quantcast.mgr", "sp_data", "consent-manager",
    ]
    consent_found = [sig for sig in CONSENT_SIGNALS if sig in html_lower]
    consent_penalty = min(len(consent_found) * qp.get("cookie_consent", 0.1), qp.get("cookie_consent_max", 0.3))
    if consent_found:
        penalties.append({
            "signal": "cookie_consent",
            "value": round(consent_penalty, 3),
            "detail": f"consent banners ({', '.join(consent_found[:3])})",
            "why": "Cookie consent managers indicate commercial data collection. If a site needs a GDPR banner, it's tracking you enough to require legal notice."
        })

    # --- Affiliate links ---
    affiliate_patterns = ["amzn.to/", "tag=", "affiliate", "ref=", "partner="]
    all_links = soup.find_all("a", href=True)
    affiliate_count = sum(1 for a in all_links
                         if any(p in (a.get("href", "").lower()) for p in affiliate_patterns))
    affiliate_penalty = min(affiliate_count * qp.get("affiliate_links", 0.05), qp.get("affiliate_links_max", 0.25))
    if affiliate_count:
        penalties.append({
            "signal": "affiliate_links",
            "value": round(affiliate_penalty, 3),
            "detail": f"{affiliate_count} affiliate links",
            "why": "Affiliate links (Amazon, partner programs) suggest the content may be commercially motivated rather than purely informational."
        })

    # --- AI content farm signals ---
    headings = soup.find_all(["h1", "h2", "h3", "h4"])
    heading_texts = [h.get_text().lower().strip() for h in headings]
    ai_signals = 0
    ai_matches = []
    AI_HEADING_PATTERNS = [
        "benefits of", "key benefits", "key takeaways", "in conclusion",
        "frequently asked questions", "what you need to know",
        "everything you need to know", "ultimate guide",
        "step-by-step guide", "pros and cons",
    ]
    for ht in heading_texts:
        for pattern in AI_HEADING_PATTERNS:
            if pattern in ht:
                ai_signals += 1
                ai_matches.append(pattern)
                break
    ai_threshold = int(qp.get("ai_content_threshold", 3))
    if ai_signals >= ai_threshold:
        penalties.append({
            "signal": "ai_content",
            "value": qp.get("ai_content_high", 0.5),
            "detail": f"{ai_signals} AI-pattern headings ({', '.join(ai_matches[:3])})",
            "why": "Multiple generic headings like 'Key Takeaways' and 'Ultimate Guide' are strong signals of AI-generated content farm pages."
        })
    elif ai_signals >= 1:
        penalties.append({
            "signal": "ai_content",
            "value": qp.get("ai_content_low", 0.15),
            "detail": f"{ai_signals} AI-pattern heading ({', '.join(ai_matches[:2])})",
            "why": "Generic headings sometimes appear in real content, but they correlate with AI-generated or SEO-optimized pages."
        })

    # --- Link density ---
    text_words = max(len(text.split()), 1)
    link_density = len(all_links) / text_words
    if link_density > qp.get("link_farm_threshold", 0.5):
        penalties.append({
            "signal": "link_farm",
            "value": qp.get("link_farm", 0.4),
            "detail": f"link density {link_density:.2f} (>{qp.get('link_farm_threshold', 0.5)})",
            "why": "When more than half the words are links, the page is likely a directory, blogroll, or link farm rather than original content."
        })
    elif link_density > qp.get("link_density_threshold", 0.3):
        penalties.append({
            "signal": "link_density",
            "value": qp.get("link_density", 0.2),
            "detail": f"link density {link_density:.2f} (>0.3)",
            "why": "High link-to-text ratio suggests a page focused on navigation rather than content. Some link density is fine, but this is above average."
        })

    # --- Ad-block key ---
    if "data-adblockkey" in html_lower or "x-adblock-key" in html_lower:
        penalties.append({
            "signal": "adblock_key",
            "value": qp.get("adblock_key", 0.6),
            "detail": "adblock circumvention key detected",
            "why": "The Acceptable Ads / adblock-key meta tag is used almost exclusively by domain squatters and aggressive ad sites. Per Marginalia: 'only domain squatters use this.'"
        })

    # ── Bonuses ──
    if 'rel="webmention"' in html_lower or "webmention" in html_lower:
        bonuses.append({
            "signal": "webmention",
            "value": qb.get("webmention", 0.1),
            "detail": "supports Webmention",
            "why": "Webmention is an IndieWeb standard for cross-site conversations. Indicates the author participates in the decentralized web community."
        })
    if 'rel="indieauth"' in html_lower or "indieauth" in html_lower:
        bonuses.append({
            "signal": "indieauth",
            "value": qb.get("indieauth", 0.1),
            "detail": "supports IndieAuth",
            "why": "IndieAuth lets you sign in with your own domain. Strong signal of someone who owns their identity on the web."
        })
    if 'type="application/rss+xml"' in html_lower or 'type="application/atom+xml"' in html_lower:
        bonuses.append({
            "signal": "rss_feed",
            "value": qb.get("rss_feed", 0.05),
            "detail": "has RSS/Atom feed",
            "why": "Offering an RSS feed means the author cares about open, follow-able content outside of walled-garden platforms."
        })
    if "h-card" in html_lower or "h-entry" in html_lower:
        bonuses.append({
            "signal": "microformats",
            "value": qb.get("microformats", 0.05),
            "detail": "uses microformats (h-card/h-entry)",
            "why": "Microformats are machine-readable markup used by IndieWeb tools. Indicates a technically engaged author building for the open web."
        })

    # ── Final score ──
    total_penalty = sum(p["value"] for p in penalties)
    total_bonus = sum(b["value"] for b in bonuses)
    score = base - total_penalty + total_bonus
    score = round(max(0.0, min(1.0, score)), 3)

    return {
        "score": score,
        "base": round(base, 3),
        "text_ratio": round(text_ratio, 4),
        "total_penalty": round(total_penalty, 3),
        "total_bonus": round(total_bonus, 3),
        "penalties": penalties,
        "bonuses": bonuses,
    }


# ── Platform / Big-site detection ─────────────────────────────────────
# Marginalia's key insight: the "small web" is defined by what it's NOT.
# Major platforms, news sites, e-commerce, social media, and developer
# infrastructure sites are not discoveries — they're the background noise
# that the small web exists in contrast to.

# Domains that are never interesting as discoveries.
# These are platforms, not content. Finding "github.com" or "discord.gg"
# in your crawl results is like finding "google.com" — it tells you nothing.
PLATFORM_DOMAINS = {
    # Social media & messaging
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "linkedin.com", "pinterest.com", "snapchat.com", "threads.net",
    "discord.gg", "discord.com", "t.me", "telegram.org", "t.co",
    "mastodon.social", "joinmastodon.org",
    # Code hosting (the platform, not user content)
    "github.com", "gitlab.com", "codeberg.org", "bitbucket.org",
    "gist.github.com", "raw.githubusercontent.com",
    # Big tech
    "google.com", "apple.com", "microsoft.com", "amazon.com",
    "docs.google.com", "drive.google.com", "play.google.com",
    "apps.apple.com", "support.apple.com",
    # Publishing platforms (the platform itself)
    "wordpress.org", "wordpress.com", "squarespace.com", "wix.com",
    "ghost.org", "webflow.com", "carrd.co",
    "medium.com", "substack.com", "blogger.com",
    # Video/media platforms
    "youtube.com", "youtu.be", "vimeo.com", "spotify.com",
    "soundcloud.com", "twitch.tv",
    # Commerce
    "store.steampowered.com", "steampowered.com", "amazon.com",
    "ebay.com", "etsy.com", "shopify.com",
    # Developer tools/infrastructure
    "npmjs.com", "npm.js.com", "pypi.org", "crates.io",
    "stackoverflow.com", "stackexchange.com",
    "gitbook.com", "www.gitbook.com", "readthedocs.org",
    "netlify.com", "vercel.com", "heroku.com", "cloudflare.com",
    # Major news/media (not small web)
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "bbc.co.uk", "bbc.com", "cnn.com", "reuters.com",
    "bloomberg.com", "forbes.com", "techcrunch.com",
    "wired.com", "arstechnica.com", "theverge.com",
    "vice.com", "buzzfeed.com", "huffpost.com",
    "metro.co.uk",
    # Reference (not discoveries)
    "en.wikipedia.org", "wikipedia.org", "www.wikipedia.org", "wikidata.org",
    "wikimedia.org", "archive.org", "web.archive.org",
    # Education mega-sites
    "www.khanacademy.org", "khanacademy.org",
    # Misc platforms
    "reddit.com", "news.ycombinator.com", "lobste.rs",
    "patreon.com", "ko-fi.com", "buymeacoffee.com",
    "anchor.fm", "gumroad.com",
    "notion.so", "notion.site", "airtable.com",
    "figma.com", "www.figma.com", "canva.com",
}

# Patterns that indicate a platform subdomain (e.g. *.github.io is fine,
# but github.com itself is not)
PLATFORM_PATTERNS = [
    ".slack.com",
    ".atlassian.net",
    ".zendesk.com",
    ".salesforce.com",
    ".sharepoint.com",
    ".wikipedia.org",
    ".wikimedia.org",
    ".wiktionary.org",
    ".wikibooks.org",
]


def is_platform_domain(domain: str) -> bool:
    """Check if a domain is a major platform (not small web)."""
    domain = domain.lower()
    # Exact match
    if domain in PLATFORM_DOMAINS:
        return True
    # www. prefix
    if domain.startswith("www.") and domain[4:] in PLATFORM_DOMAINS:
        return True
    # Without www.
    if f"www.{domain}" in PLATFORM_DOMAINS:
        return True
    # Pattern match
    for pattern in PLATFORM_PATTERNS:
        if domain.endswith(pattern):
            return True
    return False


import re

# URL spam patterns — from Marginalia's UrlBlocklist + our own additions.
# These catch link farms, download bait, git repos, and auto-generated content.
_URL_SPAM_PATTERNS = [
    re.compile(r'\.git/'),                          # Git repository paths
    re.compile(r'wp-content/upload'),               # WordPress upload dirs
    re.compile(r'-download-free'),                  # Download bait
    re.compile(r'/download/.+/download'),           # Download farm loops
    re.compile(r'/permalink/'),                     # Permalink farms
    re.compile(r'[0-9a-f]{32,}'),                   # Long hex strings (git hashes, session IDs)
    re.compile(r'/tag/|/tags/|/category/|/categories/'),  # Taxonomy pages (low content)
    re.compile(r'/page/\d+/?$'),                    # Paginated listing pages
    re.compile(r'/feed/?$|/rss/?$|/atom\.xml'),     # Feed URLs (not content)
    re.compile(r'\.(pdf|zip|tar|gz|exe|dmg|pkg|deb|rpm|msi|iso)$', re.I),  # Binary files
    re.compile(r'\.(png|jpg|jpeg|gif|svg|webp|ico|mp3|mp4|wav|avi|mov)$', re.I),  # Media files
    re.compile(r'/wp-json/'),                       # WordPress API endpoints
    re.compile(r'/xmlrpc\.php'),                    # WordPress XML-RPC
    re.compile(r'/api/|/graphql'),                  # API endpoints
    re.compile(r'/(login|signin|signup|register|auth|oauth)/?', re.I),  # Auth pages
    re.compile(r'/search\?|/search/\?'),            # Search result pages
    re.compile(r'[?&](utm_|fbclid|gclid|ref=)'),   # Tracking parameters
]


def is_spam_url(url: str) -> bool:
    """Check if a URL matches known spam/low-value patterns."""
    for pattern in _URL_SPAM_PATTERNS:
        if pattern.search(url):
            return True
    return False


def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    return urlparse(url).netloc.lower()


def _prioritize_queue(queue: List[Tuple[str, int]], domain_counts: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Sort the crawl queue to prefer URLs from new/underrepresented domains.

    This is the "domain diversity" heuristic: instead of pure BFS order,
    we bump up URLs from domains we haven't seen much yet. This encourages
    the crawler to explore broadly across many sites rather than going deep
    into a single domain like Wikipedia.

    URLs are sorted by: (domain_count, depth, original_position)
    - domain_count: domains we've crawled less come first
    - depth: shallower pages come first (preserve BFS-ish behavior)
    - original_position: break ties by insertion order
    """
    return sorted(
        queue,
        key=lambda item: (
            domain_counts.get(_get_domain(item[0]), 0),  # fewer pages from this domain = higher priority
            item[1],  # shallower depth first
        )
    )


async def crawl(seeds: List[str], max_hops: int = 2, max_pages: int = 200,
                concurrent: int = 10, name: str = "unnamed",
                domain_cap: int = 20) -> WebGraph:
    """
    Crawl outward from seed URLs, building a local web graph.

    Args:
        seeds:      Starting URLs to crawl from
        max_hops:   Maximum link-distance from seeds (default: 2)
        max_pages:  Maximum total pages to crawl (default: 200)
        concurrent: Number of concurrent HTTP requests (default: 10)
        name:       Name for this graph
        domain_cap: Maximum pages to crawl per domain (default: 20).
                    Prevents any single site (e.g. Wikipedia) from
                    consuming the entire crawl budget. Set to 0 to disable.
    """
    graph = WebGraph()
    graph.metadata["name"] = name
    graph.metadata["domain_cap"] = domain_cap

    # Normalize and add seeds
    for url in seeds:
        if not url.startswith("http"):
            url = "https://" + url
        graph.add_seed(url)
        graph.add_node(url, depth=0)

    # BFS crawl with domain-aware prioritization
    queue: List[Tuple[str, int]] = [(url, 0) for url in graph.seeds]  # (url, depth)
    visited: Set[str] = set()
    domain_counts: Dict[str, int] = defaultdict(int)  # domain -> pages crawled from it
    domain_outlink_diversity: Dict[str, Set[str]] = defaultdict(set)  # domain -> set of domains it links to
    seed_domains = {_get_domain(url) for url in graph.seeds}
    pages_crawled = 0

    connector = aiohttp.TCPConnector(limit=concurrent, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        while queue and pages_crawled < max_pages:
            # Re-sort queue to prefer underrepresented domains.
            # This is the key diversity mechanism: instead of pure FIFO,
            # we pull from domains we've seen least.
            queue = _prioritize_queue(queue, domain_counts)

            # Take a batch from the front of the prioritized queue
            batch = []
            skip_indices = []
            for i, (url, depth) in enumerate(queue):
                if len(batch) >= concurrent:
                    break
                normalized = graph._normalize_url(url)
                domain = _get_domain(url)

                # Skip already-visited URLs
                if normalized in visited:
                    skip_indices.append(i)
                    continue

                # Skip if beyond max hops
                if depth > max_hops:
                    skip_indices.append(i)
                    continue

                # Adaptive per-domain budget (inspired by Marginalia)
                # Seed domains get 2.5x the base cap (user explicitly chose them).
                # Domains that link to many diverse sites get a bonus (they're hubs).
                # Other domains use the base cap.
                if domain_cap > 0:
                    if domain in seed_domains:
                        effective_cap = int(domain_cap * 2.5)
                    else:
                        # Bonus for domains that link to diverse sites in our graph
                        diversity = len(domain_outlink_diversity.get(domain, set()))
                        if diversity >= 5:
                            effective_cap = int(domain_cap * 1.5)
                        else:
                            effective_cap = domain_cap
                    if domain_counts[domain] >= effective_cap:
                        skip_indices.append(i)
                        continue

                visited.add(normalized)
                skip_indices.append(i)
                batch.append((url, depth))

            # Remove processed entries from queue (in reverse to preserve indices)
            for i in sorted(skip_indices, reverse=True):
                queue.pop(i)

            if not batch:
                break

            # Fetch batch concurrently
            tasks = [fetch_page(session, url) for url, _ in batch]
            results = await asyncio.gather(*tasks)

            for (url, depth), result in zip(batch, results):
                if result is None:
                    continue

                html, final_url = result
                pages_crawled += 1
                domain = _get_domain(final_url)
                domain_counts[domain] += 1

                title, description, links, anchors = extract_links_and_meta(html, final_url)

                # Compute quality score with full breakdown
                soup = BeautifulSoup(html, "html.parser")
                q_score = quality_score(html, soup)
                q_breakdown = quality_score.last_breakdown

                graph.add_node(final_url, title=title, description=description, depth=depth)
                norm_url = graph._normalize_url(final_url)
                graph.nodes[norm_url]["quality"] = q_score
                graph.nodes[norm_url]["quality_breakdown"] = q_breakdown

                # Add edges, queue new URLs, and store anchor texts
                for link in links:
                    graph.add_edge(final_url, link)
                    link_norm = graph._normalize_url(link)
                    # Track outlink diversity for adaptive budgets
                    link_domain = _get_domain(link)
                    if link_domain != domain:
                        domain_outlink_diversity[domain].add(link_domain)

                    # Store anchor texts on the target node
                    if link in anchors:
                        if link_norm not in graph.nodes:
                            graph.add_node(link, depth=depth + 1)
                        node = graph.nodes[link_norm]
                        if "anchor_texts" not in node:
                            node["anchor_texts"] = []
                        for text in anchors[link]:
                            if text not in node["anchor_texts"]:
                                node["anchor_texts"].append(text)
                        # Keep list bounded
                        node["anchor_texts"] = node["anchor_texts"][:20]

                    if link_norm not in visited and depth + 1 <= max_hops:
                        # Don't waste crawl budget on platform domains
                        link_domain = _get_domain(link)
                        if is_platform_domain(link_domain):
                            continue
                        # Skip spam/low-value URL patterns
                        if is_spam_url(link):
                            continue
                        if link_norm not in graph.nodes:
                            graph.add_node(link, depth=depth + 1)
                        queue.append((link, depth + 1))

                # Show progress with domain count
                if domain_cap > 0:
                    eff_cap = int(domain_cap * 2.5) if domain in seed_domains else (
                        int(domain_cap * 1.5) if len(domain_outlink_diversity.get(domain, set())) >= 5
                        else domain_cap)
                    seed_tag = "★" if domain in seed_domains else ""
                    cap_info = f" [{domain}: {domain_counts[domain]}/{eff_cap}{seed_tag}]"
                else:
                    cap_info = ""
                status = f"[{pages_crawled}/{max_pages}] depth={depth}{cap_info} {title[:40] or final_url[:40]}"
                print(f"  {status}", flush=True)

    # Log domain distribution summary
    print(f"\ndomain distribution ({len(domain_counts)} domains):")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        capped = " (capped)" if domain_cap > 0 and count >= domain_cap else ""
        print(f"  {count:4d}  {domain}{capped}")

    return graph


# ── Web UI ────────────────────────────────────────────────────────────

SERVE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>smallweb - $name</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Berkeley Mono', 'IBM Plex Mono', monospace; background: #0a0a0a; color: #e0e0e0; padding: 2rem; max-width: 960px; margin: 0 auto; }
  a { color: #4fc3f7; text-decoration: none; }
  a:hover { text-decoration: underline; }

  h1 { color: #fff; margin-bottom: 0.3rem; font-size: 1.4rem; }
  h1 .back { color: #555; font-size: 0.85rem; margin-right: 0.5rem; text-decoration: none; }
  h1 .back:hover { color: #4fc3f7; }
  .meta { color: #555; font-size: 0.8rem; margin-bottom: 2rem; }

  /* Stats row */
  .stats { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
  .stat { background: #111; padding: 0.8rem 1rem; border-radius: 8px; border: 1px solid #1a1a1a; }
  .stat-value { font-size: 1.3rem; color: #fff; font-weight: 600; }
  .stat-label { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }

  /* Tabs */
  .tabs { display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 1px solid #222; }
  .tab { padding: 0.6rem 1.2rem; font-size: 0.85rem; color: #666; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; font-family: inherit; background: none; border-top: none; border-left: none; border-right: none; }
  .tab:hover { color: #aaa; }
  .tab.active { color: #4fc3f7; border-bottom-color: #4fc3f7; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Controls bar */
  .controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.2rem; flex-wrap: wrap; }
  .toggle-group { display: flex; background: #151515; border: 1px solid #222; border-radius: 6px; overflow: hidden; }
  .toggle-btn { padding: 0.4rem 0.8rem; font-size: 0.75rem; color: #666; cursor: pointer; border: none; background: none; font-family: inherit; transition: all 0.2s; }
  .toggle-btn.active { background: #1a3050; color: #4fc3f7; }
  .toggle-btn:hover:not(.active) { color: #aaa; }
  .control-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 0.04em; }

  /* Section headers */
  .section { margin-bottom: 2rem; }
  .section h2 { font-size: 0.9rem; color: #888; margin-bottom: 1rem; border-bottom: 1px solid #1a1a1a; padding-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }

  /* Discovery cards */
  .discovery { padding: 0.8rem; margin-bottom: 0.5rem; border: 1px solid #1a1a1a; border-radius: 8px; transition: border-color 0.2s; position: relative; }
  .discovery:hover { border-color: #333; background: #0d0d0d; }
  .discovery-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
  .discovery-main { flex: 1; min-width: 0; }
  .discovery-title { color: #4fc3f7; text-decoration: none; font-size: 0.9rem; display: block; }
  .discovery-title:hover { text-decoration: underline; }
  .discovery-url { color: #444; font-size: 0.7rem; margin-top: 0.15rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .discovery-desc { color: #777; font-size: 0.78rem; margin-top: 0.4rem; line-height: 1.4; }

  /* Quality bar */
  .quality-bar-wrap { display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }
  .quality-bar { width: 60px; height: 6px; background: #1a1a1a; border-radius: 3px; overflow: hidden; }
  .quality-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .quality-label { font-size: 0.65rem; color: #555; min-width: 28px; text-align: right; }

  /* Anchor text pills */
  .anchor-texts { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem; }
  .anchor-pill { background: #151520; border: 1px solid #252535; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.68rem; color: #8888bb; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Score badge */
  .score-badge { font-size: 0.65rem; color: #333; font-weight: 600; white-space: nowrap; }
  .why-toggle { cursor: pointer; color: #4a6a8a; text-decoration: none; }
  .why-toggle:hover { color: #4fc3f7; }

  /* Breakdown panel */
  .breakdown { background: #080808; border-top: 1px solid #1a1a1a; padding: 0.8rem; margin-top: 0.5rem; font-size: 0.72rem; line-height: 1.5; }
  .breakdown-section { margin-bottom: 0.8rem; }
  .breakdown-section:last-child { margin-bottom: 0; }
  .breakdown-title { color: #4fc3f7; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; font-weight: 600; }
  .breakdown-subtitle { color: #888; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.04em; margin: 0.5rem 0 0.3rem; }
  .breakdown-formula { font-family: 'Berkeley Mono', monospace; color: #b0b0b0; background: #0d0d0d; padding: 0.3rem 0.5rem; border-radius: 4px; margin-bottom: 0.5rem; font-size: 0.68rem; }
  .breakdown-components { display: flex; flex-direction: column; gap: 0.3rem; }
  .breakdown-row { display: grid; grid-template-columns: 100px 50px 1fr; gap: 0.4rem; align-items: start; padding: 0.2rem 0; border-bottom: 1px solid #111; }
  .breakdown-row.penalty .breakdown-val { color: #ef5350; }
  .breakdown-row.bonus .breakdown-val { color: #81c784; }
  .breakdown-key { color: #888; font-family: 'Berkeley Mono', monospace; font-size: 0.65rem; }
  .breakdown-val { color: #ddd; font-family: 'Berkeley Mono', monospace; font-size: 0.65rem; font-weight: 600; text-align: right; }
  .breakdown-detail { color: #999; font-size: 0.65rem; }
  .breakdown-why { color: #555; font-size: 0.62rem; font-style: italic; grid-column: 1 / -1; padding-left: 0.5rem; }
  .breakdown-note { color: #666; font-style: italic; }

  /* Algorithm explainer */
  .algo-explainer { margin-bottom: 1.2rem; }
  .algo-explainer summary { cursor: pointer; color: #4fc3f7; font-size: 0.8rem; padding: 0.5rem 0; user-select: none; }
  .algo-explainer summary:hover { color: #81d4fa; }
  .algo-content { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 8px; padding: 1rem; margin-top: 0.4rem; }
  .algo-content p { font-size: 0.75rem; color: #999; line-height: 1.5; margin-bottom: 0.6rem; }
  .algo-content a { color: #4fc3f7; }
  .algo-formula { font-family: 'Berkeley Mono', monospace; background: #111; padding: 0.5rem 0.8rem; border-radius: 6px; color: #e0e0e0; font-size: 0.78rem; margin-bottom: 0.8rem; text-align: center; }
  .algo-signals { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 0.6rem; }
  .algo-signal { background: #0d0d0d; border: 1px solid #151515; border-radius: 6px; padding: 0.6rem 0.8rem; }
  .algo-signal-name { color: #ddd; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem; }
  .algo-tag { font-size: 0.6rem; color: #4a6a8a; background: #0a1525; padding: 0.1rem 0.4rem; border-radius: 3px; margin-left: 0.4rem; font-weight: 400; }
  .algo-signal-desc { color: #777; font-size: 0.7rem; line-height: 1.5; }
  .algo-note { color: #555; font-size: 0.7rem; font-style: italic; margin-bottom: 0; }

  /* Smallweb score indicator */
  .sw-indicator { display: flex; align-items: center; gap: 0.3rem; margin-top: 0.25rem; }
  .sw-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .sw-label { font-size: 0.6rem; color: #555; }
  .sw-details { font-size: 0.58rem; color: #3a3a3a; margin-top: 0.1rem; }

  /* Taste buttons */
  .taste-btns { display: flex; gap: 0.3rem; margin-top: 0.3rem; }
  .taste-btn { background: none; border: 1px solid #1a1a1a; color: #444; font-size: 0.75rem; padding: 0.15rem 0.4rem; border-radius: 4px; cursor: pointer; font-family: inherit; transition: all 0.15s; line-height: 1; }
  .taste-btn:hover { border-color: #555; color: #aaa; }
  .taste-btn.positive { border-color: #2d5a2d; color: #81c784; background: #0a1a0a; }
  .taste-btn.negative { border-color: #5a2d2d; color: #ef5350; background: #1a0a0a; }
  .taste-score { font-size: 0.6rem; color: #555; margin-top: 0.1rem; }

  /* Taste panel */
  .taste-panel { background: #111; border: 1px solid #222; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.5rem; }
  .taste-panel h3 { font-size: 0.85rem; color: #ddd; margin-bottom: 0.5rem; }
  .taste-panel p { font-size: 0.75rem; color: #666; margin-bottom: 0.6rem; line-height: 1.4; }
  .taste-stats { display: flex; gap: 1rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
  .taste-stat { font-size: 0.75rem; color: #888; }
  .taste-stat strong { color: #ddd; }
  .btn-train { background: #4fc3f7; color: #000; border: none; border-radius: 6px; padding: 0.4rem 0.8rem; font-family: inherit; font-size: 0.8rem; font-weight: 600; cursor: pointer; }
  .btn-train:hover { background: #81d4fa; }
  .btn-train:disabled { background: #333; color: #666; cursor: not-allowed; }

  /* Why sentence */
  .why-sentence { color: #888; font-size: 0.72rem; font-style: italic; margin: 0.3rem 0; line-height: 1.4; }

  /* Stacked score bar */
  .score-stack { display: flex; height: 8px; width: 120px; border-radius: 4px; overflow: hidden; margin-bottom: 4px; background: #111; }
  .score-stack-seg { height: 100%; min-width: 2px; transition: width 0.3s; }
  .score-stack-legend { display: flex; gap: 6px; font-size: 0.62rem; color: #666; flex-wrap: wrap; }
  .score-stack-legend span { display: flex; align-items: center; gap: 2px; }
  .score-stack-legend .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; }

  /* Audit mode */
  .btn-trace { background: none; border: 1px solid #222; color: #666; font-size: 0.68rem; padding: 0.2rem 0.5rem; border-radius: 4px; cursor: pointer; font-family: inherit; transition: all 0.2s; }
  .btn-trace:hover { border-color: #ffa726; color: #ffa726; }
  .audit-section { margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid #1a1a1a; }
  .audit-chain { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
  .audit-node { background: #111; border: 1px solid #333; border-radius: 4px; padding: 0.25rem 0.5rem; font-size: 0.72rem; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .audit-node a { color: inherit; text-decoration: none; }
  .audit-node a:hover { text-decoration: underline; }
  .audit-node.seed { border-color: #4fc3f7; color: #4fc3f7; }
  .audit-node.target { border-color: #66bb6a; color: #66bb6a; }
  .audit-arrow { color: #444; font-size: 0.8rem; }
  .audit-label { font-size: 0.68rem; color: #555; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem; }
  .audit-inbound { margin-top: 0.6rem; }
  .audit-inbound-item { font-size: 0.72rem; color: #888; margin: 0.2rem 0; }
  .audit-inbound-item a { color: #888; }
  .audit-anchor { color: #ffa726; font-style: italic; }
  .audit-cocited { margin-top: 0.6rem; }
  .audit-cocited-domain { color: #81c784; font-size: 0.72rem; cursor: pointer; }
  .audit-cocited-domain:hover { text-decoration: underline; }
  .audit-alt { font-size: 0.65rem; color: #555; margin-top: 0.3rem; }

  /* Graph viz */
  .graph-controls { margin-bottom: 0.8rem; display: flex; gap: 1rem; align-items: center; font-size: 0.8rem; flex-wrap: wrap; }
  .graph-controls label { color: #888; display: flex; align-items: center; gap: 0.3rem; }
  .graph-controls input[type="number"] { background: #111; color: #eee; border: 1px solid #333; padding: 2px 6px; border-radius: 4px; font-family: inherit; font-size: 0.8rem; width: 60px; }
  .graph-controls input[type="checkbox"] { accent-color: #4fc3f7; }
  #graphContainer { background: #050505; }
  #graphTooltip { max-width: 260px; line-height: 1.4; font-family: inherit; }

  /* Similar button */
  .btn-similar { background: none; border: 1px solid #222; color: #666; font-size: 0.68rem; padding: 0.2rem 0.5rem; border-radius: 4px; cursor: pointer; font-family: inherit; transition: all 0.2s; margin-top: 0.4rem; }
  .btn-similar:hover { border-color: #4fc3f7; color: #4fc3f7; }

  /* Similarity explorer */
  .sim-explorer { background: #111; border: 1px solid #222; border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; }
  .sim-search { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
  .sim-search input { flex: 1; background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 0.5rem 0.8rem; color: #e0e0e0; font-family: inherit; font-size: 0.85rem; outline: none; }
  .sim-search input:focus { border-color: #4fc3f7; }
  .sim-search input::placeholder { color: #333; }
  .btn { background: #4fc3f7; color: #000; border: none; border-radius: 6px; padding: 0.5rem 1rem; font-family: inherit; font-size: 0.85rem; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #81d4fa; }
  .btn:disabled { background: #333; color: #666; cursor: not-allowed; }
  .btn-sm { padding: 0.3rem 0.7rem; font-size: 0.75rem; }

  .sim-results { max-height: 400px; overflow-y: auto; }
  .sim-result { display: flex; align-items: center; gap: 0.8rem; padding: 0.5rem 0; border-bottom: 1px solid #1a1a1a; }
  .sim-result:last-child { border-bottom: none; }
  .sim-bar-wrap { width: 80px; flex-shrink: 0; }
  .sim-bar { height: 4px; background: #1a1a1a; border-radius: 2px; overflow: hidden; }
  .sim-bar-fill { height: 100%; background: #81c784; border-radius: 2px; }
  .sim-score { font-size: 0.75rem; color: #81c784; min-width: 40px; font-weight: 600; }
  .sim-domain { font-size: 0.85rem; color: #e0e0e0; flex: 1; }
  .sim-shared { font-size: 0.7rem; color: #555; }

  /* All-pairs similarity */
  .pair-grid { display: grid; grid-template-columns: 1fr; gap: 0.3rem; }
  .pair { display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.8rem; }
  .pair:hover { background: #111; }
  .pair-score { color: #81c784; font-weight: 600; min-width: 40px; font-size: 0.75rem; }
  .pair-arrow { color: #333; }
  .pair-domain { color: #e0e0e0; cursor: pointer; }
  .pair-domain:hover { color: #4fc3f7; }

  /* Seeds */
  .seeds { list-style: none; }
  .seeds li { padding: 0.3rem 0; }
  .seeds a { color: #81c784; text-decoration: none; font-size: 0.85rem; }
  .seeds a:hover { text-decoration: underline; }

  /* Domain tags */
  .domain-list { display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .domain-tag { background: #151515; border: 1px solid #1a1a1a; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.75rem; color: #888; cursor: pointer; transition: all 0.2s; }
  .domain-tag:hover { border-color: #4fc3f7; color: #4fc3f7; }

  /* Fork info */
  .fork-info { background: #1a1510; border: 1px solid #332a15; padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 0.8rem; color: #aa8844; }

  /* Loading */
  .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #333; border-top-color: #4fc3f7; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 0.4rem; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading { color: #555; font-size: 0.85rem; padding: 1rem 0; }

  /* Empty state */
  .empty { color: #444; font-size: 0.85rem; padding: 1rem 0; }

  /* Footer */
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1a1a1a; color: #333; font-size: 0.7rem; }

  /* Tune panel */
  .tune-panel { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 8px; margin-bottom: 1rem; }
  .tune-panel summary { padding: 0.6rem 1rem; cursor: pointer; color: #888; font-size: 0.78rem; }
  .tune-panel summary:hover { color: #4fc3f7; }
  .tune-panel[open] summary { border-bottom: 1px solid #1a1a1a; color: #4fc3f7; }
  .tune-content { padding: 1rem; }
  .tune-presets { display: flex; gap: 0.4rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .preset-btn { background: #151515; border: 1px solid #222; color: #888; padding: 0.3rem 0.7rem; border-radius: 4px; font-size: 0.72rem; cursor: pointer; transition: all 0.15s; }
  .preset-btn:hover { border-color: #4fc3f7; color: #4fc3f7; }
  .preset-btn.active { border-color: #4fc3f7; color: #4fc3f7; background: #0a1520; }
  .tune-section { margin-bottom: 0.8rem; }
  .tune-section-title { color: #555; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
  .slider-row { display: grid; grid-template-columns: 90px 1fr 45px; gap: 0.5rem; align-items: center; padding: 0.2rem 0; }
  .slider-label { color: #999; font-size: 0.72rem; font-family: 'Berkeley Mono', monospace; }
  .slider-value { color: #ddd; font-size: 0.72rem; font-family: 'Berkeley Mono', monospace; text-align: right; font-weight: 600; }
  input[type="range"] { -webkit-appearance: none; width: 100%; height: 4px; background: #222; border-radius: 2px; outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; background: #4fc3f7; border-radius: 50%; cursor: pointer; }
  input[type="range"]::-moz-range-thumb { width: 14px; height: 14px; background: #4fc3f7; border-radius: 50%; cursor: pointer; border: none; }
  .tune-actions { display: flex; gap: 0.5rem; margin-top: 0.8rem; align-items: center; }
  .tune-btn { background: #151515; border: 1px solid #222; color: #888; padding: 0.35rem 0.8rem; border-radius: 4px; font-size: 0.72rem; cursor: pointer; }
  .tune-btn:hover { border-color: #4fc3f7; color: #4fc3f7; }
  .tune-btn.primary { border-color: #4fc3f7; color: #4fc3f7; }
  .tune-msg { font-size: 0.7rem; color: #555; }
  .tune-formula { font-family: 'Berkeley Mono', monospace; color: #888; font-size: 0.7rem; background: #0d0d0d; padding: 0.4rem 0.6rem; border-radius: 4px; margin-bottom: 0.8rem; }
</style>
</head>
<body>
  <h1><a class="back" href="/smallweb/">&larr;</a>smallweb / $name</h1>
  <div class="meta">$description</div>

  <div class="stats">
    <div class="stat"><div class="stat-value">$n_nodes</div><div class="stat-label">pages</div></div>
    <div class="stat"><div class="stat-value">$n_edges</div><div class="stat-label">links</div></div>
    <div class="stat"><div class="stat-value">$n_seeds</div><div class="stat-label">seeds</div></div>
    <div class="stat"><div class="stat-value">$n_domains</div><div class="stat-label">domains</div></div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" data-tab="discoveries">discoveries</button>
    <button class="tab" data-tab="similarity">similarity</button>
    <button class="tab" data-tab="seeds">seeds & domains</button>
    <button class="tab" data-tab="graph">graph</button>
  </div>

  <!-- Tab: Discoveries -->
  <div class="tab-panel active" id="tab-discoveries">
    <details class="algo-explainer">
      <summary>how scoring works</summary>
      <div class="algo-content">
        <p>Every discovered page gets a single <strong>overall score</strong> that blends four signals:</p>
        <div class="algo-formula">score = pagerank_pct &times; quality &times; smallweb &times; (0.5 + 0.5 &times; taste)</div>
        <div class="algo-signals">
          <div class="algo-signal">
            <div class="algo-signal-name">pagerank <span class="algo-tag">link authority</span></div>
            <div class="algo-signal-desc">How well-connected this page is in the link graph. Computed as personalized PageRank (biased toward your seed sites), then converted to a percentile (0&ndash;1) so it doesn't drown out other signals. A page that many other pages link to ranks higher.</div>
          </div>
          <div class="algo-signal">
            <div class="algo-signal-name">quality <span class="algo-tag">html cleanliness</span></div>
            <div class="algo-signal-desc">Measures how clean the page's HTML is. Starts from the text-to-HTML ratio (more text, less markup = better), then applies penalties for trackers, ad-tech scripts, cookie banners, affiliate links, and AI-generated content patterns. Bonuses for IndieWeb signals like Webmention, IndieAuth, RSS feeds, and microformats. Inspired by <a href="https://search.marginalia.nu" target="_blank">Marginalia Search</a>.</div>
          </div>
          <div class="algo-signal">
            <div class="algo-signal-name">smallweb <span class="algo-tag">indie score</span></div>
            <div class="algo-signal-desc">Is this domain part of the small web? Combines two sub-signals: <strong>popularity</strong> (a bell curve &mdash; too many inbound links means it's a platform, not a personal site) and <strong>outlink profile</strong> (what fraction of its outlinks go to other small sites vs. big platforms).</div>
          </div>
          <div class="algo-signal">
            <div class="algo-signal-name">taste <span class="algo-tag">neural preference</span></div>
            <div class="algo-signal-desc">Optional. If you upvote/downvote discoveries, a small neural classifier learns your preferences. Uses sentence embeddings (MiniLM) on page URLs to predict whether you'd like new pages. Neutral (0.5) when untrained.</div>
          </div>
        </div>
        <p class="algo-note">Click the score badge (&oplus;) on any result to see exactly how it was scored, including which penalties and bonuses fired.</p>
      </div>
    </details>
    <div class="controls">
      <div>
        <div class="control-label">ranking</div>
        <div class="toggle-group">
          <button class="toggle-btn active" data-rank="personalized" onclick="switchRank('personalized')">personalized</button>
          <button class="toggle-btn" data-rank="standard" onclick="switchRank('standard')">standard</button>
        </div>
      </div>
      <div>
        <div class="control-label">sort by</div>
        <div class="toggle-group">
          <button class="toggle-btn active" data-sort="blended" onclick="switchSort('blended')">overall</button>
          <button class="toggle-btn" data-sort="pagerank" onclick="switchSort('pagerank')">pagerank</button>
          <button class="toggle-btn" data-sort="quality" onclick="switchSort('quality')">quality</button>
          <button class="toggle-btn" data-sort="smallweb" onclick="switchSort('smallweb')">smallweb</button>
          <button class="toggle-btn" data-sort="taste" onclick="switchSort('taste')">taste</button>
        </div>
      </div>
    </div>

    <details class="tune-panel" id="tunePanel">
      <summary>tune algorithm</summary>
      <div class="tune-content">
        <div class="tune-presets">
          <button class="preset-btn active" onclick="applyPreset('default')">default</button>
          <button class="preset-btn" onclick="applyPreset('indie_purist')">indie purist</button>
          <button class="preset-btn" onclick="applyPreset('quality_focused')">quality focused</button>
          <button class="preset-btn" onclick="applyPreset('broad_discovery')">broad discovery</button>
        </div>
        <div class="tune-formula" id="tuneFormula">score = pagerank &times; quality &times; smallweb</div>
        <div class="tune-section">
          <div class="tune-section-title">signal weights (exponent &mdash; higher = more discriminating)</div>
          <div class="slider-row">
            <span class="slider-label">pagerank</span>
            <input type="range" id="sl-pagerank" min="0" max="2" step="0.1" value="1.0" oninput="onSlider(this)">
            <span class="slider-value" id="sv-pagerank">1.0</span>
          </div>
          <div class="slider-row">
            <span class="slider-label">quality</span>
            <input type="range" id="sl-quality" min="0" max="2" step="0.1" value="1.0" oninput="onSlider(this)">
            <span class="slider-value" id="sv-quality">1.0</span>
          </div>
          <div class="slider-row">
            <span class="slider-label">smallweb</span>
            <input type="range" id="sl-smallweb" min="0" max="2" step="0.1" value="1.0" oninput="onSlider(this)">
            <span class="slider-value" id="sv-smallweb">1.0</span>
          </div>
          <div class="slider-row" id="sl-taste-row" style="display:none">
            <span class="slider-label">taste</span>
            <input type="range" id="sl-taste-weight" min="0" max="1" step="0.1" value="0.5" oninput="onSlider(this)">
            <span class="slider-value" id="sv-taste-weight">0.5</span>
          </div>
        </div>
        <div class="tune-section">
          <div class="tune-section-title">unfetched page penalty</div>
          <div class="slider-row">
            <span class="slider-label">fetched boost</span>
            <input type="range" id="sl-fetched" min="0" max="1" step="0.05" value="0.2" oninput="onSlider(this)">
            <span class="slider-value" id="sv-fetched">0.2</span>
          </div>
        </div>
        <div class="tune-actions">
          <button class="tune-btn primary" onclick="saveConfig()">save to graph</button>
          <button class="tune-btn" onclick="resetConfig()">reset to defaults</button>
          <span class="tune-msg" id="tuneMsg"></span>
        </div>
      </div>
    </details>

    <div id="tastePanel" class="taste-panel" style="display:none">
      <h3>taste classifier</h3>
      <p>thumbs up/down discoveries to train a neural classifier that learns your taste. needs 3+ of each to train.</p>
      <div class="taste-stats" id="tasteStats"></div>
      <button class="btn-train" id="btnTrain" onclick="trainTaste()" disabled>train model</button>
      <span id="tasteMsg" style="font-size:0.75rem;color:#666;margin-left:0.5rem"></span>
    </div>

    <div id="discoveriesContainer">
      <div class="loading"><span class="spinner"></span>loading discoveries...</div>
    </div>
  </div>

  <!-- Tab: Similarity -->
  <div class="tab-panel" id="tab-similarity">
    <div class="sim-explorer">
      <h2 style="border:none;padding:0;margin-bottom:0.8rem">find similar sites</h2>
      <p style="color:#555;font-size:0.78rem;margin-bottom:1rem">enter a domain to find sites with similar inbound link patterns (co-citation)</p>
      <div class="sim-search">
        <input type="text" id="simTarget" placeholder="e.g. 100r.co or wiki.xxiivv.com" onkeydown="if(event.key==='Enter')findSimilar()">
        <button class="btn btn-sm" onclick="findSimilar()">search</button>
      </div>
      <div id="simResults"></div>
    </div>

    <div class="section">
      <h2>top similar pairs</h2>
      <div id="pairsContainer">
        <div class="loading"><span class="spinner"></span>computing similarities...</div>
      </div>
    </div>
  </div>

  <!-- Tab: Seeds & Domains -->
  <div class="tab-panel" id="tab-seeds">
    <div class="section">
      <h2>seeds ($n_seeds)</h2>
      <ul class="seeds">$seeds_html</ul>
    </div>
    <div class="section">
      <h2>domains ($n_domains)</h2>
      <div class="domain-list">$domains_html</div>
    </div>
  </div>

  <!-- Tab: Graph Visualization -->
  <div class="tab-panel" id="tab-graph">
    <div class="graph-controls">
      <label>show top <input type="number" id="graphNodeLimit" value="80" min="10" max="500"> domains</label>
      <label><input type="checkbox" id="graphShowLabels" checked> labels</label>
      <button class="btn btn-sm" onclick="loadGraphViz()">reload</button>
    </div>
    <div id="graphContainer" style="width:100%;height:600px;border:1px solid #222;border-radius:8px;overflow:hidden;position:relative"></div>
    <div id="graphTooltip" style="display:none;position:fixed;background:#111;border:1px solid #333;padding:8px 12px;border-radius:6px;font-size:0.72rem;pointer-events:none;z-index:1000;color:#ccc"></div>
  </div>

  <footer>
    <a href="/smallweb/">smallweb</a> &mdash; graph: $graph_id
  </footer>

<script>
const GRAPH_ID = '$graph_id';
const API = (location.pathname.startsWith('/smallweb') ? '/smallweb' : '') + '/api';

let currentSort = 'blended';
let currentRank = 'personalized';
let discoveriesCache = { personalized: null, standard: null };

// ── Tabs ──

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');

    // Lazy-load similarity pairs
    if (tab.dataset.tab === 'similarity' && !document.getElementById('pairsContainer').dataset.loaded) {
      loadPairs();
    }
    // Lazy-load graph visualization
    if (tab.dataset.tab === 'graph' && !graphLoaded) {
      loadGraphViz();
    }
  });
});

// ── Discoveries ──

async function loadDiscoveries(rank) {
  const key = rank || currentRank;
  if (discoveriesCache[key]) {
    renderDiscoveries(discoveriesCache[key]);
    return;
  }

  const container = document.getElementById('discoveriesContainer');
  container.innerHTML = '<div class="loading"><span class="spinner"></span>loading discoveries...</div>';

  try {
    const personalized = key === 'personalized' ? 'true' : 'false';
    let url = API + '/graphs/' + GRAPH_ID + '/discoveries?top=50&personalized=' + personalized;

    // If tuning panel is open and sliders have been changed, send config overrides
    const tunePanel = document.getElementById('tunePanel');
    if (tunePanel && tunePanel.open) {
      const overrides = collectSliderValues();
      url += '&config=' + encodeURIComponent(JSON.stringify(overrides));
    }

    const res = await fetch(url);
    const data = await res.json();
    discoveriesCache[key] = data.discoveries || [];

    // Update config state from response
    if (data.config) {
      currentConfig = data.config;
      // Show taste slider if taste is active
      if (data.taste_active) {
        document.getElementById('sl-taste-row').style.display = '';
      }
      // Set sliders to match returned config (on first load)
      if (!tunePanel.open) {
        setSliders(data.config);
      }
    }

    renderDiscoveries(discoveriesCache[key]);
  } catch (e) {
    container.innerHTML = '<div class="empty">error loading discoveries: ' + e.message + '</div>';
  }
}

function renderDiscoveries(discoveries) {
  const container = document.getElementById('discoveriesContainer');

  // Sort
  let sorted = [...discoveries];
  if (currentSort === 'blended') {
    sorted.sort((a, b) => (b.score || 0) - (a.score || 0));
  } else if (currentSort === 'pagerank') {
    sorted.sort((a, b) => (b.pagerank || 0) - (a.pagerank || 0));
  } else if (currentSort === 'quality') {
    sorted.sort((a, b) => (b.quality || 0) - (a.quality || 0));
  } else if (currentSort === 'smallweb') {
    sorted.sort((a, b) => (b.smallweb_score || 0) - (a.smallweb_score || 0));
  } else if (currentSort === 'taste') {
    sorted.sort((a, b) => (b.taste_score || 0.5) - (a.taste_score || 0.5));
  }

  if (sorted.length === 0) {
    container.innerHTML = '<div class="empty">no discoveries yet &mdash; try crawling with more hops</div>';
    return;
  }

  container.innerHTML = sorted.map((d, i) => {
    const title = d.title || new URL(d.url).hostname;
    const quality = d.quality != null ? d.quality : 0.5;
    const qMeasured = d.quality_measured !== false;
    const qColor = !qMeasured ? '#555' : quality >= 0.7 ? '#81c784' : quality >= 0.4 ? '#ffb74d' : '#ef5350';
    const qPct = Math.round(quality * 100);
    const domain = d.domain || new URL(d.url).hostname;
    const sw = d.smallweb_score != null ? d.smallweb_score : 0.5;
    const swColor = sw >= 0.7 ? '#81c784' : sw >= 0.4 ? '#ffb74d' : sw >= 0.15 ? '#ef5350' : '#666';
    const inbound = d.inbound_domains || 0;
    const outlink = d.outlink_score != null ? Math.round(d.outlink_score * 100) : '?';
    const taste = d.taste_score != null ? d.taste_score : 0.5;

    // Anchor text pills (max 5)
    const anchors = (d.anchor_texts || []).slice(0, 5);
    const anchorHtml = anchors.length > 0
      ? '<div class="anchor-texts">' + anchors.map(a => '<span class="anchor-pill" title="' + escHtml(a) + '">' + escHtml(a) + '</span>').join('') + '</div>'
      : '';

    // Build breakdown HTML
    const breakdownId = 'breakdown-' + i;
    let breakdownHtml = '';

    // Overall score formula
    const sb = d.score_breakdown;
    if (sb) {
      breakdownHtml += '<div class="breakdown-section"><div class="breakdown-title">overall score</div>';
      breakdownHtml += '<div class="breakdown-formula">' + escHtml(sb.formula) + '</div>';
      breakdownHtml += '<div class="breakdown-components">';
      for (const [key, comp] of Object.entries(sb.components || {})) {
        const val = comp.value != null ? (typeof comp.value === 'number' ? comp.value.toFixed(3) : comp.value) : '?';
        breakdownHtml += '<div class="breakdown-row"><span class="breakdown-key">' + escHtml(key) + '</span><span class="breakdown-val">' + val + '</span><span class="breakdown-why">' + escHtml(comp.why || '') + '</span></div>';
      }
      breakdownHtml += '</div></div>';
    }

    // Quality breakdown (if measured)
    const qb = d.quality_breakdown;
    if (qb) {
      breakdownHtml += '<div class="breakdown-section"><div class="breakdown-title">quality breakdown</div>';
      breakdownHtml += '<div class="breakdown-row"><span class="breakdown-key">text ratio</span><span class="breakdown-val">' + (qb.text_ratio * 100).toFixed(1) + '%</span><span class="breakdown-why">What fraction of the HTML is actual text vs markup. Higher = less bloat.</span></div>';
      breakdownHtml += '<div class="breakdown-row"><span class="breakdown-key">base score</span><span class="breakdown-val">' + qb.base.toFixed(3) + '</span><span class="breakdown-why">Starting score from text-to-html ratio. log(ratio) normalized to 0\u20131.</span></div>';
      if (qb.penalties && qb.penalties.length > 0) {
        breakdownHtml += '<div class="breakdown-subtitle">penalties</div>';
        for (const p of qb.penalties) {
          breakdownHtml += '<div class="breakdown-row penalty"><span class="breakdown-key">' + escHtml(p.signal) + '</span><span class="breakdown-val">-' + p.value.toFixed(3) + '</span><span class="breakdown-detail">' + escHtml(p.detail) + '</span><span class="breakdown-why">' + escHtml(p.why) + '</span></div>';
        }
      }
      if (qb.bonuses && qb.bonuses.length > 0) {
        breakdownHtml += '<div class="breakdown-subtitle">bonuses</div>';
        for (const b of qb.bonuses) {
          breakdownHtml += '<div class="breakdown-row bonus"><span class="breakdown-key">' + escHtml(b.signal) + '</span><span class="breakdown-val">+' + b.value.toFixed(3) + '</span><span class="breakdown-detail">' + escHtml(b.detail) + '</span><span class="breakdown-why">' + escHtml(b.why) + '</span></div>';
        }
      }
      breakdownHtml += '</div>';
    } else if (!qMeasured) {
      breakdownHtml += '<div class="breakdown-section"><div class="breakdown-title">quality</div><div class="breakdown-note">Not measured \u2014 this page was discovered as an outlink but not fetched. Quality defaults to 0.5 (neutral).</div></div>';
    }

    var whyText = whySentence(d);
    var auditId = 'audit-' + i;

    return '<div class="discovery">' +
      '<div class="discovery-header">' +
        '<div class="discovery-main">' +
          '<a class="discovery-title" href="' + d.url + '" target="_blank">' + escHtml(title) + '</a>' +
          '<div class="discovery-url">' + escHtml(d.url) + '</div>' +
          (d.description ? '<div class="discovery-desc">' + escHtml(d.description.slice(0, 150)) + '</div>' : '') +
          (whyText ? '<div class="why-sentence">' + escHtml(whyText) + '</div>' : '') +
          anchorHtml +
          '<div class="taste-btns">' +
            '<button class="taste-btn' + (tasteLabels.positive.includes(d.url) ? ' positive' : '') + '" onclick="labelTaste(\\'' + escJs(d.url) + '\\', \\'positive\\', this)" title="good discovery">\u25B2</button>' +
            '<button class="taste-btn' + (tasteLabels.negative.includes(d.url) ? ' negative' : '') + '" onclick="labelTaste(\\'' + escJs(d.url) + '\\', \\'negative\\', this)" title="not useful">\u25BC</button>' +
            '<button class="btn-similar" onclick="findSimilarFrom(\\'' + escJs(domain) + '\\')">similar</button>' +
            '<button class="btn-trace" onclick="loadAudit(\\'' + escJs(d.url) + '\\', \\'' + auditId + '\\')">trace</button>' +
          '</div>' +
        '</div>' +
        '<div style="text-align:right">' +
          scoreStackHtml(d) +
          '<div class="score-badge"><a class="why-toggle" onclick="toggleBreakdown(\\'' + breakdownId + '\\')" title="show score breakdown">#' + (i + 1) + ' \u00b7 ' + d.score.toFixed(4) + ' \u24d8</a></div>' +
        '</div>' +
      '</div>' +
      '<div class="breakdown" id="' + breakdownId + '" style="display:none">' + breakdownHtml + '</div>' +
      '<div class="audit-section" id="' + auditId + '" style="display:none"></div>' +
    '</div>';
  }).join('');
}

// ── Why Sentence ──

function whySentence(d) {
  var parts = [];
  var pr = d.pagerank_pct || 0;
  var q = d.quality || 0.5;
  var sw = d.smallweb_score || 0.5;
  var taste = d.taste_score || 0.5;
  var inbound = d.inbound_domains || 0;
  var qMeasured = d.quality_measured !== false;

  if (pr >= 0.8) parts.push('highly connected in your seed network');
  else if (pr >= 0.5) parts.push('well-linked from sites you follow');

  if (qMeasured && q >= 0.8) {
    var bonuses = (d.quality_breakdown && d.quality_breakdown.bonuses || []).map(function(b) { return b.signal.replace(/_/g, ' '); });
    parts.push(bonuses.length ? 'clean HTML with ' + bonuses.join(', ') : 'very clean HTML');
  } else if (qMeasured && q >= 0.6) {
    parts.push('decent quality markup');
  } else if (qMeasured && q < 0.4) {
    var penalties = d.quality_breakdown && d.quality_breakdown.penalties || [];
    var topP = penalties[0];
    if (topP) parts.push('lower quality (' + topP.signal.replace(/_/g, ' ') + ')');
  }

  if (sw >= 0.8) parts.push('sweet spot of ' + inbound + ' inbound domains');
  else if (sw >= 0.5) parts.push('linked by ' + inbound + ' sites');

  var anchors = d.anchor_texts || [];
  if (anchors.length > 0 && anchors[0].length > 3 && anchors[0].length < 50) {
    parts.push('described as \\u201c' + anchors[0] + '\\u201d');
  }

  if (taste > 0.7) parts.push('matches your taste');
  else if (taste < 0.3) parts.push('outside your usual taste');

  if (parts.length === 0) return '';
  var s = parts.join(' \\u00b7 ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ── Stacked Score Bar ──

function scoreStackHtml(d) {
  var pr = d.pagerank_pct || 0.01;
  var q = d.quality || 0.5;
  var sw = d.smallweb_score || 0.5;
  var taste = d.taste_score || 0.5;
  var cfg = (currentConfig && currentConfig.formula) || {};
  var prExp = cfg.pagerank_exp || 1.0;
  var qExp = cfg.quality_exp || 1.0;
  var swExp = cfg.smallweb_exp || 1.0;

  var logPr = Math.abs(prExp * Math.log(Math.max(pr, 0.001)));
  var logQ = Math.abs(qExp * Math.log(Math.max(q, 0.001)));
  var logSw = Math.abs(swExp * Math.log(Math.max(sw, 0.001)));
  var logT = Math.abs(Math.log(Math.max(0.5 + 0.5 * taste, 0.001)));
  var total = logPr + logQ + logSw + logT || 1;

  var pPr = Math.round(logPr / total * 100);
  var pQ = Math.round(logQ / total * 100);
  var pSw = Math.round(logSw / total * 100);
  var pT = Math.max(0, 100 - pPr - pQ - pSw);

  return '<div class="score-stack" title="score composition: pr ' + (logPr/total*100).toFixed(0) + '% / q ' + (logQ/total*100).toFixed(0) + '% / sw ' + (logSw/total*100).toFixed(0) + '%">' +
    '<div class="score-stack-seg" style="width:' + pPr + '%;background:#4fc3f7"></div>' +
    '<div class="score-stack-seg" style="width:' + pQ + '%;background:#66bb6a"></div>' +
    '<div class="score-stack-seg" style="width:' + pSw + '%;background:#ffa726"></div>' +
    (pT > 1 ? '<div class="score-stack-seg" style="width:' + pT + '%;background:#ab47bc"></div>' : '') +
  '</div>' +
  '<div class="score-stack-legend">' +
    '<span><span class="dot" style="background:#4fc3f7"></span>pr ' + pr.toFixed(2) + '</span>' +
    '<span><span class="dot" style="background:#66bb6a"></span>q ' + q.toFixed(2) + '</span>' +
    '<span><span class="dot" style="background:#ffa726"></span>sw ' + sw.toFixed(2) + '</span>' +
    (taste !== 0.5 ? '<span><span class="dot" style="background:#ab47bc"></span>t ' + taste.toFixed(2) + '</span>' : '') +
  '</div>';
}

// ── Audit Mode ──

async function loadAudit(url, auditId) {
  var el = document.getElementById(auditId);
  if (!el) return;

  // Toggle
  if (el.style.display !== 'none' && el.innerHTML) {
    el.style.display = 'none';
    return;
  }

  el.style.display = 'block';
  el.innerHTML = '<div class="loading"><span class="spinner"></span>tracing discovery chain...</div>';

  try {
    var res = await fetch(API + '/graphs/' + GRAPH_ID + '/audit?url=' + encodeURIComponent(url));
    var data = await res.json();

    if (data.error) {
      el.innerHTML = '<div class="empty">' + escHtml(data.error) + '</div>';
      return;
    }

    var html = '';

    // Chain visualization
    if (data.chain && data.chain.length > 0) {
      html += '<div class="audit-label">discovery path</div>';
      html += '<div class="audit-chain">';
      data.chain.forEach(function(node, idx) {
        var cls = node.is_seed ? 'audit-node seed' : (idx === data.chain.length - 1 ? 'audit-node target' : 'audit-node');
        var label = node.title || node.domain || new URL(node.url).hostname;
        if (label.length > 30) label = label.slice(0, 28) + '..';
        html += '<div class="' + cls + '" title="' + escHtml(node.url) + '"><a href="' + escHtml(node.url) + '" target="_blank">' + escHtml(label) + '</a></div>';
        if (idx < data.chain.length - 1) html += '<span class="audit-arrow">\\u2192</span>';
      });
      html += '</div>';
    }

    if (data.alternative_paths > 0) {
      html += '<div class="audit-alt">' + data.alternative_paths + ' alternative path' + (data.alternative_paths > 1 ? 's' : '') + ' from other seeds</div>';
    }

    // Inbound links
    if (data.inbound_links && data.inbound_links.length > 0) {
      html += '<div class="audit-inbound"><div class="audit-label">who links here (' + data.inbound_links.length + ')</div>';
      data.inbound_links.slice(0, 10).forEach(function(link) {
        var linkDomain = link.domain || '?';
        html += '<div class="audit-inbound-item"><a href="' + escHtml(link.url) + '" target="_blank">' + escHtml(linkDomain) + '</a>';
        if (link.anchor_text) html += ' \\u2014 <span class="audit-anchor">"' + escHtml(link.anchor_text) + '"</span>';
        html += '</div>';
      });
      if (data.inbound_links.length > 10) html += '<div class="audit-inbound-item" style="color:#444">+ ' + (data.inbound_links.length - 10) + ' more</div>';
      html += '</div>';
    }

    // Co-cited domains
    if (data.co_cited_with && data.co_cited_with.length > 0) {
      html += '<div class="audit-cocited"><div class="audit-label">co-cited with</div>';
      data.co_cited_with.slice(0, 8).forEach(function(d) {
        html += '<span class="audit-cocited-domain" onclick="findSimilarFrom(\\'' + escJs(d) + '\\')" style="margin-right:0.6rem">' + escHtml(d) + '</span>';
      });
      html += '</div>';
    }

    el.innerHTML = html || '<div class="empty">no path data available</div>';
  } catch (e) {
    el.innerHTML = '<div class="empty">error: ' + e.message + '</div>';
  }
}

// ── Graph Visualization ──

var graphLoaded = false;

async function loadGraphViz() {
  var container = document.getElementById('graphContainer');
  if (!container) return;

  container.innerHTML = '<div class="loading" style="padding:2rem"><span class="spinner"></span>loading graph data...</div>';

  try {
    var limit = parseInt(document.getElementById('graphNodeLimit').value) || 80;
    var showLabels = document.getElementById('graphShowLabels').checked;
    var res = await fetch(API + '/graphs/' + GRAPH_ID + '/domain-graph?top=' + limit);
    var data = await res.json();

    if (!data.nodes || data.nodes.length === 0) {
      container.innerHTML = '<div class="empty" style="padding:2rem">no graph data available</div>';
      return;
    }

    // Clear container
    container.innerHTML = '';

    var width = container.clientWidth;
    var height = container.clientHeight || 600;

    var svg = d3.select(container).append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('background', '#050505');

    var g = svg.append('g');

    // Zoom
    var zoom = d3.zoom()
      .scaleExtent([0.1, 6])
      .on('zoom', function(event) { g.attr('transform', event.transform); });
    svg.call(zoom);

    // Build node/link data
    var nodeMap = {};
    data.nodes.forEach(function(n) { nodeMap[n.id] = n; });

    var links = data.links.filter(function(l) {
      return nodeMap[l.source] && nodeMap[l.target];
    });

    // Scale node size by pagerank
    var prMax = d3.max(data.nodes, function(n) { return n.pagerank; }) || 1;
    var rScale = d3.scaleSqrt().domain([0, prMax]).range([3, 18]);

    // Simulation
    var simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(links).id(function(d) { return d.id; }).distance(60).strength(0.3))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(function(d) { return rScale(d.pagerank) + 4; }));

    // Links
    var link = g.append('g').selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#1a1a1a')
      .attr('stroke-opacity', function(d) { return Math.min(0.6, 0.1 + d.weight * 0.05); })
      .attr('stroke-width', function(d) { return Math.min(3, 0.5 + d.weight * 0.3); });

    // Nodes
    var node = g.append('g').selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', function(d) { return rScale(d.pagerank); })
      .attr('fill', function(d) {
        if (d.is_seed) return '#4fc3f7';
        if (d.quality >= 0.7) return '#66bb6a';
        if (d.quality >= 0.4) return '#ffa726';
        return '#555';
      })
      .attr('stroke', '#0a0a0a')
      .attr('stroke-width', 1)
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', function(event, d) {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', function(event, d) {
          d.fx = event.x; d.fy = event.y;
        })
        .on('end', function(event, d) {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
      );

    // Labels
    var label = g.append('g').selectAll('text')
      .data(data.nodes)
      .join('text')
      .text(function(d) { return d.id; })
      .attr('font-size', '8px')
      .attr('font-family', "'Berkeley Mono', monospace")
      .attr('fill', '#666')
      .attr('dx', function(d) { return rScale(d.pagerank) + 3; })
      .attr('dy', '0.3em')
      .style('display', showLabels ? 'block' : 'none')
      .style('pointer-events', 'none');

    // Tooltip
    var tooltip = document.getElementById('graphTooltip');

    node.on('mouseover', function(event, d) {
      tooltip.style.display = 'block';
      tooltip.innerHTML = '<strong>' + escHtml(d.id) + '</strong><br>' +
        (d.top_title ? escHtml(d.top_title) + '<br>' : '') +
        d.pages + ' pages<br>' +
        'pr: ' + d.pagerank.toFixed(2) +
        ' \\u00b7 q: ' + d.quality.toFixed(2) +
        ' \\u00b7 sw: ' + d.smallweb.toFixed(2) +
        (d.is_seed ? '<br><span style="color:#4fc3f7">seed</span>' : '');
      // Highlight connections
      link.attr('stroke', function(l) {
        return (l.source.id === d.id || l.target.id === d.id) ? '#4fc3f7' : '#1a1a1a';
      }).attr('stroke-opacity', function(l) {
        return (l.source.id === d.id || l.target.id === d.id) ? 0.8 : 0.1;
      });
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
    })
    .on('mousemove', function(event) {
      tooltip.style.left = (event.clientX + 12) + 'px';
      tooltip.style.top = (event.clientY - 10) + 'px';
    })
    .on('mouseout', function() {
      tooltip.style.display = 'none';
      link.attr('stroke', '#1a1a1a').attr('stroke-opacity', function(d) { return Math.min(0.6, 0.1 + d.weight * 0.05); });
      d3.select(this).attr('stroke', '#0a0a0a').attr('stroke-width', 1);
    })
    .on('click', function(event, d) {
      findSimilar(d.id);
    });

    // Tick
    simulation.on('tick', function() {
      link
        .attr('x1', function(d) { return d.source.x; })
        .attr('y1', function(d) { return d.source.y; })
        .attr('x2', function(d) { return d.target.x; })
        .attr('y2', function(d) { return d.target.y; });
      node
        .attr('cx', function(d) { return d.x; })
        .attr('cy', function(d) { return d.y; });
      label
        .attr('x', function(d) { return d.x; })
        .attr('y', function(d) { return d.y; });
    });

    // Label toggle
    document.getElementById('graphShowLabels').onchange = function() {
      label.style('display', this.checked ? 'block' : 'none');
    };

    graphLoaded = true;

  } catch (e) {
    container.innerHTML = '<div class="empty" style="padding:2rem">error loading graph: ' + e.message + '</div>';
  }
}

function switchRank(rank) {
  currentRank = rank;
  document.querySelectorAll('[data-rank]').forEach(b => b.classList.toggle('active', b.dataset.rank === rank));
  loadDiscoveries(rank);
}

function switchSort(sort) {
  currentSort = sort;
  document.querySelectorAll('[data-sort]').forEach(b => b.classList.toggle('active', b.dataset.sort === sort));
  const cached = discoveriesCache[currentRank];
  if (cached) renderDiscoveries(cached);
}

function toggleBreakdown(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// ── Similarity ──

async function findSimilar(target) {
  const domain = target || document.getElementById('simTarget').value.trim();
  if (!domain) return;

  document.getElementById('simTarget').value = domain;
  const container = document.getElementById('simResults');
  container.innerHTML = '<div class="loading"><span class="spinner"></span>finding similar sites...</div>';

  // Switch to similarity tab
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'similarity'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-similarity'));

  try {
    const res = await fetch(API + '/graphs/' + GRAPH_ID + '/similar?target=' + encodeURIComponent(domain) + '&top=20');
    const data = await res.json();

    if (!data.similar || data.similar.length === 0) {
      container.innerHTML = '<div class="empty">no similar sites found for ' + escHtml(domain) + ' &mdash; needs cross-domain links (try more hops)</div>';
      return;
    }

    container.innerHTML = '<div class="sim-results">' + data.similar.map(s => {
      const pct = Math.round(s.similarity * 100);
      return '<div class="sim-result">' +
        '<span class="sim-score">' + (s.similarity).toFixed(2) + '</span>' +
        '<div class="sim-bar-wrap"><div class="sim-bar"><div class="sim-bar-fill" style="width:' + pct + '%"></div></div></div>' +
        '<span class="sim-domain pair-domain" onclick="findSimilar(\\'' + escJs(s.domain) + '\\')">' + escHtml(s.domain) + '</span>' +
        '<span class="sim-shared">' + s.shared_sources + ' shared</span>' +
      '</div>';
    }).join('') + '</div>';
  } catch (e) {
    container.innerHTML = '<div class="empty">error: ' + e.message + '</div>';
  }
}

function findSimilarFrom(domain) {
  findSimilar(domain);
}

async function loadPairs() {
  const container = document.getElementById('pairsContainer');
  container.dataset.loaded = 'true';

  try {
    const res = await fetch(API + '/graphs/' + GRAPH_ID + '/similarities?min_shared=2&top=30');
    const data = await res.json();

    if (!data.pairs || data.pairs.length === 0) {
      container.innerHTML = '<div class="empty">no similar pairs found &mdash; needs more cross-domain links</div>';
      return;
    }

    container.innerHTML = '<div class="pair-grid">' + data.pairs.map(p => {
      return '<div class="pair">' +
        '<span class="pair-score">' + p.similarity.toFixed(2) + '</span>' +
        '<span class="pair-domain" onclick="findSimilar(\\'' + escJs(p.domain_a) + '\\')">' + escHtml(p.domain_a) + '</span>' +
        '<span class="pair-arrow">&harr;</span>' +
        '<span class="pair-domain" onclick="findSimilar(\\'' + escJs(p.domain_b) + '\\')">' + escHtml(p.domain_b) + '</span>' +
        '<span class="sim-shared">' + p.shared_sources + ' shared</span>' +
      '</div>';
    }).join('') + '</div>';
  } catch (e) {
    container.innerHTML = '<div class="empty">error: ' + e.message + '</div>';
  }
}

// ── Domain tag clicks → similarity ──

document.querySelectorAll('.domain-tag').forEach(tag => {
  tag.addEventListener('click', () => {
    const domain = tag.textContent.split(' (')[0].trim();
    findSimilar(domain);
  });
});

// ── Helpers ──

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function escJs(s) {
  return s.replace(/\\\\/g, '\\\\\\\\').replace(/'/g, "\\\\'");
}

// ── Taste System ──

let tasteLabels = { positive: [], negative: [] };

async function loadTasteStatus() {
  try {
    const res = await fetch(API + '/graphs/' + GRAPH_ID + '/taste');
    const data = await res.json();
    tasteLabels.positive = data.positive_urls || [];
    tasteLabels.negative = data.negative_urls || [];

    const panel = document.getElementById('tastePanel');
    panel.style.display = 'block';

    const stats = document.getElementById('tasteStats');
    stats.innerHTML =
      '<span class="taste-stat"><strong>' + tasteLabels.positive.length + '</strong> liked</span>' +
      '<span class="taste-stat"><strong>' + tasteLabels.negative.length + '</strong> rejected</span>' +
      (data.is_trained ? '<span class="taste-stat" style="color:#81c784">model trained</span>' : '');

    const btn = document.getElementById('btnTrain');
    btn.disabled = tasteLabels.positive.length < 3 || tasteLabels.negative.length < 3;
  } catch (e) {
    // taste not available, hide panel
  }
}

async function labelTaste(url, label, btn) {
  // Toggle: if already this label, remove it
  const isAlreadyLabeled =
    (label === 'positive' && tasteLabels.positive.includes(url)) ||
    (label === 'negative' && tasteLabels.negative.includes(url));

  const actualLabel = isAlreadyLabeled ? 'remove' : label;

  try {
    await fetch(API + '/graphs/' + GRAPH_ID + '/taste/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, label: actualLabel }),
    });

    // Update local state
    tasteLabels.positive = tasteLabels.positive.filter(u => u !== url);
    tasteLabels.negative = tasteLabels.negative.filter(u => u !== url);
    if (actualLabel === 'positive') tasteLabels.positive.push(url);
    if (actualLabel === 'negative') tasteLabels.negative.push(url);

    // Update button states in this card
    const card = btn.closest('.discovery');
    card.querySelectorAll('.taste-btn').forEach(b => {
      b.classList.remove('positive', 'negative');
    });
    if (actualLabel !== 'remove') {
      btn.classList.add(actualLabel);
    }

    loadTasteStatus();
  } catch (e) {
    console.error('taste label error:', e);
  }
}

async function trainTaste() {
  const btn = document.getElementById('btnTrain');
  const msg = document.getElementById('tasteMsg');
  btn.disabled = true;
  msg.textContent = 'training...';

  try {
    const res = await fetch(API + '/graphs/' + GRAPH_ID + '/taste/train', {
      method: 'POST',
    });
    const data = await res.json();

    if (data.trained) {
      msg.textContent = 'trained! accuracy: ' + (data.accuracy * 100).toFixed(0) + '% (' + data.n_positive + '+ / ' + data.n_negative + '-)';
      msg.style.color = '#81c784';
      // Reload discoveries with taste scoring
      discoveriesCache = { personalized: null, standard: null };
      loadDiscoveries();
    } else {
      msg.textContent = data.error || 'training failed';
      msg.style.color = '#ef5350';
    }
  } catch (e) {
    msg.textContent = 'error: ' + e.message;
    msg.style.color = '#ef5350';
  }

  btn.disabled = false;
  loadTasteStatus();
}

// ── Tuning System ──

let currentConfig = null;  // populated from discoveries response
let rescoreTimer = null;

function onSlider(el) {
  // Update the value display
  const id = el.id.replace('sl-', 'sv-');
  document.getElementById(id).textContent = parseFloat(el.value).toFixed(1);
  updateFormulaDisplay();
  scheduleRescore();
  // Clear preset highlights
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
}

function updateFormulaDisplay() {
  const pr = parseFloat(document.getElementById('sl-pagerank').value);
  const q = parseFloat(document.getElementById('sl-quality').value);
  const sw = parseFloat(document.getElementById('sl-smallweb').value);
  const tw = parseFloat(document.getElementById('sl-taste-weight').value);
  const expStr = (name, exp) => exp === 1.0 ? name : name + '^' + exp.toFixed(1);
  let parts = [expStr('pagerank', pr), expStr('quality', q), expStr('smallweb', sw)];
  const tasteRow = document.getElementById('sl-taste-row');
  if (tasteRow && tasteRow.style.display !== 'none') {
    parts.push('(' + (1 - tw).toFixed(1) + ' + ' + tw.toFixed(1) + ' × taste)');
  }
  document.getElementById('tuneFormula').textContent = 'score = ' + parts.join(' × ');
}

function collectSliderValues() {
  return {
    formula: {
      pagerank_exp: parseFloat(document.getElementById('sl-pagerank').value),
      quality_exp: parseFloat(document.getElementById('sl-quality').value),
      smallweb_exp: parseFloat(document.getElementById('sl-smallweb').value),
      taste_weight: parseFloat(document.getElementById('sl-taste-weight').value),
      taste_base: 1.0 - parseFloat(document.getElementById('sl-taste-weight').value),
      fetched_boost: parseFloat(document.getElementById('sl-fetched').value),
    }
  };
}

function setSliders(cfg) {
  const f = (cfg && cfg.formula) || {};
  document.getElementById('sl-pagerank').value = f.pagerank_exp != null ? f.pagerank_exp : 1.0;
  document.getElementById('sv-pagerank').textContent = parseFloat(document.getElementById('sl-pagerank').value).toFixed(1);
  document.getElementById('sl-quality').value = f.quality_exp != null ? f.quality_exp : 1.0;
  document.getElementById('sv-quality').textContent = parseFloat(document.getElementById('sl-quality').value).toFixed(1);
  document.getElementById('sl-smallweb').value = f.smallweb_exp != null ? f.smallweb_exp : 1.0;
  document.getElementById('sv-smallweb').textContent = parseFloat(document.getElementById('sl-smallweb').value).toFixed(1);
  document.getElementById('sl-taste-weight').value = f.taste_weight != null ? f.taste_weight : 0.5;
  document.getElementById('sv-taste-weight').textContent = parseFloat(document.getElementById('sl-taste-weight').value).toFixed(1);
  document.getElementById('sl-fetched').value = f.fetched_boost != null ? f.fetched_boost : 0.2;
  document.getElementById('sv-fetched').textContent = parseFloat(document.getElementById('sl-fetched').value).toFixed(2);
  updateFormulaDisplay();
}

function scheduleRescore() {
  clearTimeout(rescoreTimer);
  rescoreTimer = setTimeout(() => {
    // Show loading state
    const container = document.getElementById('discoveriesContainer');
    container.innerHTML = '<div class="loading"><span class="spinner"></span>re-scoring with new weights...</div>';
    // Clear cache and reload with config overrides
    discoveriesCache = { personalized: null, standard: null };
    loadDiscoveries();
  }, 300);
}

async function applyPreset(name) {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  if (typeof event !== 'undefined' && event && event.target) event.target.classList.add('active');
  if (name === 'default') {
    setSliders({ formula: { pagerank_exp: 1.0, quality_exp: 1.0, smallweb_exp: 1.0, taste_weight: 0.5, fetched_boost: 0.2 } });
  } else {
    try {
      const res = await fetch(API + '/graphs/' + GRAPH_ID + '/config/presets');
      const data = await res.json();
      const preset = data.presets[name];
      if (preset) {
        // Start from defaults, apply preset
        const base = { pagerank_exp: 1.0, quality_exp: 1.0, smallweb_exp: 1.0, taste_weight: 0.5, fetched_boost: 0.2 };
        const merged = Object.assign({}, base, preset.formula || {});
        setSliders({ formula: merged });
      }
    } catch (e) {
      console.error('preset error:', e);
    }
  }
  scheduleRescore();
}

async function saveConfig() {
  const overrides = collectSliderValues();
  const msg = document.getElementById('tuneMsg');
  try {
    const res = await fetch(API + '/graphs/' + GRAPH_ID + '/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(overrides),
    });
    const data = await res.json();
    if (data.ok) {
      msg.textContent = 'saved!';
      msg.style.color = '#81c784';
      currentConfig = data.config;
    } else {
      msg.textContent = data.error || 'save failed';
      msg.style.color = '#ef5350';
    }
  } catch (e) {
    msg.textContent = 'error: ' + e.message;
    msg.style.color = '#ef5350';
  }
  setTimeout(() => { msg.textContent = ''; }, 3000);
}

function resetConfig() {
  setSliders({ formula: { pagerank_exp: 1.0, quality_exp: 1.0, smallweb_exp: 1.0, taste_weight: 0.5, fetched_boost: 0.2 } });
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('.preset-btn').classList.add('active');
  scheduleRescore();
}

// ── Init ──
loadDiscoveries();
loadTasteStatus();
</script>
</body>
</html>"""


def render_html(graph: WebGraph, graph_file: str = "graph.json", graph_id: str = "") -> str:
    """Render the graph as an HTML page."""
    from string import Template
    import html as html_module

    stats = graph.stats()

    # Derive graph_id from graph_file if not provided
    if not graph_id:
        graph_id = Path(graph_file).stem

    seeds_html = ""
    for seed in sorted(graph.seeds):
        node = graph.nodes.get(seed, {})
        title = html_module.escape(node.get("title") or seed)
        seeds_html += f'<li><a href="{seed}" target="_blank">{title}</a></li>\n'

    domains = graph.domains()
    domains_html = ""
    for domain, count in list(domains.items())[:30]:
        domains_html += f'<span class="domain-tag">{html_module.escape(domain)} ({count})</span>\n'

    return Template(SERVE_HTML).safe_substitute(
        name=html_module.escape(graph.metadata.get("name", "unnamed")),
        description=html_module.escape(graph.metadata.get("description", "")),
        n_nodes=stats["nodes"],
        n_edges=stats["edges"],
        n_seeds=stats["seeds"],
        n_domains=stats["domains"],
        seeds_html=seeds_html,
        domains_html=domains_html,
        graph_id=graph_id,
        graph_file=graph_file,
    )


async def serve(graph: WebGraph, port: int = 8080, graph_file: str = "graph.json"):
    """Serve the web UI."""
    from aiohttp import web

    html = render_html(graph, graph_file)

    async def handle(request):
        return web.Response(text=html, content_type="text/html")

    async def handle_json(request):
        return web.json_response(graph.to_json())

    app = web.Application()
    app.router.add_get("/", handle)
    app.router.add_get("/graph.json", handle_json)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"\nsmallweb serving at http://localhost:{port}")
    print(f"graph JSON at http://localhost:{port}/graph.json")
    print("press ctrl+c to stop\n")

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


# ── CLI ───────────────────────────────────────────────────────────────

def parse_seeds(arg: str) -> List[str]:
    """Parse seeds from a comma-separated string or file."""
    path = Path(arg)
    if path.exists():
        text = path.read_text()
        # Try JSON first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            # It's a graph file - use its seeds
            if "seeds" in data:
                return data["seeds"]
        except json.JSONDecodeError:
            pass
        # Plain text, one URL per line
        return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    else:
        return [s.strip() for s in arg.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="smallweb - niche web discovery engine")
    subparsers = parser.add_subparsers(dest="command")

    # crawl
    crawl_p = subparsers.add_parser("crawl", help="Crawl from seed URLs")
    crawl_p.add_argument("seeds", help="Comma-separated URLs, file of URLs, or existing graph.json")
    crawl_p.add_argument("--hops", "-n", type=int, default=2, help="Max hops from seeds (default: 2)")
    crawl_p.add_argument("--max-pages", "-m", type=int, default=200, help="Max pages to crawl (default: 200)")
    crawl_p.add_argument("--output", "-o", default="graph.json", help="Output file (default: graph.json)")
    crawl_p.add_argument("--name", default="", help="Name for this graph")
    crawl_p.add_argument("--concurrent", "-c", type=int, default=10, help="Concurrent requests (default: 10)")
    crawl_p.add_argument("--domain-cap", "-d", type=int, default=20, help="Max pages per domain (default: 20, 0=unlimited)")

    # rank
    rank_p = subparsers.add_parser("rank", help="Show PageRank of all pages")
    rank_p.add_argument("graph", help="Graph JSON file")
    rank_p.add_argument("--top", "-n", type=int, default=20, help="Top N results")
    rank_p.add_argument("--damping", type=float, default=0.95, help="PageRank damping factor (default: 0.95, higher=deeper)")
    rank_p.add_argument("--iterations", type=int, default=50, help="PageRank iterations (default: 50)")

    # discover
    disc_p = subparsers.add_parser("discover", help="Show top discoveries (non-seed pages)")
    disc_p.add_argument("graph", help="Graph JSON file")
    disc_p.add_argument("--top", "-n", type=int, default=20, help="Top N results")
    disc_p.add_argument("--damping", type=float, default=0.95, help="PageRank damping factor (default: 0.95)")
    disc_p.add_argument("--iterations", type=int, default=50, help="PageRank iterations (default: 50)")

    # fork
    fork_p = subparsers.add_parser("fork", help="Fork a graph with new params/seeds")
    fork_p.add_argument("graph", help="Graph JSON file to fork")
    fork_p.add_argument("--output", "-o", default="", help="Output file")
    fork_p.add_argument("--name", default="", help="Name for forked graph")
    fork_p.add_argument("--author", default="", help="Author of the fork")
    fork_p.add_argument("--add-seeds", nargs="+", default=[], help="URLs to add as seeds")
    fork_p.add_argument("--promote-top", type=int, default=0,
                         help="Auto-promote top N discoveries to seeds (e.g. --promote-top 10)")
    fork_p.add_argument("--recrawl", action="store_true",
                         help="After forking, re-crawl from all seeds (including promoted ones)")
    fork_p.add_argument("--hops", type=int, default=2, help="Hops for re-crawl (default: 2)")
    fork_p.add_argument("--max-pages", type=int, default=200, help="Max pages for re-crawl (default: 200)")
    fork_p.add_argument("--domain-cap", type=int, default=20, help="Domain cap for re-crawl (default: 20)")
    fork_p.add_argument("--damping", type=float, default=0.95, help="PageRank damping for this fork (default: 0.95)")
    fork_p.add_argument("--iterations", type=int, default=50, help="PageRank iterations (default: 50)")

    # merge
    merge_p = subparsers.add_parser("merge", help="Merge two graphs")
    merge_p.add_argument("graph_a", help="First graph")
    merge_p.add_argument("graph_b", help="Second graph")
    merge_p.add_argument("--output", "-o", default="merged.json", help="Output file")
    merge_p.add_argument("--name", default="", help="Name for merged graph")

    # info
    info_p = subparsers.add_parser("info", help="Show graph stats")
    info_p.add_argument("graph", help="Graph JSON file")

    # serve
    serve_p = subparsers.add_parser("serve", help="Serve web UI")
    serve_p.add_argument("graph", help="Graph JSON file")
    serve_p.add_argument("--port", "-p", type=int, default=8080, help="Port (default: 8080)")

    # similar (find sites similar to a given URL/domain)
    sim_p = subparsers.add_parser("similar", help="Find similar sites via co-citation")
    sim_p.add_argument("graph", help="Graph JSON file")
    sim_p.add_argument("target", help="URL or domain to find similar sites for")
    sim_p.add_argument("--top", "-n", type=int, default=20, help="Top N results")

    # similarities (find all similar pairs in graph)
    sims_p = subparsers.add_parser("similarities", help="Find all similar domain pairs")
    sims_p.add_argument("graph", help="Graph JSON file")
    sims_p.add_argument("--min-shared", type=int, default=2, help="Min shared sources (default: 2)")
    sims_p.add_argument("--top", "-n", type=int, default=50, help="Top N pairs")

    # export-html
    html_p = subparsers.add_parser("html", help="Export static HTML")
    html_p.add_argument("graph", help="Graph JSON file")
    html_p.add_argument("--output", "-o", default="", help="Output HTML file")

    args = parser.parse_args()

    if args.command == "crawl":
        seeds = parse_seeds(args.seeds)
        if not seeds:
            print("no seeds found!")
            sys.exit(1)
        name = args.name or Path(args.output).stem
        print(f"crawling from {len(seeds)} seeds, max {args.hops} hops, max {args.max_pages} pages, domain cap {args.domain_cap}")
        print(f"seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''}\n")
        graph = asyncio.run(crawl(seeds, max_hops=args.hops, max_pages=args.max_pages,
                                  concurrent=args.concurrent, name=name,
                                  domain_cap=args.domain_cap))
        graph.save(args.output)

    elif args.command == "rank":
        graph = WebGraph.load(args.graph)
        ranks = graph.pagerank(damping=args.damping, iterations=args.iterations)
        print(f"top {args.top} pages by local pagerank (damping={args.damping}, iter={args.iterations}):\n")
        for i, (url, score) in enumerate(list(ranks.items())[:args.top]):
            node = graph.nodes.get(url, {})
            title = node.get("title", "")
            seed_marker = " [SEED]" if url in graph.seeds else ""
            print(f"  {i+1:3d}. {score:.6f} {title[:60] or url[:60]}{seed_marker}")
            if title:
                print(f"       {url}")

    elif args.command == "discover":
        graph = WebGraph.load(args.graph)
        discoveries = graph.discoveries(top_n=args.top, damping=args.damping, iterations=args.iterations)
        print(f"top {args.top} discoveries (damping={args.damping}, iter={args.iterations}):\n")
        for i, (url, score, node) in enumerate(discoveries):
            title = node.get("title", "")
            desc = node.get("description", "")
            print(f"  {i+1:3d}. {score:.6f} {title[:60] or url[:60]}")
            print(f"       {url}")
            if desc:
                print(f"       {desc[:80]}")
            print()

    elif args.command == "fork":
        graph = WebGraph.load(args.graph)
        forked = graph.fork(
            name=args.name,
            author=args.author,
            add_seeds=args.add_seeds if args.add_seeds else None,
            promote_top_n=args.promote_top,
        )

        # Store pagerank params in metadata so they persist with the graph
        forked.metadata["damping"] = args.damping
        forked.metadata["iterations"] = args.iterations

        output = args.output or f"fork-{Path(args.graph).stem}.json"

        # Show what changed
        original_seeds = len(graph.seeds)
        new_seeds = len(forked.seeds)
        if new_seeds > original_seeds:
            print(f"seeds: {original_seeds} → {new_seeds} (+{new_seeds - original_seeds} new)")
            for seed in sorted(forked.seeds - graph.seeds):
                node = graph.nodes.get(seed, {})
                print(f"  + {node.get('title', seed)[:60]}")
        print(f"damping: {args.damping}, iterations: {args.iterations}")

        if args.recrawl:
            # Re-crawl from the forked graph's seeds with new params
            print(f"\nre-crawling from {new_seeds} seeds...")
            new_graph = asyncio.run(crawl(
                list(forked.seeds),
                max_hops=args.hops,
                max_pages=args.max_pages,
                name=forked.metadata.get("name", "fork"),
                domain_cap=args.domain_cap,
            ))
            # Preserve fork provenance in the new graph
            new_graph.metadata["forked_from"] = forked.metadata.get("forked_from", "unknown")
            new_graph.metadata["forked_at"] = forked.metadata.get("forked_at", "")
            new_graph.metadata["damping"] = args.damping
            new_graph.metadata["iterations"] = args.iterations
            if forked.metadata.get("seeds_promoted"):
                new_graph.metadata["seeds_promoted"] = forked.metadata["seeds_promoted"]
            new_graph.save(output)
        else:
            forked.save(output)

    elif args.command == "merge":
        graph_a = WebGraph.load(args.graph_a)
        graph_b = WebGraph.load(args.graph_b)
        merged = WebGraph.merge(graph_a, graph_b, name=args.name)
        merged.save(args.output)

    elif args.command == "info":
        graph = WebGraph.load(args.graph)
        stats = graph.stats()
        print(f"graph: {stats['name']}")
        print(f"  nodes:   {stats['nodes']}")
        print(f"  edges:   {stats['edges']}")
        print(f"  seeds:   {stats['seeds']}")
        print(f"  domains: {stats['domains']}")
        print(f"  avg outlinks: {stats['avg_outlinks']:.1f}")
        print(f"  created: {stats['created_at']}")
        if graph.metadata.get("forked_from"):
            print(f"  forked from: {graph.metadata['forked_from']}")
        if graph.metadata.get("merged_from"):
            print(f"  merged from: {graph.metadata['merged_from']}")
        print(f"\ntop domains:")
        for domain, count in list(graph.domains().items())[:10]:
            print(f"  {count:4d}  {domain}")

    elif args.command == "serve":
        graph = WebGraph.load(args.graph)
        asyncio.run(serve(graph, port=args.port, graph_file=args.graph))

    elif args.command == "similar":
        graph = WebGraph.load(args.graph)
        results = graph.similar_sites(args.target, top_n=args.top)
        target_display = urlparse(args.target).netloc if "://" in args.target else args.target
        if not results:
            print(f"no similar sites found for {target_display}")
            print("(needs cross-domain links — try crawling with more hops)")
        else:
            print(f"sites similar to {target_display} (by co-citation):\n")
            for i, (domain, cosine, shared) in enumerate(results):
                print(f"  {i+1:3d}. {cosine:.3f}  {domain}  ({shared} shared sources)")

    elif args.command == "similarities":
        graph = WebGraph.load(args.graph)
        pairs = graph.all_similarities(min_shared=args.min_shared, top_n=args.top)
        if not pairs:
            print(f"no similar pairs found (min {args.min_shared} shared sources)")
        else:
            print(f"similar domain pairs (min {args.min_shared} shared sources):\n")
            for i, (d_a, d_b, cosine, shared) in enumerate(pairs):
                print(f"  {i+1:3d}. {cosine:.3f}  {d_a}  ↔  {d_b}  ({shared} shared)")

    elif args.command == "html":
        graph = WebGraph.load(args.graph)
        html = render_html(graph, args.graph)
        output = args.output or f"{Path(args.graph).stem}.html"
        Path(output).write_text(html)
        print(f"saved HTML to {output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
