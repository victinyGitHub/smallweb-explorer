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
        """
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
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def discoveries(self, top_n: int = 20, damping: float = 0.95,
                     iterations: int = 50, use_quality: bool = True,
                     personalized: bool = True) -> List[Tuple[str, float, dict]]:
        """
        Return top-ranked pages that are NOT seeds.
        These are the things you discovered by crawling outward.

        The final score combines multiple signals:
        - PageRank (link authority, optionally personalized toward seeds)
        - Quality (HTML cleanliness: scripts, trackers, text ratio)
        - Smallweb score (data-driven: popularity bell curve × outlink profile)

        This means: a page needs good link authority AND clean HTML AND
        moderate popularity AND links to other small sites to rank highly.

        Args:
            top_n:        Number of discoveries to return
            damping:      PageRank damping factor (0.95 = follow links deep,
                          0.5 = stay close to seeds)
            iterations:   PageRank iterations (50 is usually plenty)
            use_quality:  Multiply by quality + smallweb scores
            personalized: Use personalized pagerank (biased toward seeds)
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
        sw_scores = self._smallweb_scores() if use_quality else {}

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

            if use_quality:
                q = self.nodes[url].get("quality", 1.0)
                sw = sw_scores.get(domain, {}).get("smallweb_score", 0.5)
                final_score = score * q * sw

                # Store smallweb metadata on the node for API/frontend access
                sw_data = sw_scores.get(domain, {})
                self.nodes[url]["smallweb_score"] = sw
                self.nodes[url]["inbound_domains"] = sw_data.get("inbound_domains", 0)
                self.nodes[url]["outlink_score"] = sw_data.get("outlink_score", 0.5)
                self.nodes[url]["popularity_score"] = sw_data.get("popularity_score", 0.5)
            else:
                final_score = score

            results.append((url, final_score, self.nodes[url]))
        # Re-sort by final score after quality weighting
        results.sort(key=lambda x: x[1], reverse=True)
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

    def _outlink_profile(self, domain: str, graph_domains: Set[str] = None) -> float:
        """
        Compute an "outlink profile" score for a domain (0.0 to 1.0).

        Uses two signals:
        1. What fraction of outlinks point to domains IN our graph (ecosystem links)
        2. What fraction of outlinks point to known platforms (platform links)

        A site that links to other sites we've discovered is more "part of our web."
        A site that links entirely outside our graph is likely a platform or unrelated.

        Returns 1.0 for sites deeply integrated in our graph's ecosystem.
        Returns 0.0 for sites that only link to platforms/outside our graph.
        Returns 0.5 (neutral) if no outlink data is available.
        """
        # Aggregate all outlinks from all pages on this domain
        external_targets = []
        for url, targets in self.edges.items():
            if urlparse(url).netloc.lower() == domain:
                for target in targets:
                    td = urlparse(target).netloc.lower()
                    if td != domain:  # external only
                        external_targets.append(td)

        if not external_targets:
            return 0.5  # no data = neutral

        total = len(external_targets)

        # Signal 1: ecosystem integration - links to domains in our graph
        if graph_domains:
            in_graph = sum(1 for td in external_targets if td in graph_domains)
            ecosystem_fraction = in_graph / total
        else:
            ecosystem_fraction = 0.5

        # Signal 2: platform avoidance - links NOT to known platforms
        platform_count = sum(1 for td in external_targets if is_platform_domain(td))
        non_platform_fraction = 1.0 - (platform_count / total)

        # Combined: ecosystem is the stronger signal (0.7), platform avoidance weaker (0.3)
        score = 0.7 * ecosystem_fraction + 0.3 * non_platform_fraction

        return round(score, 3)

    def _smallweb_scores(self) -> Dict[str, dict]:
        """
        Compute per-domain "smallweb-ness" scores using data-driven signals.

        Returns dict of domain -> {
            inbound_domains: int,   # how many domains link to this one
            popularity_score: float, # bell curve peaking at moderate popularity
            outlink_score: float,    # how much it links into our ecosystem
            smallweb_score: float,   # combined score 0.0-1.0
        }
        """
        inbound = self._build_inbound_index()

        # Build set of all domains in graph for ecosystem scoring
        scores = {}
        all_domains = set()
        for url in self.nodes:
            all_domains.add(urlparse(url).netloc.lower())

        if not all_domains:
            return {}

        for domain in all_domains:
            n_inbound = len(inbound.get(domain, set()))
            outlink = self._outlink_profile(domain, graph_domains=all_domains)

            # Popularity curve: log-normal-ish bell curve
            # Peaks at ~4 inbound domains, drops off on both sides
            if n_inbound == 0:
                pop_score = 0.4  # orphan pages are suspect
            elif n_inbound <= 2:
                pop_score = 0.7  # few links, possibly real
            elif n_inbound <= 8:
                pop_score = 1.0  # sweet spot
            elif n_inbound <= 15:
                pop_score = 0.6  # getting popular
            elif n_inbound <= 30:
                pop_score = 0.3  # quite popular, probably not small web
            elif n_inbound <= 60:
                pop_score = 0.15  # very popular
            else:
                pop_score = 0.05  # platform-level

            # Combined: popularity × outlink profile
            # Both signals matter: a site should be moderately popular
            # AND link to other small sites in our ecosystem
            combined = pop_score * (0.5 + 0.5 * outlink)  # outlink moderates, doesn't dominate

            # Safety net: if domain is a known platform, cap the score
            # This isn't the primary filter (data-driven signals are), but prevents
            # edge cases where a platform has just 3 inbound links in a small graph
            if is_platform_domain(domain):
                combined = min(combined, 0.15)

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

        inbound = self._build_inbound_index()
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


def quality_score(html: str, soup: BeautifulSoup) -> float:
    """
    Compute a simple quality score for a page (0.0 to 1.0).

    Signals (from Marginalia's approach):
    - Script tag count: many scripts = commercial/heavy
    - Text-to-HTML ratio: low ratio = boilerplate-heavy
    - Link density: high ratio of links to text = link farm
    - Tracker detection: known tracking domains in script srcs

    Returns a float between 0.0 (spam) and 1.0 (clean content).
    """
    text = soup.get_text()
    text_len = len(text)
    html_len = max(len(html), 1)

    # Script count
    scripts = soup.find_all("script")
    script_count = len(scripts)

    # External script domains (tracker signal)
    tracker_domains = {
        "google-analytics.com", "googletagmanager.com", "facebook.net",
        "doubleclick.net", "hotjar.com", "segment.com", "mixpanel.com",
        "amplitude.com", "optimizely.com", "crazyegg.com", "hubspot.com",
    }
    external_trackers = 0
    for s in scripts:
        src = s.get("src", "")
        if src:
            src_domain = urlparse(src).netloc.lower()
            for tracker in tracker_domains:
                if tracker in src_domain:
                    external_trackers += 1
                    break

    # Text density
    text_ratio = text_len / html_len

    # Link density (outlinks per word)
    outlinks = len(soup.find_all("a", href=True))
    text_words = max(len(text.split()), 1)
    link_density = outlinks / text_words

    # Scoring
    score = 1.0

    # Penalize script-heavy pages
    if script_count > 15:
        score *= 0.3
    elif script_count > 10:
        score *= 0.5
    elif script_count > 5:
        score *= 0.8

    # Penalize tracker-laden pages
    if external_trackers >= 3:
        score *= 0.4
    elif external_trackers >= 1:
        score *= 0.7

    # Penalize boilerplate-heavy pages
    if text_ratio < 0.05:
        score *= 0.4
    elif text_ratio < 0.1:
        score *= 0.6

    # Penalize link farms
    if link_density > 0.5:
        score *= 0.3
    elif link_density > 0.3:
        score *= 0.6

    return round(max(0.0, min(1.0, score)), 3)


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
    "en.wikipedia.org", "wikipedia.org", "wikidata.org",
    "wikimedia.org", "archive.org", "web.archive.org",
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

                # Skip if this domain has hit its cap
                if domain_cap > 0 and domain_counts[domain] >= domain_cap:
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

                # Compute quality score
                soup = BeautifulSoup(html, "html.parser")
                q_score = quality_score(html, soup)

                graph.add_node(final_url, title=title, description=description, depth=depth)
                graph.nodes[graph._normalize_url(final_url)]["quality"] = q_score

                # Add edges, queue new URLs, and store anchor texts
                for link in links:
                    graph.add_edge(final_url, link)
                    link_norm = graph._normalize_url(link)

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
                        if link_norm not in graph.nodes:
                            graph.add_node(link, depth=depth + 1)
                        queue.append((link, depth + 1))

                # Show progress with domain count
                cap_info = f" [{domain}: {domain_counts[domain]}/{domain_cap}]" if domain_cap > 0 else ""
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

  /* Smallweb score indicator */
  .sw-indicator { display: flex; align-items: center; gap: 0.3rem; margin-top: 0.25rem; }
  .sw-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .sw-label { font-size: 0.6rem; color: #555; }
  .sw-details { font-size: 0.58rem; color: #3a3a3a; margin-top: 0.1rem; }

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
  </div>

  <!-- Tab: Discoveries -->
  <div class="tab-panel active" id="tab-discoveries">
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
          <button class="toggle-btn active" data-sort="score" onclick="switchSort('score')">pagerank</button>
          <button class="toggle-btn" data-sort="quality" onclick="switchSort('quality')">quality</button>
          <button class="toggle-btn" data-sort="smallweb" onclick="switchSort('smallweb')">smallweb</button>
          <button class="toggle-btn" data-sort="blended" onclick="switchSort('blended')">blended</button>
        </div>
      </div>
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

  <footer>
    <a href="/smallweb/">smallweb</a> &mdash; graph: $graph_id
  </footer>

<script>
const GRAPH_ID = '$graph_id';
const API = '/smallweb/api';

let currentSort = 'score';
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
    const url = API + '/graphs/' + GRAPH_ID + '/discoveries?top=50&personalized=' + personalized;
    const res = await fetch(url);
    const data = await res.json();
    discoveriesCache[key] = data.discoveries || [];
    renderDiscoveries(discoveriesCache[key]);
  } catch (e) {
    container.innerHTML = '<div class="empty">error loading discoveries: ' + e.message + '</div>';
  }
}

function renderDiscoveries(discoveries) {
  const container = document.getElementById('discoveriesContainer');

  // Sort
  let sorted = [...discoveries];
  if (currentSort === 'quality') {
    sorted.sort((a, b) => (b.quality || 0) - (a.quality || 0));
  } else if (currentSort === 'smallweb') {
    sorted.sort((a, b) => (b.smallweb_score || 0) - (a.smallweb_score || 0));
  } else if (currentSort === 'blended') {
    sorted.sort((a, b) => ((b.score || 0) * (b.quality || 1) * (b.smallweb_score || 0.5)) - ((a.score || 0) * (a.quality || 1) * (a.smallweb_score || 0.5)));
  }
  // 'score' is already default order from API (pagerank * quality * smallweb)

  if (sorted.length === 0) {
    container.innerHTML = '<div class="empty">no discoveries yet &mdash; try crawling with more hops</div>';
    return;
  }

  container.innerHTML = sorted.map((d, i) => {
    const title = d.title || new URL(d.url).hostname;
    const quality = d.quality != null ? d.quality : 1.0;
    const qColor = quality >= 0.7 ? '#81c784' : quality >= 0.4 ? '#ffb74d' : '#ef5350';
    const qPct = Math.round(quality * 100);
    const domain = d.domain || new URL(d.url).hostname;
    const sw = d.smallweb_score != null ? d.smallweb_score : 0.5;
    const swColor = sw >= 0.7 ? '#81c784' : sw >= 0.4 ? '#ffb74d' : sw >= 0.15 ? '#ef5350' : '#666';
    const inbound = d.inbound_domains || 0;
    const outlink = d.outlink_score != null ? Math.round(d.outlink_score * 100) : '?';

    // Anchor text pills (max 5)
    const anchors = (d.anchor_texts || []).slice(0, 5);
    const anchorHtml = anchors.length > 0
      ? '<div class="anchor-texts">' + anchors.map(a => '<span class="anchor-pill" title="' + escHtml(a) + '">' + escHtml(a) + '</span>').join('') + '</div>'
      : '';

    return '<div class="discovery">' +
      '<div class="discovery-header">' +
        '<div class="discovery-main">' +
          '<a class="discovery-title" href="' + d.url + '" target="_blank">' + escHtml(title) + '</a>' +
          '<div class="discovery-url">' + escHtml(d.url) + '</div>' +
          (d.description ? '<div class="discovery-desc">' + escHtml(d.description.slice(0, 150)) + '</div>' : '') +
          anchorHtml +
          '<button class="btn-similar" onclick="findSimilarFrom(\\'' + escJs(domain) + '\\')">find similar &rarr;</button>' +
        '</div>' +
        '<div style="text-align:right">' +
          '<div class="quality-bar-wrap" title="quality: ' + qPct + '%">' +
            '<div class="quality-bar"><div class="quality-bar-fill" style="width:' + qPct + '%;background:' + qColor + '"></div></div>' +
            '<span class="quality-label" style="color:' + qColor + '">' + qPct + '</span>' +
          '</div>' +
          '<div class="sw-indicator" title="smallweb score: ' + sw.toFixed(2) + ' | inbound: ' + inbound + ' domains | outlinks: ' + outlink + '% small">' +
            '<span class="sw-dot" style="background:' + swColor + '"></span>' +
            '<span class="sw-label" style="color:' + swColor + '">sw ' + sw.toFixed(2) + '</span>' +
          '</div>' +
          '<div class="sw-details">' + inbound + ' in · ' + outlink + '% small</div>' +
          '<div class="score-badge">#' + (i + 1) + ' &middot; ' + d.score.toFixed(4) + '</div>' +
        '</div>' +
      '</div>' +
    '</div>';
  }).join('');
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

// ── Init ──
loadDiscoveries();
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
