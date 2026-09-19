"""Collect every artifact into one shareable page (`experiments/showcase/`).

The pieces exist separately — gameplay GIFs, dream-vs-real videos, learning
curves, the baseline comparison, ablations — but "show someone how this
works" needs them in one place, in an order that tells the story:

    1. it plays            -> gameplay GIFs (agent vs baselines)
    2. it plays *inside a learned model* -> real-vs-imagined videos
    3. and it learns faster than model-free baselines -> curves + summary
    4. and here is what each design choice bought -> ablations

This script never trains and never re-runs an agent: it regenerates the
plots from the benchmark CSVs (cheap, deterministic), copies whatever
videos already exist, and writes `index.html` + `SHOWCASE.md` around them.
Missing pieces are listed on the page with the exact command that produces
them, so a half-finished showcase says which half is missing instead of
quietly dropping it.

Usage:
    python viz/make_showcase.py
    python viz/make_showcase.py viz.showcase_dir=experiments/showcase \
        viz.benchmark_root=experiments/benchmark
"""

from __future__ import annotations

import html
import json
import shutil
from datetime import date
from pathlib import Path

import hydra
from omegaconf import DictConfig

CSS = """
:root { color-scheme: dark; }
body { margin:0; padding:2rem clamp(1rem,4vw,3rem); background:#0e1116; color:#e6edf3;
       font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.9rem; margin: 0 0 .3rem; }
h2 { font-size: 1.25rem; margin: 2.4rem 0 .4rem; border-bottom:1px solid #263041; padding-bottom:.3rem; }
p.lead, p.note { color:#9fb0c3; }
p.note { font-size:.9rem; }
figure { margin: 1rem 0 1.6rem; }
figure img { max-width:100%; border-radius:8px; background:#000; display:block; }
figcaption { color:#9fb0c3; font-size:.88rem; margin-top:.45rem; }
table { border-collapse: collapse; width:100%; font-size:.92rem; }
th, td { border:1px solid #263041; padding:.4rem .6rem; text-align:left; }
th { background:#161b22; }
code { background:#161b22; padding:.12rem .35rem; border-radius:4px; font-size:.88em; }
pre { background:#161b22; padding:.8rem 1rem; border-radius:8px; overflow-x:auto; font-size:.85rem; }
.missing { border-left:3px solid #d29922; background:#1c1a12; padding:.7rem 1rem; border-radius:0 8px 8px 0; }
"""

# section key -> (title, blurb, how to produce it)
SECTIONS = {
    "gameplay": (
        "1. It plays",
        "Trained agents acting greedily in the environment, each panel showing "
        "exactly the 64x64 input that agent receives. Same seeds for every agent.",
        "python viz/gameplay_gif.py '+viz.gameplay_agents=[\"dreamer=<ckpt>\",\"ppo=<ckpt>\",\"random\"]'",
    ),
    "imagination": (
        "2. It plays inside its own world model",
        "Left: what actually happened. Right: what the RSSM dreamed from the same "
        "latent state, decoded to pixels — prior-only, the path the policy is trained on.",
        "python viz/real_vs_imagined_video.py viz.video_ckpt=<ckpt>",
    ),
    "curves": (
        "3. It learns faster than the model-free baselines",
        "Episode return against environment steps (post action-repeat) — the "
        "sample-efficiency axis — and the wall-clock price of that efficiency.",
        "python viz/learning_curves.py && python viz/benchmark_comparison.py "
        "&& python viz/summary_figure.py",
    ),
    "ablations": (
        "4. What each design choice bought",
        "One Hydra override per variant, same budget and seeds as the base run.",
        "python viz/ablation_summary.py",
    ),
    "world_model": (
        "5. World-model diagnostics",
        "Reconstructions, open-loop rollouts and the dream-vs-real return check "
        "behind the numbers above.",
        "python viz/reconstruction.py viz.ckpt=<wm_ckpt> && "
        "python viz/open_loop_rollout.py viz.ckpt=<wm_ckpt>",
    ),
}


def collect(cfg: DictConfig) -> dict[str, list[Path]]:
    """Find the artifacts on disk, per section (sorted, deduplicated)."""
    v = cfg.viz
    root = Path(v.get("benchmark_root") or "experiments/benchmark")
    plots = root / "plots"
    found: dict[str, list[Path]] = {k: [] for k in SECTIONS}

    gameplay = Path(v.get("gameplay_out_dir") or "experiments/gameplay")
    if gameplay.is_dir():
        # the side-by-side strip first, it is the one worth looking at
        found["gameplay"] = sorted(gameplay.glob("*.gif"),
                                   key=lambda p: ("compare" not in p.name, p.name))
    videos = Path(v.get("video_out_dir") or "experiments/videos")
    if videos.is_dir():
        found["imagination"] = sorted(videos.glob("*.gif"))
    if plots.is_dir():
        found["curves"] = sorted(
            [p for p in plots.glob("*.png") if "ablation" not in p.name],
            key=lambda p: ("summary" not in p.name, p.name),
        )
        found["ablations"] = sorted(plots.glob("ablation_*.png"))
    for key in ("wm_out_dir", "out_dir"):
        d = Path(v.get(key) or "")
        if d.is_dir():
            found["world_model"] += sorted(
                [p for p in list(d.glob("*.png")) + list(d.glob("*.gif"))]
            )
    for run_dir in (v.get("run_dir"),):
        if run_dir and (Path(run_dir) / "dream_vs_real.png").exists():
            found["world_model"].append(Path(run_dir) / "dream_vs_real.png")
    return found


def regenerate_plots(root: Path, window: int) -> list[str]:
    """Rebuild the benchmark plots from the CSVs; report what could not run."""
    problems = []
    from viz.benchmark_comparison import make_plots
    from viz.learning_curves import make_learning_curves
    from viz.summary_figure import make_summary

    for name, fn in (("benchmark_comparison", make_plots),
                     ("learning_curves", make_learning_curves),
                     ("summary_figure", make_summary)):
        try:
            fn(root, window=window)
        except Exception as exc:  # no CSVs yet, or a single-episode run
            problems.append(f"{name}: {exc}")
    return problems


def _tables(plots: Path) -> list[tuple[str, str]]:
    """(title, markdown) for every summary/ablation table next to the plots."""
    out = []
    if plots.is_dir():
        for md in sorted(plots.glob("*_summary.md")) + sorted(plots.glob("ablation_*.md")):
            out.append((md.stem.replace("_", " "), md.read_text(encoding="utf-8")))
    return out


def _md_table_to_html(md: str) -> str:
    rows, notes = [], []
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("|") and not set(line) <= set("|- "):
            rows.append([c.strip() for c in line.strip("|").split("|")])
        elif line:
            notes.append(line)
    if not rows:
        return "<p>" + html.escape(" ".join(notes)) + "</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
        for r in rows[1:]
    )
    note = f"<p class='note'>{html.escape(' '.join(notes))}</p>" if notes else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}"


def build(cfg: DictConfig) -> Path:
    v = cfg.viz
    out = Path(v.get("showcase_dir") or "experiments/showcase")
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    root = Path(v.get("benchmark_root") or "experiments/benchmark")

    problems = []
    if bool(v.get("showcase_regenerate", True)):
        problems = regenerate_plots(root, int(v.get("dream_window", 10)))

    found = collect(cfg)
    limit = int(v.get("showcase_max_per_section", 6))
    body, md = [], [f"# {cfg.env.name} showcase", "",
                    f"Generated {date.today().isoformat()} by `viz/make_showcase.py` "
                    "— no training, only artifacts already on disk.", ""]

    gameplay_json = Path(v.get("gameplay_out_dir") or "experiments/gameplay") / "gameplay.json"
    meta = json.loads(gameplay_json.read_text(encoding="utf-8")) if gameplay_json.exists() else None

    for key, (title, blurb, command) in SECTIONS.items():
        files = found.get(key, [])[:limit]
        body.append(f"<h2>{html.escape(title)}</h2><p class='lead'>{html.escape(blurb)}</p>")
        md += [f"## {title}", "", blurb, ""]
        if not files:
            body.append(
                "<div class='missing'><p>Not generated yet. Produce it with:</p>"
                f"<pre>{html.escape(command)}</pre></div>"
            )
            md += ["_Not generated yet:_", "", f"```bash\n{command}\n```", ""]
            continue
        for src in files:
            dst = assets / src.name
            shutil.copyfile(src, dst)
            caption = _caption(key, src, meta)
            body.append(
                f"<figure><img src='assets/{html.escape(dst.name)}' "
                f"alt='{html.escape(src.stem)}'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
            md += [f"![{src.stem}](assets/{dst.name})", "", f"*{caption}*", ""]
        if key == "curves":
            for name, table in _tables(root / "plots"):
                body.append(f"<h3>{html.escape(name)}</h3>" + _md_table_to_html(table))
                md += [f"### {name}", "", table, ""]

    if problems:
        joined = "; ".join(problems)
        body.append("<div class='missing'><p>Plots that could not be regenerated: "
                    f"{html.escape(joined)}</p></div>")
        md += ["> Plots that could not be regenerated: " + joined, ""]

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(str(cfg.env.name))} — Dreamer showcase</title>"
        f"<style>{CSS}</style></head><body><main>"
        f"<h1>Dreamer (RSSM) on {html.escape(str(cfg.env.name))}</h1>"
        "<p class='lead'>Model-based RL: the agent learns a latent world model, "
        "trains its policy inside that model's imagination, and is compared "
        "against model-free baselines on identical environment steps.</p>"
        + "".join(body)
        + "<p class='note'>Every figure is generated from recorded runs; "
        "nothing on this page is hand-drawn or simulated.</p>"
        "</main></body></html>"
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "SHOWCASE.md").write_text("\n".join(md), encoding="utf-8")
    total = sum(len(f[:limit]) for f in found.values())
    print(f"[showcase] {total} artifacts -> {out / 'index.html'} (+ SHOWCASE.md)")
    return out / "index.html"


def _caption(key: str, src: Path, meta: dict | None) -> str:
    if key == "gameplay" and meta:
        agents = ", ".join(
            f"{a['label']}: return {a['kept_return']:+.1f} "
            f"(best of {len(a['all_returns'])})" for a in meta["agents"]
        )
        return f"{meta['env']}, action repeat {meta['action_repeat']} — {agents}"
    if key == "imagination":
        return f"{src.stem}: left = real environment, right = decoded imagination"
    return src.stem.replace("_", " ")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    build(cfg)


if __name__ == "__main__":
    main()
