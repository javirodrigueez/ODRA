# ODRA — project page

Static project page for **ODRA: Synthesizing Cognitive Behavioral Therapy Sessions with
Structured Chain-Of-Thought and Dynamic Patient Resistance**
([arXiv:2608.04524](https://arxiv.org/abs/2608.04524)).

Node-only toolchain — no Python, no bash. Requires **Node 20+** (developed on 26.7).

## Quick start

```bash
npm install
npm run dev
```

`dev` runs Tailwind in watch mode and a static server together, then prints a
`http://127.0.0.1:8731/` URL. The explorer fetches JSON, so opening `index.html`
over `file://` will not work — use the server.

## Deploying to `javirodrigueez.github.io/ODRA`

Copy this folder into the [ODRA repo](https://github.com/javirodrigueez/ODRA) as `docs/`,
then **Settings → Pages → Deploy from a branch → `main` / `/docs`**. Nothing else is
required: `assets/css/style.css` is committed, so the page works without a build step.

The included workflow (`.github/workflows/deploy.yml`) is optional — use it only if you
prefer CI to recompile the CSS on every push. Move it to the repository root if you do.

`.nojekyll` is present so GitHub serves `assets/` verbatim rather than running Jekyll.

## Scripts

| command | what it does |
| --- | --- |
| `npm run dev` | Tailwind watch + static server, one terminal |
| `npm run serve` | static server only — `npm run serve -- 9000` to pick a port |
| `npm run watch` | Tailwind watch only |
| `npm run build` | compile `src/input.css` → `assets/css/style.css` (minified) |
| `npm run data` | regenerate `assets/data/` from the raw simulation dumps |
| `npm run figures` | rasterise `../latex/figures/*.pdf` → `assets/img/` |
| `npm run card` | regenerate the Open Graph card |
| `npm run assets` | `data` + `figures` + `card` |
| `npm run all` | `assets` + `build` — everything, from scratch |

## Layout

```
index.html              the whole page (no templating)
src/input.css           Tailwind source + component layer
assets/css/style.css    compiled stylesheet — COMMITTED, rerun `npm run build` after edits
assets/js/data.js       fetch/escape/format helpers
assets/js/explorer.js   session explorer + sycophancy comparison
assets/js/page.js       result tabs, alignment chart, BibTeX copy
assets/img/             figures rasterised from latex/figures/*.pdf, plus the OG card
assets/data/            generated session JSON (see below)
tools/build-data.mjs    raw dumps      -> assets/data/
tools/build-figures.mjs latex/figures/ -> assets/img/
tools/social-card.mjs   -> assets/img/social-card.jpg
tools/serve.mjs         zero-dependency static server
tools/dev.mjs           runs watch + serve together
```

### Data pipeline

`tools/build-data.mjs` reads the raw simulation dumps in `../project-page/` (9 sessions:
3 profiles × 3 attitudes) and `../project-page-sycophancy/` (ODRA vs CACTUS vs MAGNeT on
one shared profile). It strips the full prompt text from every reasoning trace, keeping
the model's reasoning and the numeric fields, which takes the payload from ~5.6 MB to
~1.2 MB. Sessions are fetched lazily, one at a time.

Per turn it surfaces: patient resistance `r`, therapist estimate `p̂`, the sampled action
and its prior/contextual/fused distributions, any detected TIB, the active CBT stage, and
the decomposed resistance update (`previous_r`, `delta_llm`, `alpha`, `decay_vector`,
`new_r`) so the page can show the arithmetic of Equation 1 with real numbers.

### Figures — macOS dependency

`build-figures.mjs` and `social-card.mjs` shell out to the macOS `qlmanage` (render) and
`sips` (convert) binaries; both exit with a clear message on other platforms. On Linux,
install `poppler-utils` and replace the `qlmanage` call with:

```
pdftocairo -png -r 200 -singlefile in.pdf out
```

Paper figures keep a light plate in dark mode (`.figplate`) because they are
white-background raster.

## Things to update before launch

- **Javier's Google Scholar link.** The author list currently points to his GitHub
  profile; every other author links to Scholar.
- **Dataset and model links.** Both badges say "coming soon". Search `coming soon` and
  `— soon` in `index.html`.
- **Venue.** The page says "Under review". The GitHub repo description says "[EMNLP'26]" —
  reconcile the two.
- **Acknowledgments.** No funding statement is present; the paper's block is commented out.
- **Expert annotators.** The three licensed psychologists are unnamed. Add them if they
  consent to being credited.
