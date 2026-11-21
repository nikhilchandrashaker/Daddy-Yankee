# Daddy Yankee: Career Analytics

A portfolio analytics project on Daddy Yankee's recorded career, 1995-2025: an interactive dashboard with a searchable song table, a research paper with models and diagrams, and the underlying datasets.

## What is in this repo
| Folder | Contents |
|---|---|
| `docs/index.html` | Interactive dashboard (search bar, era filters, charts, collaborator network). Publish with GitHub Pages from `/docs`. |
| `paper/` | 7-page research paper (PDF): era analysis, OLS and logistic models, robustness checks, album sales and certifications, collaboration network, limitations. |
| `data/` | Songs (113), albums (16, with US sales and certifications where listed), collaboration edges, collaborator summary, press-reported streaming milestones. |
| `code/` | Python scripts that rebuild the data, models, figures and paper. |
| `figures/` | The six figures used in the paper. |

## Data dictionary (`data/daddy_yankee_songs.csv`)
`title`, `year`, `role` (L solo lead, C lead with collaborators, F featured), `album`, `hot100` (Billboard Hot 100 peak), `latin` (Hot Latin Songs peak), `cert` (selected certification), `era`, `top10`, `no1`.

## Method in brief
- OLS: Hot Latin peak on release year and role (charted songs only).
- Logistic regression: Latin top-10 entry on year and role (all songs).
- Robustness: Spearman correlation, bootstrap of the year slope, Kruskal-Wallis across eras, and a leave-out-2022 check.
- Collaboration network: co-appearance of credited collaborators on the same song.

## Headline findings
Peak position drifts about 0.6 places worse per year, though the drift weakens when 2022 releases are dropped; role (solo, collaborative, featured) shows no detectable effect; the logistic model does not beat the base rate. These are descriptive results on a small sample.

## Rebuild
```bash
cd code && mkdir -p figs
python m1.py     # songs/albums CSVs, models, figures
python m3.py     # collaboration edges and network figure
python m5.py     # robustness checks, album sales, certifications
python m2_pdf.py # paper PDF
```
Requires pandas, numpy, matplotlib, networkx, reportlab, pillow.

## Limitations
- Source: the public Wikipedia discography table (which cites Billboard, RIAA and others), hand-transcribed. Verify against Billboard and RIAA before citing.
- Singles and features only; album tracks that were never singles are not included.
- A blank Hot Latin cell is treated as not charting there.
- Streaming figures are press-reported milestones, not a time series. No weeks-on-chart data, so no chart-longevity model yet.
