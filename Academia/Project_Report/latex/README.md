# M.Sc. project report (LaTeX)

Self-contained advanced-project report. The Open University cover page is `\input` from `Academia/MyPapersU/22997/Advanced_Project/cover.tex` (title, author, supervisor, date). Figures reuse assets under `Academia/images/` via `\graphicspath` in `main.tex`.

## Build

From this directory:

```bash
../compile.sh
```

or:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

The PDF is written to `main.pdf` here. Front matter includes **List of Images**, **List of Tables**, and **List of Algorithms**.

## Requirements

- Same engine resolution as `Academia/compile.sh`: **`latexmk`** or **`pdflatex`** on your `PATH`, **or** executable **`Academia/tectonic`**.
- For `pdflatex`: `bibtex` and packages `natbib`, `cleveref`, `algorithm`/`algpseudocode`, `float`, `tikz`, `booktabs`, `setspace`, `placeins`, `geometry`, `microtype`.
- Cover logo: `ou_logo.png` next to `cover.tex` (included via `\graphicspath`).

## Edit

- Cover: `MyPapersU/22997/Advanced_Project/cover.tex`.
- Body: `chapters/*.tex`, `appendices/*.tex`.
- Extra UI screenshots: `figures/ui/`.
- Bibliography: `references.bib`.

## UML diagram sources

- UML sources for the chapter-3 software-structure diagrams are stored as PlantUML files under:
  - `figures_report/uml/desktop_software_structure.puml`
  - `figures_report/uml/web_software_structure.puml`
- The report currently includes the rendered figure wrappers in:
  - `figures_report/fig_desktop_software_structure.tex`
  - `figures_report/fig_web_software_structure.tex`
