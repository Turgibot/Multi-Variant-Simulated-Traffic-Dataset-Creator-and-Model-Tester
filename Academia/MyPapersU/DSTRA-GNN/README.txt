LaTeX Manuscript Package for Submission
========================================

This package contains all necessary files for the manuscript:
"Dynamic Route-Aware Graph Neural Networks for Accurate ETA Prediction"

MAIN FILES:
- main.tex          : Main LaTeX document
- references.bib    : Bibliography database
- IEEEtran.cls     : IEEE LaTeX class file

DIRECTORIES:
- sections/         : Individual section files
  - abstract.tex
  - keywords.tex
  - introduction.tex
  - related_work.tex
  - methodology.tex
  - experiments.tex
  - results.tex
  - discussion.tex
  - conclusion.tex
  - acknowledgments.tex

- figures/          : All figures and TikZ diagrams
  - dynamic_nodes_edges.png
  - eta_box.png
  - eta_dist.png
  - eta_per_hour.png
  - fig_ablation_progression.tikz
  - fig_ablation_temporal_branch.tikz
  - fig_encoder.tikz
  - fig_model_swimlane.tikz
  - fig_model_twopanel.tikz
  - fig_route_encoder.tikz
  - fig_router_moe.tikz
  - fig_temporal_moe_eta.tikz
  - fig_temporal_transformer_detail.tikz
  - rush_hour.png

COMPILATION INSTRUCTIONS:
1. Run: pdflatex main.tex
2. Run: bibtex main
3. Run: pdflatex main.tex
4. Run: pdflatex main.tex

REQUIRED PACKAGES:
- amsmath, amssymb, amsfonts
- graphicx
- cite
- booktabs
- siunitx
- microtype
- url
- subfig
- stfloats
- balance
- hyperref
- tikz

The manuscript is formatted for IEEE conference proceedings and includes
all embedded figures and tables as required for submission.
