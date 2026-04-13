# NiDS / LLNCS paper (split sections).
# From repo root: make -C Academia pdf
# From Academia/:   make pdf

TEXDIR := nids2026_article
MAIN ?= main
PDFLATEX ?= pdflatex
LATEXMK ?= latexmk
TECTONIC ?= $(CURDIR)/tectonic

.PHONY: pdf clean

pdf:
	@if command -v "$(LATEXMK)" >/dev/null 2>&1; then \
		cd "$(TEXDIR)" && "$(LATEXMK)" -pdf -interaction=nonstopmode -halt-on-error "$(MAIN).tex"; \
	elif command -v "$(PDFLATEX)" >/dev/null 2>&1; then \
		cd "$(TEXDIR)" && "$(PDFLATEX)" -interaction=nonstopmode -halt-on-error "$(MAIN).tex" && \
		"$(PDFLATEX)" -interaction=nonstopmode -halt-on-error "$(MAIN).tex"; \
	elif [ -x "$(TECTONIC)" ]; then \
		cd "$(TEXDIR)" && "$(TECTONIC)" "$(MAIN).tex"; \
	else \
		echo "No LaTeX engine: install texlive/latexmk or add tectonic at $(TECTONIC)" >&2; \
		exit 1; \
	fi

clean:
	rm -f "$(TEXDIR)/$(MAIN).pdf" "$(TEXDIR)/$(MAIN).aux" "$(TEXDIR)/$(MAIN).log" "$(TEXDIR)/$(MAIN).out" \
		"$(TEXDIR)/$(MAIN).toc" "$(TEXDIR)/$(MAIN).fls" "$(TEXDIR)/$(MAIN).fdb_latexmk" \
		"$(TEXDIR)/$(MAIN).synctex.gz" "$(TEXDIR)/$(MAIN).bbl" "$(TEXDIR)/$(MAIN).blg" \
		"$(TEXDIR)/$(MAIN).bcf" "$(TEXDIR)/$(MAIN).run.xml" "$(TEXDIR)/sections/*.aux" 2>/dev/null || true
