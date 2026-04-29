#!/usr/bin/env bash
set -euo pipefail
# Same engine resolution as Academia/compile.sh (NiDS): latexmk → pdflatex → bundled tectonic.
ROOT="$(cd "$(dirname "$0")" && pwd)"
ACADEMIA="$(cd "${ROOT}/.." && pwd)"
TECTONIC="${ACADEMIA}/tectonic"
cd "${ROOT}/latex"
MAIN="main"
if command -v latexmk >/dev/null 2>&1; then
	latexmk -pdf -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
elif command -v pdflatex >/dev/null 2>&1; then
	pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
	bibtex "${MAIN}" || true
	pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
	pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
elif [[ -x "${TECTONIC}" ]]; then
	"${TECTONIC}" "${MAIN}.tex"
else
	echo "No LaTeX engine found. Install texlive, latexmk, or place tectonic at:" >&2
	echo "  ${TECTONIC}" >&2
	exit 1
fi
echo "Wrote ${MAIN}.pdf in $(pwd)"
if command -v pdfinfo >/dev/null 2>&1; then
	pdfinfo "${MAIN}.pdf" | head -n 3
fi
