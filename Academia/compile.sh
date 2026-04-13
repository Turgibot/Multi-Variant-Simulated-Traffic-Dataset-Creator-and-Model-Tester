#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}/nids2026_article"
MAIN="${1:-main}"
TECTONIC="${ROOT}/tectonic"
if command -v latexmk >/dev/null 2>&1; then
	latexmk -pdf -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
elif command -v pdflatex >/dev/null 2>&1; then
	pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
	pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
elif [[ -x "${TECTONIC}" ]]; then
	"${TECTONIC}" "${MAIN}.tex"
else
	echo "No LaTeX engine found. Install texlive, latexmk, or place tectonic at:" >&2
	echo "  ${TECTONIC}" >&2
	exit 1
fi
echo "Wrote ${MAIN}.pdf (in nids2026_article/)"
