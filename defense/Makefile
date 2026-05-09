# Build the defense presentation.
#
# Requires a TeX Live (or MikTeX) installation with:
#   - latexmk
#   - biber
#   - the metropolis beamer theme  (texlive-latex-extra on Debian; via tlmgr install metropolis on TeXLive)
#   - algorithm2e, biblatex, siunitx, pgf/tikz, booktabs, microtype, appendixnumberbeamer

TEX = presentation
ENGINE = pdflatex
LATEXMK_OPTS = -shell-escape -interaction=nonstopmode -file-line-error

.PHONY: all clean cleanall view

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	latexmk -$(ENGINE) $(LATEXMK_OPTS) $(TEX).tex

view: $(TEX).pdf
	@start $(TEX).pdf 2>/dev/null || open $(TEX).pdf 2>/dev/null || xdg-open $(TEX).pdf

clean:
	latexmk -c $(TEX).tex

cleanall:
	latexmk -C $(TEX).tex
	rm -f $(TEX).bbl $(TEX).run.xml $(TEX).synctex.gz
