PAPER = sqisign-selkie
SPECREVIEW = sqisign-v2-spec-review

.PHONY: all paper spec-review clean

all: paper spec-review

paper: $(PAPER).pdf

spec-review: $(SPECREVIEW).pdf

$(PAPER).pdf: $(PAPER).tex prelude.tex local.bib sections/*.tex
	latexmk -pdf -interaction=nonstopmode $(PAPER).tex

$(SPECREVIEW).pdf: $(SPECREVIEW).tex
	latexmk -pdf -interaction=nonstopmode $(SPECREVIEW).tex

clean:
	latexmk -C $(PAPER)
	latexmk -C $(SPECREVIEW)
