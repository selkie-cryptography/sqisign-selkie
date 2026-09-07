PAPER = sqisign-selkie
SPECREVIEW = sqisign-v2-spec-review
SECURITYREVIEW = sqisign-v3-security-review

.PHONY: all paper spec-review security-review clean

all: paper spec-review

paper: $(PAPER).pdf

spec-review: $(SPECREVIEW).pdf

security-review: $(SECURITYREVIEW).pdf

$(PAPER).pdf: $(PAPER).tex prelude.tex local.bib sections/*.tex
	latexmk -pdf -interaction=nonstopmode $(PAPER).tex

$(SPECREVIEW).pdf: $(SPECREVIEW).tex
	latexmk -pdf -interaction=nonstopmode $(SPECREVIEW).tex

$(SECURITYREVIEW).pdf: $(SECURITYREVIEW).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(SECURITYREVIEW).tex

clean:
	latexmk -C $(PAPER)
	latexmk -C $(SPECREVIEW)
	latexmk -C $(SECURITYREVIEW)
