PAPER = sqisign-selkie
SPECREVIEW = sqisign-v3-spec-review
SECURITYREVIEW = sqisign-v3-security-review

.PHONY: all paper spec-review security-review watch watch-spec-review clean

all: paper spec-review security-review

paper: $(PAPER).pdf

spec-review: $(SPECREVIEW).pdf

security-review: $(SECURITYREVIEW).pdf

watch:
	latexmk -pdf -pvc -view=none -interaction=nonstopmode $(SECURITYREVIEW).tex

watch-spec-review:
	latexmk -pdf -pvc -view=none -interaction=nonstopmode $(SPECREVIEW).tex

$(PAPER).pdf: $(PAPER).tex prelude.tex local.bib sections/*.tex
	latexmk -pdf -interaction=nonstopmode $(PAPER).tex

$(SPECREVIEW).pdf: $(SPECREVIEW).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(SPECREVIEW).tex

$(SECURITYREVIEW).pdf: $(SECURITYREVIEW).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(SECURITYREVIEW).tex

clean:
	latexmk -C $(PAPER)
	latexmk -C $(SPECREVIEW)
	latexmk -C $(SECURITYREVIEW)
