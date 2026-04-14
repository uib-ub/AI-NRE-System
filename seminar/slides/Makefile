#!/usr/bin/make -f
.PHONY: default project cont project-cont clean

BUILD_DIR := /tmp/ai-ner-beamer-build

default:
	mkdir -p $(BUILD_DIR)
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD_DIR) main.tex
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD_DIR) main.tex
	cp $(BUILD_DIR)/main.pdf main.pdf
	open -a Preview main.pdf
project:
	mkdir -p $(BUILD_DIR)
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD_DIR) ai-ner-system-project.tex
	pdflatex -interaction=nonstopmode -output-directory=$(BUILD_DIR) ai-ner-system-project.tex
	cp $(BUILD_DIR)/ai-ner-system-project.pdf ai-ner-system-project.pdf
	open -a Preview ai-ner-system-project.pdf
cont:
	./watch-latex.sh main.tex
project-cont:
	./watch-latex.sh ai-ner-system-project.tex
clean:
	@latexmk -C
	@rm -f *.brf
	@rm -f *.lol
	@rm -f *.out
	@rm -f *.bbl
	@rm -f *.blg
	@rm -f *.upa
	@rm -f *.toc
	@rm -f *.vrb
	@rm -f *.snm
	@rm -f *.nav
	@rm -f *.pyg
	@find . -name \*.aux -delete
