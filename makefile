all: html pdf

pdf:
	pandoc thesis\ proposal.md -s --filter pandoc-crossref --filter pandoc-citeproc -o rendered.tex && pdflatex -interaction=nonstopmode -file-line-error rendered.tex

html:
	pandoc thesis\ proposal.md --to html5 -s --filter pandoc-crossref --filter pandoc-citeproc -o rendered.html --mathjax

watch:
	reload -b -s rendered.html & make html & fswatch -o thesis\ proposal.md | xargs -n1 -I% make html
