#!/bin/sh
pandoc -s  --toc --toc-depth=1 -c pandoc.css pacemaker.md -o pacemaker.html -N
pandoc -s  --toc --toc-depth=1 -c pandoc.css pacemaker.md -o pacemaker.pdf -N
