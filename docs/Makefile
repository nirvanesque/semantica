# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for SemantiCore documentation

# Build all documentation formats
all: html pdf epub

# Build HTML documentation
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "HTML documentation built in $(BUILDDIR)/html/"

# Build PDF documentation
pdf:
	@echo "Building PDF documentation..."
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX..."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo "PDF documentation built in $(BUILDDIR)/latex/"

# Build EPUB documentation
epub:
	@echo "Building EPUB documentation..."
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo "EPUB documentation built in $(BUILDDIR)/epub/"

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	rm -rf "$(BUILDDIR)"
	@echo "Build directory cleaned."

# Check for broken links
linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check completed."

# Run doctests
doctest:
	@echo "Running doctests..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Doctests completed."

# Spell check
spelling:
	@echo "Running spell check..."
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS) $(O)
	@echo "Spell check completed."

# Serve documentation locally
serve:
	@echo "Starting local documentation server..."
	@cd "$(BUILDDIR)/html" && python -m http.server 8000
	@echo "Documentation available at http://localhost:8000"

# Build and serve
dev: html serve

# Full documentation build with all checks
full: clean html linkcheck doctest
	@echo "Full documentation build completed."

# Install dependencies for documentation
install-deps:
	@echo "Installing documentation dependencies..."
	pip install sphinx sphinx-rtd-theme sphinx-copybutton sphinx-tabs myst-parser
	pip install sphinxcontrib-spelling doc8 sphinx-lint
	@echo "Documentation dependencies installed."

# Generate API documentation
api:
	@echo "Generating API documentation..."
	sphinx-apidoc -o api ../semanticore --force --module-first --no-toc
	@echo "API documentation generated."

# Update all documentation
update: api html
	@echo "Documentation updated."

# Deploy to GitHub Pages (requires gh-pages branch)
deploy: html
	@echo "Deploying to GitHub Pages..."
	@if [ -d "$(BUILDDIR)/html" ]; then \
		git checkout gh-pages; \
		cp -r "$(BUILDDIR)/html/"* .; \
		git add .; \
		git commit -m "Update documentation"; \
		git push origin gh-pages; \
		git checkout main; \
		echo "Documentation deployed to GitHub Pages."; \
	else \
		echo "Error: HTML documentation not found. Run 'make html' first."; \
		exit 1; \
	fi 