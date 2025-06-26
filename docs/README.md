# SemantiCore Documentation

This directory contains the complete documentation for SemantiCore, built using Sphinx.

## 📁 Directory Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── getting_started.rst    # Getting started guide
├── examples.rst           # Comprehensive examples
├── api/                   # API documentation
│   └── index.rst         # API reference index
├── tutorials/             # Tutorial guides
├── examples/              # Code examples
├── _static/               # Static assets
│   ├── css/              # Custom CSS
│   │   └── custom.css
│   └── js/               # Custom JavaScript
│       └── custom.js
├── _templates/            # Custom templates
├── Makefile              # Build commands
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Install documentation dependencies:**
   ```bash
   pip install -e ".[docs]"
   ```

2. **Or install manually:**
   ```bash
   pip install sphinx sphinx-rtd-theme sphinx-copybutton sphinx-tabs myst-parser
   ```

### Building Documentation

1. **Build HTML documentation:**
   ```bash
   cd docs
   make html
   ```

2. **Serve locally:**
   ```bash
   make serve
   ```

3. **Build all formats:**
   ```bash
   make all
   ```

## 📋 Available Commands

### Basic Commands

- `make html` - Build HTML documentation
- `make pdf` - Build PDF documentation
- `make epub` - Build EPUB documentation
- `make clean` - Clean build directory

### Quality Checks

- `make linkcheck` - Check for broken links
- `make doctest` - Run doctests
- `make spelling` - Spell check documentation

### Development

- `make serve` - Serve documentation locally
- `make dev` - Build and serve (development)
- `make full` - Full build with all checks

### Advanced

- `make install-deps` - Install documentation dependencies
- `make api` - Generate API documentation
- `make update` - Update all documentation
- `make deploy` - Deploy to GitHub Pages

## 🎨 Customization

### CSS Customization

Edit `_static/css/custom.css` to customize the appearance:

```css
:root {
    --semanticore-primary: #2980B9;
    --semanticore-secondary: #27AE60;
    --semanticore-accent: #8E44AD;
}
```

### JavaScript Customization

Edit `_static/js/custom.js` to add interactive features:

```javascript
// Add custom functionality
document.addEventListener('DOMContentLoaded', function() {
    // Your custom code here
});
```

### Theme Configuration

Modify `conf.py` to change theme options:

```python
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
}
```

## 📝 Writing Documentation

### RST Files

Use reStructuredText (RST) for documentation:

```rst
Title
=====

Section
--------

Subsection
~~~~~~~~~~

.. code-block:: python

    def example():
        return "Hello, World!"

.. note::

    This is a note.

.. warning::

    This is a warning.
```

### Markdown Files

Use MyST Markdown for simpler syntax:

```markdown
# Title

## Section

### Subsection

```python
def example():
    return "Hello, World!"
```

::: note
This is a note.
:::

::: warning
This is a warning.
:::
```

### Code Examples

Include code examples with syntax highlighting:

```rst
.. code-block:: python

    from semanticore import SemantiCore
    
    core = SemantiCore()
    result = core.process_document("document.pdf")
```

### API Documentation

Use autodoc for automatic API documentation:

```rst
.. automodule:: semanticore.core.engine
   :members:
   :undoc-members:
   :show-inheritance:
```

## 🔧 Configuration

### Sphinx Configuration

Key settings in `conf.py`:

- **Extensions**: List of Sphinx extensions
- **Theme**: Read the Docs theme
- **Static files**: CSS and JavaScript
- **Intersphinx**: Links to other documentation

### Build Configuration

Environment variables:

```bash
export SPHINXOPTS="-W --keep-going"
export SPHINXBUILD=sphinx-build
```

## 🚀 Deployment

### GitHub Pages

1. **Automatic deployment** (via GitHub Actions):
   - Push to `main` branch
   - Documentation builds automatically
   - Deployed to `gh-pages` branch

2. **Manual deployment**:
   ```bash
   make deploy
   ```

### Read the Docs

1. Connect repository to Read the Docs
2. Documentation builds automatically
3. Available at `https://semanticore.readthedocs.io`

## 🧪 Testing

### Link Checking

```bash
make linkcheck
```

### Spell Checking

```bash
make spelling
```

### Doctests

```bash
make doctest
```

### Full Test Suite

```bash
make full
```

## 📊 Analytics

The documentation includes Google Analytics 4 tracking:

- Page views
- User engagement
- Performance metrics

Configure in `_static/js/custom.js`:

```javascript
gtag('config', 'G-XXXXXXXXXX'); // Replace with actual GA4 ID
```

## 🤝 Contributing

### Adding New Documentation

1. Create new RST or MD file
2. Add to appropriate toctree
3. Follow style guidelines
4. Test locally before submitting

### Style Guidelines

- Use clear, concise language
- Include code examples
- Add appropriate warnings/notes
- Test all links
- Spell check content

### Review Process

1. Build documentation locally
2. Check for broken links
3. Verify code examples work
4. Submit pull request
5. Automated checks run
6. Manual review by maintainers

## 🐛 Troubleshooting

### Common Issues

**Build fails with import errors:**
```bash
pip install -e ".[docs]"
```

**Missing dependencies:**
```bash
make install-deps
```

**Broken links:**
```bash
make linkcheck
```

**Spelling errors:**
```bash
make spelling
```

### Performance Issues

- Use `make clean` before rebuilding
- Check for large images
- Optimize CSS/JS files
- Use appropriate image formats

## 📚 Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [MyST Markdown](https://myst-parser.readthedocs.io/)
- [reStructuredText](https://docutils.sourceforge.io/rst.html)

## 📞 Support

- **Documentation Issues**: GitHub Issues
- **Questions**: GitHub Discussions
- **Community**: Discord Server
- **Email**: docs@semanticore.io

---

*This documentation is built with ❤️ by the SemantiCore community.* 