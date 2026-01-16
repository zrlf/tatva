# Documentation

This is a custom framework built with [Fumadocs](https://fumadocs.dev) and [Quarto](https://quarto.org).

## Requirements

- [nodejs](https://nodejs.org/)
- [pnpm](https://pnpm.io/)
- [quarto](https://quarto.org/docs/get-started/)

Use the `tatva` environment with the docs extras

```bash
uv sync --extra docs
```

## How to use

To use this framework easily, use the `doc.py` script located in this directory.

```bash
# install website dependencies
py doc.py install

# generate api docs
py doc.py generate-api

# launch dev server with hot reload
py doc.py dev

# build static site
py doc.py build
```

The content of the documentation is located in the `docs` directory. The
`meta.json` file is to organize the sidebar and metadata of the documentation.
Every file must have the proper header (frontmatter). This is a requirement of
`fumadocs`.

## Features

- write docs in `.md`, `.mdx`, `.qmd`, and `.ipynb` formats
- automatic rendering and running of code blocks with quarto
- fully customizable website in `_website` (nextjs project with tailwindcss, fumadocs, etc.)
