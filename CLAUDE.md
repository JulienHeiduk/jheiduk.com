# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hugo static site blog — a personal data science blog by Julien Heiduk. The active theme is **paper-new** (`themes/paper-new`), managed as a git submodule. A legacy `themes/paper` submodule is also present but not in use.

## Common Commands

```bash
# Start development server with live reload
hugo server

# Build the site (output to public/)
hugo

# Create a new post
hugo new posts/my-post-title.md
```

## Content

- All blog posts live in `content/posts/` as Markdown files.
- Front matter fields used in posts:
  ```yaml
  title: "..."
  date: 2025-01-01
  draft: false
  summary: "Short description shown in post listings."
  ```
- The `summary` field controls the excerpt shown on the home page list. Set `draft: false` to publish.
- Static images (PNG, SVG) go directly in `static/` and are referenced as `/filename.png` in Markdown.

## Configuration

`config.toml` controls site-wide settings:
- Theme: `paper-new`
- KaTeX math rendering is enabled globally (`math = true`)
- Syntax highlighting via highlight.js is disabled (`disableHLJS = true`)
- Social links: GitHub (`JulienHeiduk`) and LinkedIn (`julien-heiduk`)
- Menu items: Training (external), About (`/posts/introduction/`), Applications dropdown (RAG Demo, GNN Explorer — both Streamlit apps)

## Article Command

A custom slash command `/article` writes a complete, SEO-optimised blog post on a given topic:

```
/article on transformer attention mechanisms
/article <topic>
```

The agent targets 600–900 words of prose, includes at least one runnable Python example, suggests Excalidraw diagrams where useful, and fills in the front matter (title, date, summary). It also creates a companion Jupyter notebook at `notebooks/<slug>.ipynb` that runs all the article's code top-to-bottom. The command definition lives in `.claude/commands/article.md`.

When an Excalidraw diagram is suggested, export it from excalidraw.com as SVG and place it directly in `static/` so Hugo can serve it as `/<filename>.svg`.

## Deployment

Pushing to `main` triggers the GitHub Actions workflow at `.github/workflows/hugo.yml`, which builds with Hugo 0.128.0 (extended) and deploys to GitHub Pages automatically. The `public/` directory does not need to be committed manually.
