# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a static HTML website for the "Learning Self-Correction in Vision–Language Models via Rollout Augmentation" research paper. It is a single-page academic project page with interactive demonstrations.

- **Paper**: TBD
- **Code**: TBD
- **Models**: TBD

## Development Commands

This is a static website with no build system. To preview changes:

```bash
# Python HTTP server (recommended)
python -m http.server 8000

# Or open directly in browser
open index.html
```

## Architecture

### File Structure

- **index.html** - Main single-page website containing all content, styles, and interactive case study logic
- **static/css/** - Bulma CSS framework, Font Awesome, and custom styles
- **static/js/** - Interactive components including the case study viewer with typing animation
- **static/images/** - Paper figures, case study images, and logos

### Key Interactive Component: Case Study Viewer

The page features an interactive self-correction demonstration (lines 504-526 in index.html):

- **Data structure**: `questions` object (lines 661-755) contains 4 case studies with images, problem descriptions, initial model answers, and correction trajectories
- **Typing animation**: `typeWriter()` function (lines 640-658) renders model responses with a typewriter effect
- **User flow**: Users select a case → view initial (often incorrect) answer → click "Self-Correct" to see up to 3 correction attempts

### External Dependencies

Loaded via CDN in index.html:
- Bulma CSS framework (UI components)
- KaTeX (math rendering for paper equations)
- Font Awesome (icons)
- jQuery (DOM manipulation)

## Making Changes

- **Content updates**: Edit index.html directly - the page is self-contained
- **Adding case studies**: Extend the `questions` JavaScript object with new entries following the existing schema (image, title, desc, initial, corrections array)
- **Styling**: Modify inline styles in the `<style>` block (lines 56-266) or static/css/index.css
- **Images**: Add to static/images/ and reference with relative paths

## Notes

- The website uses client-side JavaScript for the interactive case study viewer - no backend required
- Math equations are rendered using KaTeX with `$$` delimiters for display math
- The page is designed as a single-file deployment for easy hosting on GitHub Pages or similar static hosts
