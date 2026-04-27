---
name: p4-frontend-design-system
description: Improve and constrain the Vue 3 frontend of the P4 NLP project for visual polish, component decomposition, result presentation, and report or screenshot readiness. Use when refining page layout, styling, result cards, clustering views, shared design tokens, or any demo-facing frontend output in `frontend/src`.
---

# P4 Frontend Design System

Build frontend updates that are clean, structured, and easy to present in class.

## Goal

Make the Vue 3 interface look polished enough for demonstration while staying simple enough for an experiment report.

## Apply This Structure

- Keep page-level orchestration in `frontend/src/pages/`
- Keep reusable visual blocks in `frontend/src/components/`
- Keep API calls in `frontend/src/api/`
- Keep shared tokens, spacing, and global layout styles in `frontend/src/styles/main.css`
- Let `App.vue` handle shell-level navigation or top-level layout only

## Split Components Deliberately

- Let pages own request flow, loading state, error state, and high-level composition
- Let each component focus on one visual responsibility such as token results, NER results, classification summary, or cluster chart
- Extract repeated panels, cards, legends, or status blocks into reusable components instead of duplicating markup
- Avoid putting large inline style blocks and heavy business logic into page files when a component can isolate them

## Improve Visual Hierarchy

- Prefer a calm academic-demo style over flashy decoration
- Define a small set of CSS variables for color, spacing, radius, border, and shadow before expanding the page design
- Use one clear accent color plus neutral backgrounds with strong text contrast
- Separate input, actions, and results into clearly titled panels
- Make important results scannable within a few seconds during a demo
- Preserve enough whitespace for screenshots, but avoid empty oversized gaps

## Present Results Clearly

- Show tokenization results as readable chips, tags, or segmented blocks with consistent spacing
- Show NER results with obvious label differentiation and a readable empty state when no entities are found
- Show classification results with the predicted label as the primary focus and confidence as a supporting signal
- Show clustering results with both the chart and a compact textual summary when helpful for interpretation
- Keep titles, legends, captions, and labels understandable without requiring code knowledge
- Always support loading, success, empty, and error states explicitly

## Design For Report Screenshots

- Keep the first screen presentation-ready on common laptop widths
- Avoid layouts that require horizontal scrolling or deep vertical scrolling to understand the result
- Ensure charts have visible titles, legends, axes, or explanatory text when needed for screenshots
- Prefer stable card heights and predictable spacing so screenshots look intentional
- Make section headings usable as report figure captions with minimal rewriting
- Keep text large enough to remain readable after screenshots are pasted into a document

## Vue 3 Implementation Rules

- Preserve the existing Vue 3 component structure unless a clearer split is needed
- Prefer props and emitted events for component boundaries over hidden shared state
- Keep derived display formatting close to the component that renders it
- Do not hardcode mock results in UI components unless the task explicitly asks for demo placeholders
- Keep styles responsive for both desktop demo view and narrow mobile preview

## Avoid

- Avoid generic admin-dashboard styling that hides the experiment focus
- Avoid overly decorative gradients, animations, or glass effects that hurt readability
- Avoid mixing API details, presentation markup, and chart configuration into one large file
- Avoid unlabeled result blocks that make screenshots hard to explain in the report

## Output Expectations

When using this skill, produce:

- clearer page layout
- better component boundaries
- stronger result visualization
- screenshot-friendly typography and spacing
- frontend code that remains easy to explain in the experiment report
