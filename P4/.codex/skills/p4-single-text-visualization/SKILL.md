---
name: p4-single-text-visualization
description: Add or refine screenshot-friendly visualization for the P4 single text analysis page, especially token and NER result charts computed on the frontend.
---

# P4 Single Text Visualization

Use this skill when improving the single text analysis page with result charts or visual summaries.

## Goal

Make single text analysis results easier to read in class demos and experiment reports without changing backend APIs.

## Required Charts

- Show a token length distribution bar chart based on the existing frontend `tokens` result.
- Show an entity label statistics bar chart based on the existing frontend `entities` result.
- Compute all chart data in the frontend from already returned analysis results.
- Use the existing `echarts` dependency.
- Do not add another chart library or UI framework.
- Do not modify backend routes, payloads, algorithms, or response formats.

## Page Placement

- Keep request flow and high-level page composition in `frontend/src/pages/SingleTextPage.vue`.
- Put reusable chart rendering in `frontend/src/components/TextStatsCharts.vue`.
- Place the visualization module below the result overview section and before the detailed result cards.

## Visual Rules

- Follow `p4-frontend-design-system`.
- Keep the style calm, readable, and suitable for screenshots in an experiment report.
- Use clear section titles, chart titles, axes, and concise empty or loading states.
- Prefer stable card height and predictable spacing.
- Avoid decorative effects that reduce readability.

## State Rules

- When loading, show a loading state instead of stale charts.
- When there is no analysis result, show an empty state.
- When only tokens or only entities are available, render the available chart and show a small empty state for the missing one.

## Implementation Notes

- Keep the component course-project friendly: small computed properties, simple ECharts options, and readable CSS.
- Keep label names consistent with the NER display component where possible.
- Dispose ECharts instances on unmount and resize charts on window resize.
