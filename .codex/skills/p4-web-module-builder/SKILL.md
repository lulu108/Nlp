# Skill: P4 Web Module Builder

Use this skill when building or updating frontend modules for the P4 NLP web application.

## Goal

Create clean, readable, demo-friendly frontend modules for the experiment project.

## Scope

This skill applies to:

- input panels
- result cards
- cluster visualization components
- loading and error states
- simple page layout updates

## Requirements

- Each module should be easy to understand and suitable for experiment screenshots
- Keep the UI simple and clean
- Prefer reusable components
- Do not hardcode algorithm results unless explicitly required
- API calls should be separated into frontend api files
- Show loading state, empty state, success state, and error state clearly
- Default to a two-page layout: single-text analysis page and clustering visualization page

## Component Design Rules

- Input area and result area should be visually separated
- Result sections should have clear titles
- Entity labels should be easy to distinguish
- Cluster chart should support legends or labels
- Avoid overly complex animations or decorative elements

## Output Expectations

When using this skill, generate:

- frontend component code
- API call integration
- minimal styling
- readable state handling

## Notes

- Prioritize usability over fancy design
- The UI should support demonstration in class and screenshots for the report
- Layout should support both desktop display and screenshot cropping
