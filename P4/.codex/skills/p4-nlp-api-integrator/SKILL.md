# Skill: P4 NLP API Integrator

Use this skill when connecting backend API routes with NLP algorithm modules in the P4 project.

## Goal

Create stable, readable, and reusable backend integration for NLP functions.

## Scope

This skill applies to:

- route creation
- service layer implementation
- algorithm module integration
- response formatting
- request validation
- error handling

## Standard Architecture

Follow this flow:

1. route receives request
2. validate request input
3. call service function
4. service calls algorithm module
5. format structured JSON response
6. return response to frontend

## Rules

- Keep route files thin
- Do not place heavy algorithm logic directly inside routes
- Reuse service functions where possible
- Use consistent response field names
- Prioritize `docs/API_SPEC.md` as the source of truth and do not rename fields without spec updates
- Handle missing input and runtime errors gracefully
- Make algorithm modules replaceable

## JSON Output Rules

- tokenization should return tokens list
- ner should return entities list
- classification should return label and confidence
- clustering should return points list with coordinates and cluster ids

## Output Expectations

When using this skill, generate:

- route file
- service file
- algorithm integration code
- compatible JSON response format
- basic validation and error handling

## Notes

- Prioritize stable end-to-end execution
- Avoid overengineering
- Prefer simple and maintainable structures
- Prefer deterministic demo outputs where possible
