# P4 NLP Application Project Instructions

## 1. Project Goal

This project is for Experiment 4: Natural Language Processing Application Design.

The system must provide a web-based interactive NLP application with the following features:

- Chinese word segmentation
- Named entity recognition
- Text classification
- Text clustering visualization
- Basic system testing
- Result analysis support for the experiment report

The final project should be suitable for classroom demonstration, screenshots, and experiment report writing.

---

## 2. Core Requirements

The implementation should align with these course experiment requirements:

1. The system must be presented on a web page.
2. The user must be able to interact with the system.
3. Inputting one piece of text should return:
   - tokenization result
   - named entity recognition result
   - classification result
4. The system must visualize clustering results for multiple documents.
5. The project should support experiment process description and report writing.

---

## 3. Development Principles

- Prioritize a complete and runnable end-to-end system.
- Prefer simple, stable, and readable implementations over unnecessary complexity.
- Keep frontend, backend, and algorithm modules separated.
- Use unified API contracts and JSON response formats.
- Make the system easy to demonstrate and easy to explain in the report.
- Do not overengineer unless required.

---

## 4. Recommended Tech Stack

Use the following stack unless there is a strong reason to change:

- Backend: Python + Flask
- Frontend: Vue 3
- NLP algorithms: Python libraries and lightweight ML/NLP models
- Visualization: ECharts

If a stack choice is already made in the repository, follow the existing stack and do not replace it casually.

### 4.1 Runtime Versions

- Python: 3.10+
- Node.js: 18+

### 4.2 Baseline Algorithm Preference

- tokenization should use `jieba`
- text classification and clustering should prefer `scikit-learn`
- ner should prefer lightweight Chinese NER solutions

### 4.3 Minimal Run Commands

```bash
pip install -r requirements.txt
python backend/app.py
cd frontend && npm install && npm run dev
pytest
```

---

## 5. Project Structure Rules

Keep the repository organized as follows:

- `frontend/` for frontend UI code only
- `backend/` for API and service code only
- `algorithms/` for NLP algorithm logic only
- `models/` for trained model files or cached artifacts
- `data/` for demo data, training data, and example documents
- `tests/` for test scripts
- `docs/` for API specs and design documents

If folder structure changes, update the README directory tree in the same change.

Do not mix frontend code into backend folders.
Do not place heavy algorithm logic directly inside route files.

---

## 6. API Design Rules

The backend should provide the following endpoints:

- `POST /api/tokenize`
- `POST /api/ner`
- `POST /api/classify`
- `POST /api/cluster`

Input and output must use JSON.
All response field names should remain stable.
If response formats are changed, update both frontend and backend consistently.

---

## 7. Standard JSON Response Formats

### 7.1 Tokenization

Request:

```json
{
  "text": "今天天气很好，我想去北京旅游。"
}
```

Response:

```json
{
  "success": true,
  "tokens": [
    "今天",
    "天气",
    "很",
    "好",
    "，",
    "我",
    "想",
    "去",
    "北京",
    "旅游",
    "。"
  ]
}
```

### 7.2 Named Entity Recognition

Request:

```json
{
  "text": "今天天气很好，我想去北京旅游。"
}
```

Response:

```json
{
  "success": true,
  "entities": [
    {
      "text": "北京",
      "label": "LOC",
      "start": 9,
      "end": 11
    }
  ]
}
```

### 7.3 Text Classification

Request:

```json
{
  "text": "北京有很多著名景点，适合旅游观光。"
}
```

Response:

```json
{
  "success": true,
  "label": "旅游",
  "confidence": 0.91
}
```

### 7.4 Text Clustering

Request:

```json
{
  "documents": [
    { "title": "文本1", "text": "苹果公司发布了新产品。" },
    { "title": "文本2", "text": "人工智能推动了科技发展。" },
    { "title": "文本3", "text": "某球队赢得了比赛冠军。" }
  ]
}
```

Response:

```json
{
  "success": true,
  "points": [
    { "title": "文本1", "x": 1.25, "y": -0.73, "cluster": 0 },
    { "title": "文本2", "x": 0.88, "y": -0.11, "cluster": 0 },
    { "title": "文本3", "x": -1.02, "y": 1.44, "cluster": 1 }
  ]
}
```

---

## 8. Backend Design Rules

- Route files should remain thin.
- Business logic should be placed in service files.
- NLP implementation logic should be placed in `algorithms/`.
- Reuse preprocessing functions when possible.
- Add input validation and exception handling.
- Always return structured JSON instead of raw Python objects.

Recommended flow:

1. route receives request
2. validate input
3. call service layer
4. service calls algorithm layer
5. format JSON response
6. return response

## 9. Frontend Design Rules

The frontend should be simple, clean, and suitable for experiment demonstration.

The UI should contain:

- a text input area for single-text NLP tasks
- a result display area for tokenization / NER / classification
- a separate area or page for clustering visualization
- loading states
- clear error messages
- readable labels and section titles

Avoid overly complex animation or decorative design.
Prioritize clarity, usability, and screenshot-friendliness.

## 10. Testing Rules

At minimum, the project should include:

- basic backend API tests
- basic tests for tokenization, NER, classification, and clustering outputs
- at least one runnable demo path for full system testing

The system should be testable for:

- single text processing
- multi-document clustering visualization
- frontend-backend interaction
- error handling for invalid input

## 11. Development Workflow

When implementing features, prefer the following order:

1. define or update API contract
2. implement backend route
3. implement service layer
4. connect algorithm module
5. connect frontend UI
6. add or update tests
7. verify demo readiness

Prefer incremental, working changes over large untested rewrites.

## 12. Constraints

- Do not rebuild the whole project structure without strong reason.
- Do not introduce overly heavy frameworks unless necessary.
- Do not optimize prematurely before the complete demo works.
- Do not duplicate logic across multiple files if it can be shared.
- Do not silently change API field names.
- Do not introduce heavy NLP frameworks when the baseline lightweight approach is sufficient.

## 13. Deliverable Awareness

The final project should help support:

- in-class demonstration
- experiment screenshots
- experiment process description
- result analysis
- report writing

Every major feature should be implemented in a way that is easy to demonstrate and explain.
