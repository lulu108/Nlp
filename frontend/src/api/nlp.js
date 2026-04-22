const BASE_URL = "http://localhost:5000";

async function postJson(path, body) {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");

  let data = null;
  if (isJson) {
    data = await response.json();
  } else {
    const text = await response.text();
    data = text ? { message: text } : {};
  }

  if (!response.ok) {
    const message =
      data?.message || `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return data;
}

export function tokenizeText(text) {
  return postJson("/api/tokenize", { text });
}

export function recognizeEntities(text) {
  return postJson("/api/ner", { text });
}

export function classifyText(text) {
  return postJson("/api/classify", { text });
}

export function clusterDocuments(documents, clusterCount) {
  const payload = { documents };
  if (clusterCount !== undefined && clusterCount !== null) {
    payload.cluster_count = clusterCount;
  }
  return postJson("/api/cluster", payload);
}
