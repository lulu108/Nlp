export const CLUSTER_PALETTE = [
  "#2b63f0",
  "#16a34a",
  "#f59e0b",
  "#ef4444",
  "#0ea5e9",
  "#8b5cf6",
];

export function getClusterColor(cluster, fallbackIndex = 0) {
  const clusterIndex = Number.isFinite(Number(cluster)) ? Number(cluster) : fallbackIndex;
  const safeIndex = Math.abs(clusterIndex) % CLUSTER_PALETTE.length;
  return CLUSTER_PALETTE[safeIndex];
}
