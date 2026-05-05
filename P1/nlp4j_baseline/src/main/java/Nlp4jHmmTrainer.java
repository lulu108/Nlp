import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class Nlp4jHmmTrainer {

    private static final double DEFAULT_ALPHA = 1.0;

    public static void main(String[] args) throws Exception {
        String trainPath = args.length > 0 ? args[0] : "../datasets/auto/train.txt";
        String modelPath = args.length > 1 ? args[1] : "model/hmm_bmes_model.json";
        double alpha = args.length > 2 ? Double.parseDouble(args[2]) : DEFAULT_ALPHA;

        List<BmesUtils.Sample> samples = BmesUtils.readDataset(Paths.get(trainPath));
        if (samples.isEmpty()) {
            System.out.println("No training data found at: " + trainPath);
            return;
        }

        HmmModel model = train(samples, alpha);
        model.trainSize = samples.size();
        writeModel(model, Paths.get(modelPath));

        System.out.println("HMM training finished.");
        System.out.println("Train samples: " + samples.size());
        System.out.println("Vocab size: " + model.vocabSize);
        System.out.println("Model saved to: " + modelPath);
    }

    private static HmmModel train(List<BmesUtils.Sample> samples, double alpha) {
        int stateSize = BmesUtils.STATES.size();
        Map<String, Integer> stateIndex = new HashMap<>();
        for (int i = 0; i < BmesUtils.STATES.size(); i++) {
            stateIndex.put(BmesUtils.STATES.get(i), i);
        }

        double[] piCounts = new double[stateSize];
        double[][] transCounts = new double[stateSize][stateSize];
        Map<String, Map<String, Double>> emitCounts = new HashMap<>();
        double[] stateTokenCounts = new double[stateSize];

        Set<String> vocab = new HashSet<>();

        for (BmesUtils.Sample sample : samples) {
            List<String> tags = sample.tags;
            List<String> chars = sample.chars;
            if (tags.isEmpty()) {
                continue;
            }
            int firstState = stateIndex.get(tags.get(0));
            piCounts[firstState] += 1.0;

            for (int i = 0; i < tags.size(); i++) {
                String tag = tags.get(i);
                String ch = chars.get(i);
                int state = stateIndex.get(tag);
                stateTokenCounts[state] += 1.0;
                vocab.add(ch);

                emitCounts.computeIfAbsent(tag, k -> new HashMap<>());
                Map<String, Double> map = emitCounts.get(tag);
                map.put(ch, map.getOrDefault(ch, 0.0) + 1.0);

                if (i < tags.size() - 1) {
                    int nextState = stateIndex.get(tags.get(i + 1));
                    transCounts[state][nextState] += 1.0;
                }
            }
        }

        int vocabSize = vocab.size();
        HmmModel model = new HmmModel();
        model.alpha = alpha;
        model.vocabSize = vocabSize;

        model.pi = new HashMap<>();
        model.trans = new HashMap<>();
        model.emit = new HashMap<>();
        model.unkEmission = new HashMap<>();

        double totalPi = 0.0;
        for (double v : piCounts) {
            totalPi += v;
        }

        for (int i = 0; i < stateSize; i++) {
            double prob = (piCounts[i] + alpha) / (totalPi + alpha * stateSize);
            model.pi.put(BmesUtils.STATES.get(i), Math.log(prob));
        }

        for (int i = 0; i < stateSize; i++) {
            double rowTotal = 0.0;
            for (int j = 0; j < stateSize; j++) {
                rowTotal += transCounts[i][j];
            }
            Map<String, Double> row = new HashMap<>();
            for (int j = 0; j < stateSize; j++) {
                double prob = (transCounts[i][j] + alpha) / (rowTotal + alpha * stateSize);
                row.put(BmesUtils.STATES.get(j), Math.log(prob));
            }
            model.trans.put(BmesUtils.STATES.get(i), row);
        }

        for (int i = 0; i < stateSize; i++) {
            String state = BmesUtils.STATES.get(i);
            Map<String, Double> counts = emitCounts.getOrDefault(state, Map.of());
            Map<String, Double> probs = new HashMap<>();
            for (Map.Entry<String, Double> entry : counts.entrySet()) {
                double prob = (entry.getValue() + alpha) / (stateTokenCounts[i] + alpha * vocabSize);
                probs.put(entry.getKey(), Math.log(prob));
            }
            double unkProb = alpha / (stateTokenCounts[i] + alpha * vocabSize);
            model.emit.put(state, probs);
            model.unkEmission.put(state, Math.log(unkProb));
        }

        return model;
    }

    private static void writeModel(HmmModel model, Path path) throws IOException {
        path.getParent().toFile().mkdirs();
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            sb.append("\"states\":[");
            for (int i = 0; i < BmesUtils.STATES.size(); i++) {
                if (i > 0) {
                    sb.append(",");
                }
                sb.append("\"").append(BmesUtils.STATES.get(i)).append("\"");
            }
            sb.append("],");
            sb.append("\"alpha\":").append(model.alpha).append(",");
            sb.append("\"vocabSize\":").append(model.vocabSize).append(",");
            sb.append("\"trainSize\":").append(model.trainSize).append(",");
            sb.append("\"pi\":").append(writeMap(model.pi)).append(",");
            sb.append("\"trans\":").append(writeNestedMap(model.trans)).append(",");
            sb.append("\"emit\":").append(writeNestedMap(model.emit)).append(",");
            sb.append("\"unkEmission\":").append(writeMap(model.unkEmission));
            sb.append("}");
            writer.write(sb.toString());
        }
    }

    private static String writeMap(Map<String, Double> map) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        Map<String, Double> sorted = new TreeMap<>(map);
        int idx = 0;
        for (Map.Entry<String, Double> entry : sorted.entrySet()) {
            if (idx++ > 0) {
                sb.append(",");
            }
            sb.append("\"").append(escapeJson(entry.getKey())).append("\":");
            sb.append(entry.getValue());
        }
        sb.append("}");
        return sb.toString();
    }

    private static String writeNestedMap(Map<String, Map<String, Double>> map) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        Map<String, Map<String, Double>> sortedOuter = new TreeMap<>(map);
        int idx = 0;
        for (Map.Entry<String, Map<String, Double>> entry : sortedOuter.entrySet()) {
            if (idx++ > 0) {
                sb.append(",");
            }
            sb.append("\"").append(escapeJson(entry.getKey())).append("\":");
            sb.append(writeMap(entry.getValue()));
        }
        sb.append("}");
        return sb.toString();
    }

    private static String escapeJson(String text) {
        if (text == null) {
            return "";
        }
        return text.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static final class HmmModel {
        double alpha;
        int vocabSize;
        int trainSize;
        Map<String, Double> pi;
        Map<String, Map<String, Double>> trans;
        Map<String, Map<String, Double>> emit;
        Map<String, Double> unkEmission;
    }
}
