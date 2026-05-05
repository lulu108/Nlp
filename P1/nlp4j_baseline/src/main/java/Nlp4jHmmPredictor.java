import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Nlp4jHmmPredictor {

    private static final double DEFAULT_ILLEGAL_TRANSITION_PENALTY = -0.3;
    private static final double DEFAULT_START_PENALTY = -0.4;
    private static final double DEFAULT_END_PENALTY = -0.4;

    private static double illegalTransitionPenalty = DEFAULT_ILLEGAL_TRANSITION_PENALTY;
    private static double startPenalty = DEFAULT_START_PENALTY;
    private static double endPenalty = DEFAULT_END_PENALTY;

    public static void main(String[] args) throws Exception {
        String testPath = args.length > 0 ? args[0] : "../datasets/auto/test.txt";
        String modelPath = args.length > 1 ? args[1] : "model/hmm_bmes_model.json";
        String outputDir = args.length > 2 ? args[2] : "output";
        if (args.length > 3) {
            illegalTransitionPenalty = Double.parseDouble(args[3]);
        }
        if (args.length > 4) {
            startPenalty = Double.parseDouble(args[4]);
        }
        if (args.length > 5) {
            endPenalty = Double.parseDouble(args[5]);
        }

        HmmModel model = loadModel(Paths.get(modelPath));
        List<BmesUtils.Sample> samples = BmesUtils.readDataset(Paths.get(testPath));
        if (samples.isEmpty()) {
            System.out.println("No test data found at: " + testPath);
            return;
        }

        List<List<String>> predictions = new ArrayList<>();
        int[][] confusion = new int[BmesUtils.STATES.size()][BmesUtils.STATES.size()];
        int total = 0;
        int correct = 0;

        for (BmesUtils.Sample sample : samples) {
            List<String> pred = viterbi(model, sample.chars);
            predictions.add(pred);
            for (int i = 0; i < sample.tags.size(); i++) {
                String gold = sample.tags.get(i);
                String guess = pred.get(i);
                int g = BmesUtils.STATES.indexOf(gold);
                int p = BmesUtils.STATES.indexOf(guess);
                if (g >= 0 && p >= 0) {
                    confusion[g][p] += 1;
                }
                total += 1;
                if (gold.equals(guess)) {
                    correct += 1;
                }
            }
        }

        double tagAccuracy = total > 0 ? (double) correct / total : 0.0;
        BmesUtils.LabelReportResult report = BmesUtils.buildLabelReport(confusion);

        Path outDir = Paths.get(outputDir);
        outDir.toFile().mkdirs();

        Path charResult = outDir.resolve("nlp4j_hmm_char_result.tsv");
        Path tokenResult = outDir.resolve("nlp4j_hmm_token_result.tsv");
        Path confusionPath = outDir.resolve("nlp4j_hmm_confusion_matrix.tsv");
        Path reportPath = outDir.resolve("nlp4j_hmm_label_report.tsv");
        Path samplePath = outDir.resolve("nlp4j_hmm_samples.txt");
        Path metricsPath = outDir.resolve("nlp4j_hmm_metrics.json");

        writeCharResult(charResult, samples, predictions);
        writeTokenResult(tokenResult, samples, predictions);
        BmesUtils.writeConfusionTsv(confusion, confusionPath);
        BmesUtils.writeLabelReportTsv(report, reportPath);
        BmesUtils.writeSamples(samplePath, samples, predictions, 8);
        writeMetrics(metricsPath, tagAccuracy, report, model, samples.size(), modelPath,
                confusionPath, reportPath, samplePath);

        System.out.println("Evaluation finished.");
        System.out.println("Test samples: " + samples.size());
        System.out.println("Tag accuracy: " + String.format("%.6f", tagAccuracy));
        System.out.println("Metrics saved to: " + metricsPath);
    }

    private static List<String> viterbi(HmmModel model, List<String> observations) {
        int n = observations.size();
        int s = BmesUtils.STATES.size();
        double[][] dp = new double[n][s];
        int[][] back = new int[n][s];

        for (int i = 0; i < s; i++) {
            String state = BmesUtils.STATES.get(i);
            double emit = emission(model, state, observations.get(0));
            dp[0][i] = model.pi.get(state) + emit + startPenalty(state);
            back[0][i] = -1;
        }

        for (int t = 1; t < n; t++) {
            String obs = observations.get(t);
            for (int i = 0; i < s; i++) {
                String currState = BmesUtils.STATES.get(i);
                double bestScore = Double.NEGATIVE_INFINITY;
                int bestPrev = 0;
                for (int j = 0; j < s; j++) {
                    String prevState = BmesUtils.STATES.get(j);
                    double trans = model.trans.get(prevState).get(currState);
                    double score = dp[t - 1][j] + trans + transitionPenalty(prevState, currState);
                    if (score > bestScore) {
                        bestScore = score;
                        bestPrev = j;
                    }
                }
                dp[t][i] = bestScore + emission(model, currState, obs);
                back[t][i] = bestPrev;
            }
        }

        int bestLast = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < s; i++) {
            double score = dp[n - 1][i] + endPenalty(BmesUtils.STATES.get(i));
            if (score > bestScore) {
                bestScore = score;
                bestLast = i;
            }
        }

        List<String> path = new ArrayList<>();
        int idx = bestLast;
        for (int t = n - 1; t >= 0; t--) {
            path.add(0, BmesUtils.STATES.get(idx));
            idx = back[t][idx];
            if (idx < 0 && t > 0) {
                idx = 0;
            }
        }
        return path;
    }

    private static double startPenalty(String state) {
        if ("B".equals(state) || "S".equals(state)) {
            return 0.0;
        }
        return startPenalty;
    }

    private static double endPenalty(String state) {
        if ("E".equals(state) || "S".equals(state)) {
            return 0.0;
        }
        return endPenalty;
    }

    private static double transitionPenalty(String prev, String curr) {
        boolean legal = switch (prev) {
            case "B" -> "M".equals(curr) || "E".equals(curr);
            case "M" -> "M".equals(curr) || "E".equals(curr);
            case "E" -> "B".equals(curr) || "S".equals(curr);
            case "S" -> "B".equals(curr) || "S".equals(curr);
            default -> true;
        };
        return legal ? 0.0 : illegalTransitionPenalty;
    }

    private static double emission(HmmModel model, String state, String obs) {
        Map<String, Double> map = model.emit.get(state);
        if (map != null && map.containsKey(obs)) {
            return map.get(obs);
        }
        return model.unkEmission.get(state);
    }

    private static void writeCharResult(Path path, List<BmesUtils.Sample> samples,
                                        List<List<String>> predictions) throws IOException {
        path.getParent().toFile().mkdirs();
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("sentence_id\tchar\tgold_tag\tpred_tag");
            writer.newLine();
            for (int i = 0; i < samples.size(); i++) {
                BmesUtils.Sample sample = samples.get(i);
                List<String> pred = predictions.get(i);
                for (int j = 0; j < sample.chars.size(); j++) {
                    writer.write(String.format("%d\t%s\t%s\t%s",
                            i + 1, sample.chars.get(j), sample.tags.get(j), pred.get(j)));
                    writer.newLine();
                }
            }
        }
    }

    private static void writeTokenResult(Path path, List<BmesUtils.Sample> samples,
                                         List<List<String>> predictions) throws IOException {
        path.getParent().toFile().mkdirs();
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("sentence_id\tgold_tokens\tpred_tokens");
            writer.newLine();
            for (int i = 0; i < samples.size(); i++) {
                BmesUtils.Sample sample = samples.get(i);
                List<String> pred = predictions.get(i);
                String gold = String.join("/", sample.words);
                String guessed = String.join("/", BmesUtils.tagsToWords(sample.chars, pred));
                writer.write(String.format("%d\t%s\t%s", i + 1, gold, guessed));
                writer.newLine();
            }
        }
    }

    private static void writeMetrics(Path path, double tagAccuracy, BmesUtils.LabelReportResult report,
                                     HmmModel model, int testSize, String modelPath,
                                     Path confusionPath, Path reportPath, Path samplePath) throws IOException {
        path.getParent().toFile().mkdirs();
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("{");
            writer.write("\"chain\":\"nlp4j_baseline_hmm\",");
            writer.write("\"tag_accuracy\":" + String.format("%.6f", tagAccuracy) + ",");
            writer.write("\"macro_precision\":" + String.format("%.6f", report.macroPrecision) + ",");
            writer.write("\"macro_recall\":" + String.format("%.6f", report.macroRecall) + ",");
            writer.write("\"macro_f1\":" + String.format("%.6f", report.macroF1) + ",");
            writer.write("\"train_size\":" + model.trainSize + ",");
            writer.write("\"test_size\":" + testSize + ",");
            writer.write("\"model_path\":\"" + escapeJson(modelPath) + "\",");
            writer.write("\"illegal_transition_penalty\":" + String.format("%.6f", illegalTransitionPenalty) + ",");
            writer.write("\"start_penalty\":" + String.format("%.6f", startPenalty) + ",");
            writer.write("\"end_penalty\":" + String.format("%.6f", endPenalty) + ",");
            writer.write("\"confusion_matrix_path\":\"" + escapeJson(confusionPath.toString()) + "\",");
            writer.write("\"label_report_path\":\"" + escapeJson(reportPath.toString()) + "\",");
            writer.write("\"samples_path\":\"" + escapeJson(samplePath.toString()) + "\"");
            writer.write("}");
        }
    }

    private static String escapeJson(String text) {
        if (text == null) {
            return "";
        }
        return text.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static HmmModel loadModel(Path path) throws IOException {
        String json = Files.readString(path, StandardCharsets.UTF_8);
        JsonReader reader = new JsonReader(json);
        Object value = reader.readValue();
        if (!(value instanceof Map)) {
            throw new IllegalStateException("Invalid model json");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> root = (Map<String, Object>) value;
        HmmModel model = new HmmModel();
        model.alpha = toDouble(root.get("alpha"));
        model.vocabSize = (int) Math.round(toDouble(root.get("vocabSize")));
        model.trainSize = (int) Math.round(toDouble(root.get("trainSize")));
        model.pi = toDoubleMap(castMap(root.get("pi")));
        model.trans = toNestedDoubleMap(castMap(root.get("trans")));
        model.emit = toNestedDoubleMap(castMap(root.get("emit")));
        model.unkEmission = toDoubleMap(castMap(root.get("unkEmission")));
        return model;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Object value) {
        if (value instanceof Map) {
            return (Map<String, Object>) value;
        }
        return new HashMap<>();
    }

    private static Map<String, Double> toDoubleMap(Map<String, Object> map) {
        Map<String, Double> result = new HashMap<>();
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            result.put(entry.getKey(), toDouble(entry.getValue()));
        }
        return result;
    }

    private static Map<String, Map<String, Double>> toNestedDoubleMap(Map<String, Object> map) {
        Map<String, Map<String, Double>> result = new HashMap<>();
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            result.put(entry.getKey(), toDoubleMap(castMap(entry.getValue())));
        }
        return result;
    }

    private static double toDouble(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        if (value instanceof String) {
            return Double.parseDouble((String) value);
        }
        return 0.0;
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

    private static final class JsonReader {
        private final String text;
        private int index;

        JsonReader(String text) {
            this.text = text;
            this.index = 0;
        }

        Object readValue() {
            skipWhitespace();
            if (index >= text.length()) {
                return null;
            }
            char ch = text.charAt(index);
            if (ch == '{') {
                return readObject();
            }
            if (ch == '[') {
                return readArray();
            }
            if (ch == '"') {
                return readString();
            }
            if (ch == 't' || ch == 'f') {
                return readBoolean();
            }
            if (ch == 'n') {
                return readNull();
            }
            return readNumber();
        }

        private Map<String, Object> readObject() {
            Map<String, Object> map = new HashMap<>();
            expect('{');
            skipWhitespace();
            if (peek() == '}') {
                index++;
                return map;
            }
            while (true) {
                skipWhitespace();
                String key = readString();
                skipWhitespace();
                expect(':');
                Object value = readValue();
                map.put(key, value);
                skipWhitespace();
                if (peek() == ',') {
                    index++;
                    continue;
                }
                if (peek() == '}') {
                    index++;
                    break;
                }
            }
            return map;
        }

        private List<Object> readArray() {
            List<Object> list = new ArrayList<>();
            expect('[');
            skipWhitespace();
            if (peek() == ']') {
                index++;
                return list;
            }
            while (true) {
                Object value = readValue();
                list.add(value);
                skipWhitespace();
                if (peek() == ',') {
                    index++;
                    continue;
                }
                if (peek() == ']') {
                    index++;
                    break;
                }
            }
            return list;
        }

        private String readString() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (index < text.length()) {
                char ch = text.charAt(index++);
                if (ch == '"') {
                    break;
                }
                if (ch == '\\') {
                    char next = text.charAt(index++);
                    switch (next) {
                        case '"':
                            sb.append('"');
                            break;
                        case '\\':
                            sb.append('\\');
                            break;
                        case '/':
                            sb.append('/');
                            break;
                        case 'b':
                            sb.append('\b');
                            break;
                        case 'f':
                            sb.append('\f');
                            break;
                        case 'n':
                            sb.append('\n');
                            break;
                        case 'r':
                            sb.append('\r');
                            break;
                        case 't':
                            sb.append('\t');
                            break;
                        case 'u':
                            String hex = text.substring(index, index + 4);
                            sb.append((char) Integer.parseInt(hex, 16));
                            index += 4;
                            break;
                        default:
                            sb.append(next);
                            break;
                    }
                } else {
                    sb.append(ch);
                }
            }
            return sb.toString();
        }

        private Boolean readBoolean() {
            if (text.startsWith("true", index)) {
                index += 4;
                return Boolean.TRUE;
            }
            index += 5;
            return Boolean.FALSE;
        }

        private Object readNull() {
            index += 4;
            return null;
        }

        private Number readNumber() {
            int start = index;
            while (index < text.length()) {
                char ch = text.charAt(index);
                if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
                    index++;
                } else {
                    break;
                }
            }
            String num = text.substring(start, index);
            if (num.contains(".") || num.contains("e") || num.contains("E")) {
                return Double.parseDouble(num);
            }
            return Long.parseLong(num);
        }

        private void skipWhitespace() {
            while (index < text.length()) {
                char ch = text.charAt(index);
                if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') {
                    index++;
                } else {
                    break;
                }
            }
        }

        private char peek() {
            if (index >= text.length()) {
                return '\0';
            }
            return text.charAt(index);
        }

        private void expect(char ch) {
            if (text.charAt(index) != ch) {
                throw new IllegalStateException("Expected " + ch + " at position " + index);
            }
            index++;
        }
    }
}
