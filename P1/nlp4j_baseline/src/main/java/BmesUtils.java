import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class BmesUtils {

    public static final List<String> STATES = List.of("B", "M", "E", "S");

    private BmesUtils() {}

    public static final class Sample {
        public final String sentence;
        public final List<String> chars;
        public final List<String> tags;
        public final List<String> words;

        public Sample(String sentence, List<String> chars, List<String> tags, List<String> words) {
            this.sentence = sentence;
            this.chars = chars;
            this.tags = tags;
            this.words = words;
        }
    }

    public static final class LabelMetrics {
        public final String label;
        public final double precision;
        public final double recall;
        public final double f1;
        public final int support;

        public LabelMetrics(String label, double precision, double recall, double f1, int support) {
            this.label = label;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.support = support;
        }
    }

    public static final class LabelReportResult {
        public final List<LabelMetrics> rows;
        public final double macroPrecision;
        public final double macroRecall;
        public final double macroF1;

        public LabelReportResult(List<LabelMetrics> rows, double macroPrecision, double macroRecall, double macroF1) {
            this.rows = rows;
            this.macroPrecision = macroPrecision;
            this.macroRecall = macroRecall;
            this.macroF1 = macroF1;
        }
    }

    public static List<Sample> readDataset(Path path) throws IOException {
        if (!Files.exists(path)) {
            return Collections.emptyList();
        }
        List<Sample> samples = new ArrayList<>();
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        for (String line : lines) {
            if (line == null || line.isBlank()) {
                continue;
            }
            String[] parts = line.trim().split("\t");
            if (parts.length != 4) {
                continue;
            }
            String sentence = parts[0];
            List<String> chars = splitTokens(parts[1]);
            List<String> tags = splitTokens(parts[2]);
            List<String> words = splitTokens(parts[3]);
            if (chars.size() != tags.size()) {
                continue;
            }
            samples.add(new Sample(sentence, chars, tags, words));
        }
        return samples;
    }

    public static List<String> tagsToWords(List<String> chars, List<String> tags) {
        List<String> words = new ArrayList<>();
        StringBuilder buffer = new StringBuilder();
        for (int i = 0; i < chars.size(); i++) {
            String ch = chars.get(i);
            String tag = tags.get(i);
            switch (tag) {
                case "S":
                    if (buffer.length() > 0) {
                        words.add(buffer.toString());
                        buffer.setLength(0);
                    }
                    words.add(ch);
                    break;
                case "B":
                    if (buffer.length() > 0) {
                        words.add(buffer.toString());
                    }
                    buffer.setLength(0);
                    buffer.append(ch);
                    break;
                case "M":
                    buffer.append(ch);
                    break;
                case "E":
                    buffer.append(ch);
                    words.add(buffer.toString());
                    buffer.setLength(0);
                    break;
                default:
                    if (buffer.length() > 0) {
                        words.add(buffer.toString());
                        buffer.setLength(0);
                    }
                    words.add(ch);
                    break;
            }
        }
        if (buffer.length() > 0) {
            words.add(buffer.toString());
        }
        return words;
    }

    public static LabelReportResult buildLabelReport(int[][] confusion) {
        List<LabelMetrics> rows = new ArrayList<>();
        List<Double> ps = new ArrayList<>();
        List<Double> rs = new ArrayList<>();
        List<Double> f1s = new ArrayList<>();

        for (int i = 0; i < STATES.size(); i++) {
            String label = STATES.get(i);
            int tp = confusion[i][i];
            int fp = 0;
            int fn = 0;
            for (int r = 0; r < STATES.size(); r++) {
                fp += confusion[r][i];
            }
            for (int c = 0; c < STATES.size(); c++) {
                fn += confusion[i][c];
            }
            fp -= tp;
            fn -= tp;
            int support = 0;
            for (int c = 0; c < STATES.size(); c++) {
                support += confusion[i][c];
            }

            double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
            double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
            double f1 = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;

            rows.add(new LabelMetrics(label, precision, recall, f1, support));
            ps.add(precision);
            rs.add(recall);
            f1s.add(f1);
        }

        double macroP = average(ps);
        double macroR = average(rs);
        double macroF1 = average(f1s);
        return new LabelReportResult(rows, macroP, macroR, macroF1);
    }

    public static void writeConfusionTsv(int[][] confusion, Path path) throws IOException {
        path.getParent().toFile().mkdirs();
        List<String> lines = new ArrayList<>();
        lines.add("true\\pred\t" + String.join("\t", STATES));
        for (int i = 0; i < STATES.size(); i++) {
            StringBuilder row = new StringBuilder();
            row.append(STATES.get(i));
            for (int j = 0; j < STATES.size(); j++) {
                row.append("\t").append(confusion[i][j]);
            }
            lines.add(row.toString());
        }
        Files.write(path, lines, StandardCharsets.UTF_8);
    }

    public static void writeLabelReportTsv(LabelReportResult report, Path path) throws IOException {
        path.getParent().toFile().mkdirs();
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("label\tprecision\trecall\tf1\tsupport");
            writer.newLine();
            for (LabelMetrics row : report.rows) {
                writer.write(String.format("%s\t%.6f\t%.6f\t%.6f\t%d",
                        row.label, row.precision, row.recall, row.f1, row.support));
                writer.newLine();
            }
        }
    }

    public static void writeSamples(Path path, List<Sample> samples, List<List<String>> predTags, int limit) throws IOException {
        path.getParent().toFile().mkdirs();
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            int count = Math.min(limit, samples.size());
            for (int i = 0; i < count; i++) {
                Sample sample = samples.get(i);
                List<String> goldTags = sample.tags;
                List<String> pred = predTags.get(i);
                writer.write("Sentence: " + sample.sentence);
                writer.newLine();
                writer.write("Gold BMES: " + String.join(" ", goldTags));
                writer.newLine();
                writer.write("Pred BMES: " + String.join(" ", pred));
                writer.newLine();
                writer.write("Gold tokens: " + String.join("/", sample.words));
                writer.newLine();
                writer.write("Pred tokens: " + String.join("/", tagsToWords(sample.chars, pred)));
                writer.newLine();
                writer.newLine();
            }
        }
    }

    public static List<String> splitTokens(String text) {
        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }
        return new ArrayList<>(Arrays.asList(text.trim().split("\\s+")));
    }

    private static double average(List<Double> values) {
        if (values.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }
}
