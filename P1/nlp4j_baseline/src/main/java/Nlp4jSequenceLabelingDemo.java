import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Nlp4jSequenceLabelingDemo {

    private static final String HEADER = "sentence_id\ttoken\tpos\tentity";
    private static final Path DEFAULT_INPUT_IN_SUBDIR = Paths.get("input", "sample_input.txt");
    private static final Path DEFAULT_INPUT_IN_ROOT = Paths.get("sample_input.txt");
    private static final Path OUTPUT_PATH = Paths.get("output", "nlp4j_result.tsv");
    private static final Path DEFAULT_LIB_DIR = Paths.get("lib");

    public static void main(String[] args) throws IOException {
        Path inputPath = resolveInputPath(args);
        List<String> sentences = readSentences(inputPath);
        Path libDir = resolveLibDir();
        List<Path> jarFiles = listJarFiles(libDir);

        validateRuntimePrerequisites(libDir, jarFiles);

        // TODO: After confirming the real NLP4J dependency coordinates or local jars,
        // replace this placeholder with actual NLP4J tokenizer/POS/NER pipeline calls.
        List<LabeledTokenRow> rows = runNlp4jPipeline(sentences, jarFiles);
        writeRows(rows, OUTPUT_PATH);

        System.out.println("Saved: " + OUTPUT_PATH.toAbsolutePath());
    }

    private static Path resolveInputPath(String[] args) {
        if (args.length > 0 && !args[0].isBlank()) {
            Path explicitPath = Paths.get(args[0]);
            if (Files.exists(explicitPath)) {
                return explicitPath;
            }
            throw new IllegalArgumentException("Input file not found: " + explicitPath.toAbsolutePath());
        }

        if (Files.exists(DEFAULT_INPUT_IN_SUBDIR)) {
            return DEFAULT_INPUT_IN_SUBDIR;
        }
        if (Files.exists(DEFAULT_INPUT_IN_ROOT)) {
            return DEFAULT_INPUT_IN_ROOT;
        }

        throw new IllegalStateException(
                "No input file found. Expected input/sample_input.txt or sample_input.txt."
        );
    }

    private static List<String> readSentences(Path inputPath) throws IOException {
        List<String> lines = Files.readAllLines(inputPath, StandardCharsets.UTF_8);
        return lines.stream()
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .collect(Collectors.toList());
    }

    private static Path resolveLibDir() {
        String configured = System.getProperty("nlp4j.lib.dir");
        if (configured == null || configured.isBlank()) {
            return DEFAULT_LIB_DIR;
        }
        return Paths.get(configured);
    }

    private static List<Path> listJarFiles(Path libDir) throws IOException {
        if (!Files.isDirectory(libDir)) {
            return List.of();
        }
        try (Stream<Path> pathStream = Files.list(libDir)) {
            return pathStream
                    .filter(path -> Files.isRegularFile(path))
                    .filter(path -> path.getFileName().toString().toLowerCase().endsWith(".jar"))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    private static void validateRuntimePrerequisites(Path libDir, List<Path> jarFiles) {
        if (!Files.isDirectory(libDir) || jarFiles.isEmpty()) {
            throw new IllegalStateException(
                    "No NLP4J runtime jars detected. Please place the confirmed NLP4J jars under "
                            + libDir.toAbsolutePath()
                            + " and then implement the real API adapter in runNlp4jPipeline()."
            );
        }
    }

    private static List<LabeledTokenRow> runNlp4jPipeline(List<String> sentences, List<Path> jarFiles) {
        // TODO: Confirm the actual NLP4J API surface and instantiate the tokenizer/POS/NER objects here.
        // TODO: Convert the real NLP4J output into rows with columns: sentence_id, token, pos, entity.
        throw new UnsupportedOperationException(
                "NLP4J jars were detected (" + jarFiles.size()
                        + " file(s)), but the real NLP4J API adapter is not implemented yet."
                        + " Wire the confirmed NLP4J tokenizer/POS/NER calls into runNlp4jPipeline()."
        );
    }

    private static void writeRows(List<LabeledTokenRow> rows, Path outputPath) throws IOException {
        Files.createDirectories(outputPath.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
            writer.write(HEADER);
            writer.newLine();
            for (LabeledTokenRow row : rows) {
                writer.write(row.toTsvLine());
                writer.newLine();
            }
        }
    }

    private record LabeledTokenRow(int sentenceId, String token, String pos, String entity) {
        private LabeledTokenRow {
            if (sentenceId <= 0) {
                throw new IllegalArgumentException("sentenceId must be positive.");
            }
            requireNonBlank(token, "token");
            requireNonBlank(pos, "pos");
            requireAllowedEntity(entity);
        }

        private static void requireNonBlank(String value, String fieldName) {
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(fieldName + " must not be blank.");
            }
        }

        private static void requireAllowedEntity(String entity) {
            List<String> allowed = new ArrayList<>();
            allowed.add("PER");
            allowed.add("LOC");
            allowed.add("ORG");
            allowed.add("O");
            if (entity == null || !allowed.contains(entity)) {
                throw new IllegalArgumentException("entity must be one of PER, LOC, ORG, O.");
            }
        }

        private String toTsvLine() {
            return sentenceId + "\t" + token + "\t" + pos + "\t" + entity;
        }
    }
}
