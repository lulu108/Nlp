import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class Nlp4jSequenceLabelingDemo {

    private static final String HEADER = "sentence_id\ttoken\tpos\tentity";
    private static final Path DEFAULT_INPUT_IN_SUBDIR = Paths.get("input", "sample_input.txt");
    private static final Path DEFAULT_INPUT_IN_ROOT = Paths.get("sample_input.txt");
    private static final Path OUTPUT_PATH = Paths.get("output", "nlp4j_result.tsv");
    private static final Path DICT_DIR = Paths.get("dict");
    private static final Path PERSON_DICT = DICT_DIR.resolve("person.txt");
    private static final Path LOCATION_DICT = DICT_DIR.resolve("location.txt");
    private static final Path ORG_DICT = DICT_DIR.resolve("organization.txt");

    public static void main(String[] args) throws IOException {
        Path inputPath = resolveInputPath(args);
        List<String> sentences = readSentences(inputPath);
        DictionaryResources dictionaries = loadDictionaries();
        List<LabeledTokenRow> rows = runRuleBasedBaseline(sentences, dictionaries);
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

    private static DictionaryResources loadDictionaries() throws IOException {
        List<String> person = readDictLines(PERSON_DICT);
        List<String> location = readDictLines(LOCATION_DICT);
        List<String> organization = readDictLines(ORG_DICT);

        Map<String, String> entityMap = new HashMap<>();
        addToEntityMap(entityMap, person, "PER");
        addToEntityMap(entityMap, location, "LOC");
        addToEntityMap(entityMap, organization, "ORG");

        Map<Character, List<String>> byFirstChar = indexByFirstChar(entityMap.keySet());
        return new DictionaryResources(entityMap, byFirstChar);
    }

    private static List<String> readDictLines(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IllegalStateException("Dictionary not found: " + path.toAbsolutePath());
        }
        return Files.readAllLines(path, StandardCharsets.UTF_8).stream()
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .collect(Collectors.toList());
    }

    private static void addToEntityMap(Map<String, String> target, List<String> words, String entity) {
        for (String word : words) {
            target.put(word, entity);
        }
    }

    private static Map<Character, List<String>> indexByFirstChar(Set<String> words) {
        Map<Character, List<String>> index = new HashMap<>();
        for (String word : words) {
            if (word.isEmpty()) {
                continue;
            }
            char first = word.charAt(0);
            index.computeIfAbsent(first, key -> new ArrayList<>()).add(word);
        }
        for (List<String> bucket : index.values()) {
            bucket.sort(Comparator.comparingInt(String::length).reversed());
        }
        return index;
    }

    private static List<LabeledTokenRow> runRuleBasedBaseline(List<String> sentences, DictionaryResources dictionaries) {
        List<LabeledTokenRow> rows = new ArrayList<>();
        int sentenceId = 1;
        for (String sentence : sentences) {
            rows.addAll(tagSentence(sentenceId, sentence, dictionaries));
            sentenceId += 1;
        }
        return rows;
    }

    private static List<LabeledTokenRow> tagSentence(int sentenceId, String sentence, DictionaryResources dictionaries) {
        List<LabeledTokenRow> rows = new ArrayList<>();
        int index = 0;
        while (index < sentence.length()) {
            char current = sentence.charAt(index);
            if (Character.isWhitespace(current)) {
                index += 1;
                continue;
            }

            String match = longestMatch(sentence, index, dictionaries.byFirstChar());
            String token;
            String entity;

            if (match != null) {
                token = match;
                entity = dictionaries.entityMap().getOrDefault(match, "O");
                index += match.length();
            } else {
                token = String.valueOf(current);
                entity = "O";
                index += 1;
            }

            String pos = guessPos(token, entity);
            rows.add(new LabeledTokenRow(sentenceId, token, pos, entity));
        }
        return rows;
    }

    private static String longestMatch(String sentence, int start, Map<Character, List<String>> index) {
        char first = sentence.charAt(start);
        List<String> candidates = index.get(first);
        if (candidates == null) {
            return null;
        }
        for (String candidate : candidates) {
            if (sentence.startsWith(candidate, start)) {
                return candidate;
            }
        }
        return null;
    }

    private static String guessPos(String token, String entity) {
        if (!"O".equals(entity)) {
            return "PROPN";
        }
        if (ADP_WORDS.contains(token)) {
            return "ADP";
        }
        if (VERB_WORDS.contains(token)) {
            return "VERB";
        }
        if (token.length() >= 2) {
            return "NOUN";
        }
        return "X";
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

    private record DictionaryResources(Map<String, String> entityMap, Map<Character, List<String>> byFirstChar) {
    }

    private static final Set<String> ADP_WORDS = new HashSet<>(List.of("在", "于", "从", "到"));
    private static final Set<String> VERB_WORDS = new HashSet<>(List.of("参加", "发布", "毕业", "工作", "前往"));
}
