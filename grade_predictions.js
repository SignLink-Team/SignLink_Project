const fs = require("fs");

const inputPath = "C:/Users/win1/sign-language-backend/tmp_valid_predictions.json";
const outputPath = "C:/Users/win1/sign-language-backend/finetuned_valid_predictions_graded.csv";

const rows = JSON.parse(fs.readFileSync(inputPath, "utf8").replace(/^\uFEFF/, ""));

function clean(value) {
  return String(value || "")
    .normalize("NFC")
    .replace(/\([^)]*\)/g, "")
    .replace(/[0-9#]/g, "")
    .replace(/[^가-힣a-zA-Z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokens(value) {
  return clean(value).split(/\s+/).filter(Boolean);
}

function ngrams(value, n) {
  const text = clean(value).replace(/\s+/g, "");
  const result = [];
  for (let i = 0; i <= text.length - n; i += 1) {
    result.push(text.slice(i, i + n));
  }
  return result;
}

function f1(left, right) {
  if (!left.length || !right.length) return 0;
  const counts = new Map();
  for (const item of right) counts.set(item, (counts.get(item) || 0) + 1);

  let hits = 0;
  for (const item of left) {
    const count = counts.get(item) || 0;
    if (count > 0) {
      hits += 1;
      counts.set(item, count - 1);
    }
  }

  const precision = hits / left.length;
  const recall = hits / right.length;
  return precision && recall ? (2 * precision * recall) / (precision + recall) : 0;
}

function unique(items) {
  return [...new Set(items)];
}

function hasRepetition(value) {
  const text = String(value || "");
  return /(.)\1{5,}/.test(text) || text.includes("@@@");
}

function isKoreanLike(value) {
  return /[가-힣]/.test(String(value || ""));
}

function sentenceCount(value) {
  const text = String(value || "").trim();
  if (!text) return 0;
  const parts = text.split(/[.!?。！？]+\s*/).map((part) => part.trim()).filter(Boolean);
  return parts.length || 1;
}

function grade(row) {
  const pred = row.pred || "";
  const gold = row.gold || "";

  if (!pred.trim()) {
    return { grade: "E", reason: "빈 출력", score: 0, c2: 0, c3: 0, tf: 0 };
  }
  if (hasRepetition(pred)) {
    return { grade: "E", reason: "반복 토큰/문자 발생", score: 0, c2: 0, c3: 0, tf: 0 };
  }
  if (!isKoreanLike(pred)) {
    return { grade: "E", reason: "한글 문장 출력 실패", score: 0, c2: 0, c3: 0, tf: 0 };
  }

  const c2 = f1(ngrams(pred, 2), ngrams(gold, 2));
  const c3 = f1(ngrams(pred, 3), ngrams(gold, 3));
  const tf = f1(unique(tokens(pred)), unique(tokens(gold)));
  const score = 0.45 * c2 + 0.35 * c3 + 0.2 * tf;
  const tooLong = pred.length > gold.length * 1.8 && pred.length > 80;
  const oneOrTwoSentences = sentenceCount(pred) <= 2;

  if (score >= 0.46 && oneOrTwoSentences && !tooLong) {
    return { grade: "A", reason: "정답문과 의미/표현 유사도가 높음", score, c2, c3, tf };
  }
  if (score >= 0.31 && oneOrTwoSentences) {
    return { grade: "B", reason: "핵심 의미는 대체로 유사하나 표현 차이/어색함 있음", score, c2, c3, tf };
  }
  if (score >= 0.18) {
    return { grade: "C", reason: "일부 핵심 의미만 반영되거나 누락/추가 가능성 있음", score, c2, c3, tf };
  }
  return { grade: "D", reason: "정답문과 의미 겹침이 낮아 오역 가능성 높음", score, c2, c3, tf };
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

const graded = rows.map((row, index) => {
  const result = grade(row);
  return {
    index: index + 1,
    grade: result.grade,
    reason: result.reason,
    auto_score: result.score.toFixed(4),
    char_bigram_f1: result.c2.toFixed(4),
    char_trigram_f1: result.c3.toFixed(4),
    token_f1: result.tf.toFixed(4),
    ...row,
  };
});

const counts = { A: 0, B: 0, C: 0, D: 0, E: 0 };
for (const row of graded) counts[row.grade] += 1;

const headers = Object.keys(graded[0]);
const csv = [headers.join(",")]
  .concat(graded.map((row) => headers.map((header) => csvEscape(row[header])).join(",")))
  .join("\r\n");

fs.writeFileSync(outputPath, `\uFEFF${csv}`, "utf8");

console.log(JSON.stringify({
  total: graded.length,
  counts,
  percent: Object.fromEntries(
    Object.entries(counts).map(([key, value]) => [key, `${((value / graded.length) * 100).toFixed(1)}%`]),
  ),
  outputPath,
}, null, 2));
