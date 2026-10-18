const form = document.querySelector("#question-form");
const question = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const answerRegion = document.querySelector("#answer-region");
const formMessage = document.querySelector("#form-message");
const statusDot = document.querySelector("#status-dot");
const statusCopy = document.querySelector("#status-copy");

function clearRegion() {
  answerRegion.replaceChildren();
}

function showLoading() {
  clearRegion();
  const loading = document.createElement("div");
  loading.className = "loading";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  loading.append(spinner, document.createTextNode("公式ドキュメントを検索し、回答を作成しています…"));
  answerRegion.append(loading);
}

function showNotice(title, message, isError = false) {
  clearRegion();
  const notice = document.createElement("div");
  notice.className = `notice${isError ? " error" : ""}`;
  const heading = document.createElement("strong");
  heading.textContent = title;
  const body = document.createElement("span");
  body.textContent = message;
  notice.append(heading, body);
  answerRegion.append(notice);
}

function showAnswer(result) {
  clearRegion();
  const card = document.createElement("article");
  card.className = "answer-card";

  const main = document.createElement("div");
  main.className = "answer-main";
  const label = document.createElement("p");
  label.className = "answer-label";
  label.textContent = "SUPPORTLENS の回答";
  const text = document.createElement("p");
  text.className = "answer-text";
  text.textContent = result.answer;
  main.append(label, text);
  card.append(main);

  const sources = document.createElement("section");
  sources.className = "sources";
  const heading = document.createElement("h3");
  heading.className = "sources-heading";
  heading.append(document.createTextNode("参照した公式ドキュメント"));
  const hint = document.createElement("span");
  hint.textContent = "リンク先で詳細を確認できます";
  heading.append(hint);
  const list = document.createElement("div");
  list.className = "citation-list";

  for (const citation of result.citations ?? []) {
    const link = document.createElement("a");
    link.className = "citation";
    link.href = citation.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const title = document.createElement("span");
    title.className = "citation-title";
    title.textContent = citation.title;
    const excerpt = document.createElement("span");
    excerpt.className = "citation-excerpt";
    excerpt.textContent = citation.excerpt;
    const url = document.createElement("span");
    url.className = "citation-url";
    url.textContent = citation.url;
    link.append(title, excerpt, url);
    list.append(link);
  }
  sources.append(heading, list);
  card.append(sources);
  answerRegion.append(card);
}

async function updateConnectionStatus() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    if (health.status === "ok") {
      statusDot.className = "status-dot ready";
      statusCopy.textContent = "回答できます";
      return;
    }
    statusDot.className = "status-dot offline";
    statusCopy.textContent = "ナレッジを準備中";
  } catch {
    statusDot.className = "status-dot offline";
    statusCopy.textContent = "接続できません";
  }
}

async function askQuestion() {
  const value = question.value.trim();
  if (!value) {
    formMessage.textContent = "質問を入力してください。";
    formMessage.classList.add("error");
    question.focus();
    return;
  }

  submitButton.disabled = true;
  answerRegion.setAttribute("aria-busy", "true");
  formMessage.classList.remove("error");
  formMessage.textContent = "回答を準備しています…";
  showLoading();

  try {
    const response = await fetch("/v1/answers", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: value, mode: "hybrid", top_k: 3 }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result?.detail?.code || "request_failed");
    }
    if (result.status === "insufficient_evidence") {
      showNotice("十分な根拠を見つけられませんでした。", "質問の言い換えや、もう少し具体的な言葉で試してください。");
    } else {
      showAnswer(result);
    }
    formMessage.textContent = "⌘ + Enter でも送信できます";
  } catch (error) {
    const message = error instanceof Error && error.message === "dependency_unavailable"
      ? "ナレッジベースを準備中です。少し待ってからもう一度お試しください。"
      : "回答を取得できませんでした。ネットワークや起動状態を確認して、もう一度お試しください。";
    showNotice("回答を表示できませんでした。", message, true);
    formMessage.textContent = "送信に失敗しました。";
    formMessage.classList.add("error");
  } finally {
    submitButton.disabled = false;
    answerRegion.setAttribute("aria-busy", "false");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void askQuestion();
});

question.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    void askQuestion();
  }
});

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    question.value = button.dataset.question ?? "";
    question.focus();
  });
});

void updateConnectionStatus();
