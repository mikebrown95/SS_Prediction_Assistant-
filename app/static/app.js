const form = document.querySelector("#prediction-form");
const error = document.querySelector("#error");
const probability = document.querySelector("#probability");
const band = document.querySelector("#band");
const meterFill = document.querySelector("#meter-fill");
const factors = document.querySelector("#factors");
const predictedLevel = document.querySelector("#predicted-level");
const awardLevels = document.querySelector("#award-levels");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";

  const data = Object.fromEntries(new FormData(form).entries());
  const payload = {
    age: Number(data.age),
    dateOfDisability: data.dateOfDisability,
    sex: data.sex,
    state: data.state,
    education: data.education,
    condition: data.condition,
    impairmentMonths: Number(data.impairmentMonths),
    yearsWorked: Number(data.yearsWorked),
    workLevel: data.workLevel,
    hasSpecialistEvidence: Boolean(data.hasSpecialistEvidence),
    hasRecentWorkAttempt: Boolean(data.hasRecentWorkAttempt),
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Prediction failed");
    }

    renderResult(result);
  } catch (err) {
    error.textContent = err.message;
  }
});

function renderResult(result) {
  const pct = Math.round(result.probability * 100);
  probability.textContent = `${pct}%`;
  band.textContent = `${result.band} estimated award probability`;
  meterFill.style.width = `${pct}%`;
  predictedLevel.textContent = result.predictedAwardLevel;

  factors.innerHTML = "";
  for (const factor of result.factors) {
    const item = document.createElement("li");
    item.innerHTML = `
      <span>
        <span class="factor-label">${escapeHtml(factor.label)}</span>
        <span class="factor-detail">${escapeHtml(factor.detail)}</span>
      </span>
      <span class="impact ${escapeHtml(factor.impact)}">${escapeHtml(factor.impact)}</span>
    `;
    factors.appendChild(item);
  }

  awardLevels.innerHTML = "";
  for (const level of result.awardLevels) {
    const item = document.createElement("li");
    const levelPct = Math.round(level.probability * 100);
    const timing = level.estimatedDecisionDate
      ? `${level.estimatedMonthsFromDisability} months • ${formatDate(level.estimatedDecisionDate)}`
      : `${level.estimatedMonthsFromDisability} months`;
    item.innerHTML = `
      <span>
        <span class="factor-label">${escapeHtml(level.label)}</span>
        <span class="factor-detail">${escapeHtml(level.detail)}</span>
        <span class="factor-detail">${escapeHtml(timing)}</span>
      </span>
      <span class="level-probability">${levelPct}%</span>
    `;
    awardLevels.appendChild(item);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  const date = new Date(`${value}T00:00:00`);
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
