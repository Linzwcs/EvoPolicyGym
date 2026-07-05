const LANGUAGES = ["en", "zh"];

let currentLang = initialLanguage();

init();

function init() {
  setupLanguageToggle();
  renderLanguage();
}

function setupLanguageToggle() {
  document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextLang = button.getAttribute("data-lang-toggle");
      if (!LANGUAGES.includes(nextLang) || nextLang === currentLang) return;
      currentLang = nextLang;
      try {
        window.localStorage.setItem("evopolicygym-protocol-lang", nextLang);
      } catch {
        // Ignore storage failures in restricted browsing contexts.
      }
      renderLanguage();
    });
  });
}

function renderLanguage() {
  document.documentElement.lang = currentLang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-lang-view]").forEach((article) => {
    article.hidden = article.getAttribute("data-lang-view") !== currentLang;
  });
  document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
    const isActive = button.getAttribute("data-lang-toggle") === currentLang;
    button.setAttribute("aria-pressed", String(isActive));
  });
}

function initialLanguage() {
  try {
    const saved = window.localStorage.getItem("evopolicygym-protocol-lang");
    if (LANGUAGES.includes(saved)) return saved;
  } catch {
    // Ignore storage failures in restricted browsing contexts.
  }

  const browserLang = navigator.language || "";
  return browserLang.toLowerCase().startsWith("zh") ? "zh" : "en";
}
