function initTabs() {
  document.querySelectorAll("[data-tabs]").forEach(root => {
    const tabs = root.querySelectorAll("[data-tab]");
    tabs.forEach(tab => tab.addEventListener("click", () => {
      tabs.forEach(t => {
        const on = t === tab;
        t.setAttribute("aria-selected", on);
        document.getElementById(t.dataset.tab).hidden = !on;
      });
    }));
  });
}

function initMath() {
  if (!window.katex) return;
  document.querySelectorAll(".eq[data-tex]").forEach(el =>
    katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false }));
  document.querySelectorAll(".m").forEach(el =>
    katex.render(el.textContent, el, { throwOnError: false }));
}

function initCopy() {
  const btn = document.querySelector(".bib .copy");
  btn.addEventListener("click", async () => {
    const text = document.getElementById("bib-text").textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Select & copy";
    }
    setTimeout(() => (btn.textContent = "Copy"), 1600);
  });
}

document.querySelectorAll(".btn[data-todo]").forEach(a =>
  a.addEventListener("click", e => e.preventDefault()));

initMath();
initTabs();
initCopy();
