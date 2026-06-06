import { useEffect, useState } from "react";

type Theme = "dark" | "light";

// Theme is a `data-theme` attribute on <html>; the CSS does the rest via tokens.
// Persisted to localStorage; defaults to dark (the product's primary look).
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("himmy-theme");
    return saved === "light" ? "light" : "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("himmy-theme", theme);
  }, [theme]);

  return {
    theme,
    toggleTheme: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
  };
}
