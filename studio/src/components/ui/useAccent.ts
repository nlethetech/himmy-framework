import { useEffect, useState } from "react";

export const ACCENTS: { name: string; value: string }[] = [
  { name: "Red", value: "#e06c75" }, // Odysseus default
  { name: "Blue", value: "#61afef" },
  { name: "Green", value: "#50fa7b" },
  { name: "Purple", value: "#c678dd" },
  { name: "Cyan", value: "#56b6c2" },
  { name: "Orange", value: "#e5a96b" },
];

function apply(value: string) {
  const r = document.documentElement;
  r.style.setProperty("--red", value);
  r.style.setProperty("--brand", value);
  r.style.setProperty("--accent", value);
}

// The brand accent — persisted, applied to the live token layer so the whole UI
// re-tints. Defaults to Odysseus red.
export function useAccent() {
  const [accent, setAccent] = useState<string>(
    () => localStorage.getItem("himmy-accent") || "#e06c75",
  );
  useEffect(() => {
    apply(accent);
    localStorage.setItem("himmy-accent", accent);
  }, [accent]);
  return { accent, setAccent };
}
