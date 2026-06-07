import { Topbar, Page } from "../components/Page";
import { useTheme } from "../components/ui/useTheme";
import { useAccent, ACCENTS } from "../components/ui/useAccent";

export default function Theme() {
  const { theme, toggleTheme } = useTheme();
  const { accent, setAccent } = useAccent();

  return (
    <>
      <Topbar title="Theme" sub="Make Himmy yours" />
      <Page>
        <div className="card card-pad" style={{ maxWidth: 560, marginBottom: 16 }}>
          <span className="section-title">Appearance</span>
          <div className="theme-modes">
            {(["dark", "light"] as const).map((m) => (
              <button
                key={m}
                className={"theme-mode" + (theme === m ? " active" : "")}
                onClick={() => theme !== m && toggleTheme()}
              >
                <span className="theme-swatch" data-mode={m} />
                {m === "dark" ? "Dark" : "Light"}
              </button>
            ))}
          </div>
        </div>

        <div className="card card-pad" style={{ maxWidth: 560 }}>
          <span className="section-title">Accent</span>
          <div className="theme-accents">
            {ACCENTS.map((a) => (
              <button
                key={a.value}
                className={"theme-accent" + (accent === a.value ? " active" : "")}
                title={a.name}
                onClick={() => setAccent(a.value)}
                style={{ background: a.value }}
              />
            ))}
          </div>
          <div className="dim" style={{ marginTop: 10, fontSize: 12 }}>
            The accent re-tints the whole UI (brand, active states, links).
          </div>
        </div>
      </Page>
    </>
  );
}
