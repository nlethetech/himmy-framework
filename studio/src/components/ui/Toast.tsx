import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";

type ToastKind = "ok" | "err" | "info";
interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastApi {
  show: (message: string, kind?: ToastKind) => void;
}

const Ctx = createContext<ToastApi>({ show: () => {} });

export function useToast(): ToastApi {
  return useContext(Ctx);
}

// App-wide transient notifications. Replaces per-screen ad-hoc toast state.
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seq = useRef(0);

  const show = useCallback((message: string, kind: ToastKind = "info") => {
    const id = ++seq.current;
    setToasts((t) => [...t, { id, kind, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  }, []);

  return (
    <Ctx.Provider value={{ show }}>
      {children}
      <div className="toasts">
        {toasts.map((t) => (
          <div className={"toast " + t.kind} key={t.id}>
            {t.message}
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}
