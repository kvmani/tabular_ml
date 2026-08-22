import { CircleHelp, X } from "lucide-react";
import { useId, useState } from "react";

interface HelpButtonProps {
  title: string;
  children: React.ReactNode;
}

export function HelpButton({ title, children }: HelpButtonProps) {
  const [open, setOpen] = useState(false);
  const dialogId = useId();
  return (
    <>
      <button
        type="button"
        className="icon-button help-button"
        aria-label={`Help: ${title}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={dialogId}
        onClick={() => setOpen(true)}
      >
        <CircleHelp size={16} aria-hidden="true" />
      </button>
      {open ? (
        <div className="modal-backdrop" role="presentation">
          <section
            id={dialogId}
            className="modal help-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${dialogId}-title`}
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">CONCEPT GUIDE</p>
                <h2 id={`${dialogId}-title`}>{title}</h2>
              </div>
              <button
                type="button"
                className="icon-button"
                aria-label="Close help"
                onClick={() => setOpen(false)}
              >
                <X aria-hidden="true" />
              </button>
            </header>
            <div className="help-content">{children}</div>
            <footer className="modal-actions">
              <button
                type="button"
                className="button primary"
                onClick={() => setOpen(false)}
              >
                Got it
              </button>
            </footer>
          </section>
        </div>
      ) : null}
    </>
  );
}

