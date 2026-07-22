import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  ReactNode,
  SelectHTMLAttributes
} from "react";

export function Button({
  active = false,
  className = "",
  tone = "default",
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
  tone?: "default" | "primary" | "danger";
}) {
  const classes = [
    "pt-button",
    tone !== "default" && `pt-button-${tone}`,
    active && "pt-button-active",
    className
  ].filter(Boolean).join(" ");
  return <button type={type} className={classes} {...props} />;
}

export function PanelHeader({ title, meta }: { title: string; meta?: ReactNode }) {
  return (
    <header className="pt-panel-header">
      <h2 className="pt-panel-title">{title}</h2>
      {meta}
    </header>
  );
}

export function Badge({
  children,
  tone = "default"
}: {
  children: ReactNode;
  tone?: "default" | "success" | "warning" | "danger";
}) {
  return (
    <span className={`pt-badge ${tone !== "default" ? `pt-badge-${tone}` : ""}`}>
      {children}
    </span>
  );
}

export function Notice({
  children,
  tone = "default",
  className = "",
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  tone?: "default" | "warning" | "danger";
}) {
  return (
    <div
      className={`pt-notice ${tone !== "default" ? `pt-notice-${tone}` : ""} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function Field({ label, children, hint }: {
  label: string;
  children: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={`pt-input ${props.className ?? ""}`} {...props} />;
}

export function Loading({ fullPage = false, label = "Loading" }: {
  fullPage?: boolean;
  label?: string;
}) {
  return (
    <div className={`loading ${fullPage ? "full-page" : ""}`} role="status">
      <span className="loading-rule" />
      {label}
    </div>
  );
}
