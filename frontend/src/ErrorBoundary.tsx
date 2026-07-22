import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Eyck frontend error", error, info);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <main className="fatal-page">
        <div className="pt-notice pt-notice-danger">
          <strong>The annotation interface encountered an error.</strong>
          <p>{this.state.error.message}</p>
          <button className="pt-button" type="button" onClick={() => window.location.reload()}>
            Reload interface
          </button>
        </div>
      </main>
    );
  }
}
