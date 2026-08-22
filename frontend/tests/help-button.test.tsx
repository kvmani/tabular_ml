import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HelpButton } from "../src/components/HelpButton";

describe("HelpButton", () => {
  it("opens and closes an accessible concept dialog", () => {
    render(<HelpButton title="Cross-validation"><p>Fold-local preprocessing avoids leakage.</p></HelpButton>);
    fireEvent.click(screen.getByRole("button", { name: /help: cross-validation/i }));
    expect(screen.getByRole("dialog", { name: "Cross-validation" })).toBeVisible();
    expect(screen.getByText(/avoids leakage/i)).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: /close help/i }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
