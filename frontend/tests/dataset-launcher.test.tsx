import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DatasetLauncher } from "../src/features/DatasetLauncher";
import type { DatasetSummary } from "../src/types";

const dataset: DatasetSummary = {
  key: "penguins",
  name: "Palmer Penguins",
  task: "classification",
  target: "species",
  rows: 344,
  columns: 8,
  summary: "Measurements for three penguin species.",
  license: "CC0",
  tags: ["multiclass", "missing-values"],
};

describe("DatasetLauncher", () => {
  it("presents a curated dataset and launches it", () => {
    const onLoad = vi.fn();
    render(<DatasetLauncher datasets={[dataset]} onLoad={onLoad} onUpload={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "Palmer Penguins" })).toBeVisible();
    expect(screen.getByText("344")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: /open dataset/i }));
    expect(onLoad).toHaveBeenCalledWith("penguins");
  });

  it("accepts a dropped CSV file", () => {
    const onUpload = vi.fn();
    const { container } = render(<DatasetLauncher datasets={[]} onLoad={vi.fn()} onUpload={onUpload} />);
    const file = new File(["x,y\n1,2\n"], "sample.csv", { type: "text/csv" });
    const stage = container.querySelector(".upload-stage");
    expect(stage).not.toBeNull();
    fireEvent.drop(stage as Element, { dataTransfer: { files: [file] } });
    expect(onUpload).toHaveBeenCalledWith(file);
  });
});
