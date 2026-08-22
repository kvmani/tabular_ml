import { describe, expect, it } from "vitest";
import { chartLayout } from "../src/components/ChartPanel";
import type { ChartSettings } from "../src/types";

const settings: ChartSettings = {
  title: "Custom scientific figure",
  palette: "Colorblind",
  fontFamily: "Arial, sans-serif",
  fontSize: 15,
  background: "#fefefe",
  showGrid: false,
  showLegend: false,
  showAnnotations: true,
  exportScale: 3,
};

describe("chartLayout", () => {
  it("applies publication and accessibility settings without losing axis labels", () => {
    const layout = chartLayout(
      "Default title",
      { xaxis: { title: { text: "Observed value" } } },
      settings,
    );
    expect(layout.title).toMatchObject({ text: "Custom scientific figure" });
    expect(layout.font).toMatchObject({ family: "Arial, sans-serif", size: 15 });
    expect(layout.paper_bgcolor).toBe("#fefefe");
    expect(layout.showlegend).toBe(false);
    expect(layout.xaxis).toMatchObject({
      title: { text: "Observed value" },
      gridcolor: "rgba(0,0,0,0)",
    });
  });
});
