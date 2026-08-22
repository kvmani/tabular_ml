from __future__ import annotations

import io


def test_visualization_route_returns_chart_ready_data(client) -> None:
    rows = ["x,y,group"]
    rows.extend(f"{index},{index * 3},{'a' if index % 2 else 'b'}" for index in range(30))
    loaded = client.post(
        "/tabular_ml/api/v1/datasets/upload",
        data={"file": (io.BytesIO(("\n".join(rows) + "\n").encode()), "chart.csv")},
        content_type="multipart/form-data",
    )
    session_id = loaded.get_json()["data"]["session_id"]
    response = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/visualizations",
        json={"type": "scatter", "x": "x", "y": "y", "color": "group"},
    )
    data = response.get_json()["data"]
    assert response.status_code == 200
    assert data["type"] == "scatter"
    assert data["data"]["trendline"]["r2"] == 1.0
