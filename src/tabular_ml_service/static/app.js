const statusNode = document.querySelector("#service-status");

fetch("./api/v1/health", { headers: { Accept: "application/json" } })
  .then((response) => response.json())
  .then((payload) => {
    const data = payload.data;
    statusNode.textContent = `Local service ${data.status} · version ${data.version}`;
  })
  .catch(() => {
    statusNode.textContent = "The local API is unavailable. Restart the service and try again.";
  });

