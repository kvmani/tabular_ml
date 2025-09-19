# Frontend Smoke Checklist

Because the frontend runs fully offline, the quickest validation loop is manual. The steps below cover the primary workflows.

1. **Start services**
   ```bash
   python run_app.py &
   cd frontend && npm install && npm run dev -- --host 0.0.0.0 --port 5173
   ```
2. **Load homepage**
   - Navigate to `http://localhost:5173`.
   - Confirm the banner indicates file uploads are disabled (unless configured otherwise).
3. **Dataset registry**
   - In the Dataset card, choose a sample dataset such as *Titanic Survival* and confirm the preview + summary populate.
   - Observe the configuration card shows the dataset registry with ≥20 entries.
4. **Preprocess**
   - Use “Create train/validation/test split” with target `Survived`. Check that splits succeed and a split identifier appears.
5. **Train**
   - Select the split and algorithm `Random Forest`, then start training. Training metrics should render and `/runs/last` should update (visible via config card refresh).
6. **Evaluate**
   - Click “Evaluate latest model” to generate metrics and charts.

> Tip: the CLI (`python cli.py ...`) mirrors the same flows and is useful for automated smoke checks alongside this manual pass.
