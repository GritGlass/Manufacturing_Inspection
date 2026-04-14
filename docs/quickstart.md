## 0. Installation

- Install the required packages.

```bash
pip install -r requirements.txt

cd third_party/OpenXAI
pip install -e .
```

- `OpenXAI` installation is required for XAI features.
- Download an LLM model.
  - Save it in the `model` folder.
  - The model folder name must match the expected name exactly.
  - [Gemma](https://huggingface.co/google/gemma-4-E4B-it)
  - Other LLM models can also be used.

## 1. Database connection

- If no DB is connected, the app runs with the default sample data.
- Create the `.streamlit/secrets.toml` file.
- Add the DB connection settings as shown below. This app currently supports only Supabase.

```toml
[connections.supabase]
SUPABASE_URL = "http://000.0.0.0:12345"
SUPABASE_KEY = "abcdefghij"
```

- On the `Setting` page, click `DB Settings` to change the DB connection settings.

## 2. Dashboard page

- This page provides a summary view of the `Summary`, `Detail`, `Fine-tuning`, `Setting`, and `Log` pages.
- At the top of the page, set the data period and click `Load Data`. For sample data, keep the default date values.
- If you do not click `Load Data`, inference and other features will not run.

## 3. Summary page

- This page shows the data analysis summary.
- Click `Download Report` to save the current summary as a PDF file.

## 4. Detail page

- This page provides detailed AI inference results for images and related analysis features.
- Analysis starts only after you select images in `Select images`.
- Date and class filters are available.
- The `Result` tab shows the predicted class and image.
- The `3D Visualization` tab compresses the selected images into lower dimensions with PCA, t-SNE, or UMAP for visualization. It works only when at least 3 images are selected.
- The `XAI` tab uses OpenXAI to help interpret the model's predictions.

## 5. Fine-tuning

- This page provides classification model fine-tuning for the currently loaded images.
- In the left `Image Pool`, you can select the data to use for fine-tuning.
  - `Trained`: indicates whether the image has already been used for model training. When querying training images, only images with `trained = false` are loaded, so most values will be `false`.
  - `Predicted Class`: the class predicted by the AI model.
  - `Select`: the column used to choose images for fine-tuning.
- Check the label under the `Image Pool` panel, then click `Save labels`.
  - Double-click the `Label` field to edit it. Be careful with spelling.
- In the `Interactive Fine-tuning` panel, active learning can automatically select training images. `Random` and `Margin Sampling` are available.
  - Use the slider to set the sampling rate (%).
  - Click `Start active learning` to run it.
- In the same panel, `Manual settings` lets you configure `Epochs`, `Learning rate`, `Repeat count`, and `Preprocessing`.
  - Click `Start fine-tuning` to begin fine-tuning.
  - The trained model is saved in the `output` folder.

## 6. Setting

- `DB Settings`: input values for the DB connection keys.
- `LLM Runtime`: options for the LLM model used in the left sidebar and in Summary analysis.

## 7. Log

- You can check the logs generated while using the app.
- Logs are automatically saved by date in the `log` folder.

## 8. Sidebar: LLM

- You can interact with the LLM through the command box in the sidebar.
