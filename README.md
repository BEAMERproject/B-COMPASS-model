# BEAMER Model Framework API

REST API for the **BEAMER model framework**, providing patient segmentation based on the **B-COMPASS** model.

This repository contains a Flask service that uses machine learning to classify patients into segments and groups according to their psychological and demographic characteristics.

> **Public release note:** Environment-specific deployment details (e.g., internal production URLs and CI/CD deployment pipelines) are intentionally **not** included in this public version.

---

## Getting started

### Prerequisites

- Python 3.11+, or Docker

### Run locally

```bash
pip install -r requirements.txt
python main.py
```

The service will start on `http://localhost:5000`.

### Run with Docker

```bash
docker build -t beamer-api .
docker run -p 5000:5000 beamer-api
```

---

## API Endpoints

### `GET /` — Health check

Verifies that the service is running.

**Response**
```json
{"status": "ok"}
```

---

### `POST /compassmodel` — Patient segmentation

Generates the patient's **B-COMPASS** segmentation based on their characteristics.

#### Request body

| Parameter | Type | Description |
|---|---|---|
| `Acceptance` | `float` | Acceptance (e.g., 4.5) |
| `Control` | `float` | Perceived control (e.g., 6) |
| `Health_Consciousness` | `float` | Health consciousness (e.g., 4.5) |
| `Health_Priority` | `float` | Health priority (e.g., 6) |
| `Concern` | `float` | Concerns (e.g., 6) |
| `Age` | `int` | Patient age (e.g., 56) |

#### Example request

```bash
curl -X POST http://localhost:5000/compassmodel \
  -H "Content-Type: application/json" \
  -d '{
    "Acceptance": 4.5,
    "Control": 6,
    "Health_Consciousness": 4.5,
    "Health_Priority": 6,
    "Concern": 6,
    "Age": 56
  }'
```

#### Example response

```json
{
  "1st-level": "1",
  "2nd-level": "A",
  "3rd-level": "A",
  "Final_Segment": "1AA",
  "Group": "1"
}
```

#### Response fields

| Field | Description |
|---|---|
| `1st-level` | SEH segment number (1–4) |
| `2nd-level` | Sub-segment letter based on Health Consciousness / Concerns / Health Priority |
| `3rd-level` | Adherence tendency prediction |
| `Final_Segment` | Combined segment code (e.g., `1AA`) |
| `Group` | Final patient group (1–8) |

---

## Repository structure

```text
├── main.py                 # Flask application
├── requirements.txt        # Dependencies
├── Dockerfile              # Container definition
└── models/                 # ML models
   ├── BEAMER_pred_model_binary28_5varsAge.joblib  # Predictive model
   └── min_max_scaler.pkl                         # MinMax scaler
```

---

## Citation

If you use this work, please cite it using the metadata in [CITATION.cff](CITATION.cff).

---

## License

This project is licensed under **Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.

**In short:** you may **share** (copy and redistribute) the material with **attribution**, but you may **not** use it for **commercial purposes**, and you may **not distribute modified versions**.

See: https://creativecommons.org/licenses/by-nc-nd/4.0/

---

## Acknowledgements

Part of the work presented in this paper draws on knowledge, insights, and experiences in the BEAMER project. This project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking (JU) under grant agreement No 101034369. The JU receives support from the European Union's Horizon 2020 research and innovation programme, EFPIA and LINK2TRIALS BV. This communication reflects only the authors' views, and neither IMI2 JU, the European Union, EFPIA, or LINK2TRIALS BV are responsible for any use that may be made of the information contained herein.

---

## Disclaimer

This repository is provided "as is" for transparency and reuse under the license terms above. Outputs and segment labels should be interpreted in the context of the underlying model and input data.
