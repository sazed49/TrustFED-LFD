
# TrustFed

This repository provides the official PyTorch implementation of **TrustFed**, a lightweight and unsupervised federated learning (FL) defense framework designed to detect and mitigate **label-flipping (LF) attacks** in safety-critical environments, particularly **autonomous underground mining systems**.

---

## 📄 Paper

**Label-Flip Attack Detection via Trust-Weighted Aggregation in Federated Learning for Underground Mine Security**
(*Link will be updated upon publication*)

---

## 🚩 Motivation

Modern underground mines increasingly deploy:

* Autonomous haulage vehicles
* Robotic drilling systems
* Humanoid inspection robots
* Intelligent sensing platforms

These systems operate in **confined, GPS-denied, low-illumination tunnel environments** and rely on distributed perception models for navigation, hazard detection, and operational coordination.

While **Federated Learning (FL)** preserves data privacy by keeping raw operational data local, it introduces new attack surfaces. In particular, **label-flipping attacks** allow compromised clients to manipulate training labels and inject corrupted semantic information into the global model.

In underground mining, such corruption may lead to:

* Misclassification of navigation signs
* Unsafe vehicle redirection inside tunnel networks
* Incorrect hazard assessment
* Disrupted robotic coordination

TrustFed is designed to address these vulnerabilities under **heterogeneous, non-IID, and resource-constrained mining environments**.

---

## 🧠 Framework Overview

TrustFed enhances aggregation robustness through:

* 🔎 **Last-layer gradient norm filtering** to remove extreme anomalous updates
* 📊 **Class-aware gradient memory modeling** to identify suspicious target classes
* 📌 **Unsupervised clustering (KMeans)** in gradient feature space
* 📈 **Soft trust scoring based on cluster proximity**
* ⚖️ **Trust-weighted aggregation** instead of rigid client exclusion

Key properties:

* Does **not** require clean validation data
* Works under **strong non-IID distributions**
* Lightweight and suitable for bandwidth-constrained environments
* Theoretically grounded robustness guarantee

---

## 📊 Supported Datasets

TrustFed is evaluated on:

* **MNIST** – Auto-download
* **CIFAR-10** – Auto-download
* **CIFAR-100** – Auto-download
* **MineSigns (Proposed Dataset)**

### MineSigns

A real-world underground mining vision dataset containing **13 safety-critical signs** collected inside an experimental tunnel-based mine under:

* Low illumination
* Structural irregularities
* Narrow tunnel geometries
* Occlusion and environmental degradation

🔒 The dataset will be publicly released upon publication.

---

## ⚙️ Experimental Configuration

TrustFed supports:

* IID and NON-IID (Dirichlet-based) data distributions
* Extreme non-IID settings
* Configurable attacker ratio
* Custom source–target label-flip classes
* Multiple aggregation rules:

  * FedAvg
  * Median
  * Trimmed Mean
  * Krum
  * TrustFed (Proposed)

---

## ▶️ Running the Code

Each dataset has a corresponding experiment script or notebook.

Example configuration:

```python
RULE = "trustfed"
ATTACK_TYPE = "label_flipping"
attackers_ratio = 0.4
DD_TYPE = "NON_IID"
```

Run:

```bash
python experiment_federated.py
```

You can configure:

* Number of clients
* Local epochs
* Learning rate
* Dirichlet alpha
* Attack intensity
* Aggregation rule

---

## 📦 Dependencies

* Python 3.8+
* PyTorch 1.10+
* TensorFlow 2.x (for IMDB preprocessing)
* scikit-learn
* NumPy
* pandas
* matplotlib

Install:

```bash
pip install -r requirements.txt
```

---

## 📈 Key Contributions of TrustFed

* Mining-specific FL security focus
* Unsupervised LF attack detection
* Trust-weighted aggregation instead of binary filtering
* Robust under severe non-IID conditions
* Lower server aggregation complexity compared to clustering-heavy defenses

---

## 📜 Citation

If you use this repository in your research, please cite:

```bibtex
@article{rahman2026trustfed,
  title={Label-Flip Attack Detection via Trust-Weighted Aggregation in Federated Learning for Underground Mine Security},
  author={Rahman, Md. Sazedur and others},
  journal={Under Review},
  year={2026}
}
```

