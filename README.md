# 🌐 Network Traffic Anomaly Detection Using Machine Learning (Aligned with Zero Trust Security)

This project applies supervised machine learning to detect **anomalous network traffic** by analyzing protocol patterns from live packet captures. It supports **Zero Trust Architecture (ZTA)** by continuously verifying communication behavior within IoT or enterprise networks to identify threats before access is granted.

---

## 👩‍💻 Authors

**Aniyah Hall**  
B.S. in Computer Technology, Health Tech & Cybersecurity  
Bowie State University 

---

## 🎯 Objective

To develop a real-time network traffic monitoring tool that uses protocol statistics and machine learning to classify traffic as normal or anomalous. This supports **Zero Trust Security Implementation** by validating every network request and enforcing strict traffic policies.

---

## 🧠 Machine Learning Model Used

| Model               | Purpose         | Accuracy (Test) | Notes                                       |
|---------------------|------------------|------------------|----------------------------------------------|
| Logistic Regression | Classification   | ~87.5%           | Effective in identifying basic anomalies     |

---

## 🧪 Dataset

- **Source:** Real-time network traffic captured using `scapy`
- **Features:** Counts of protocol types (TCP, UDP, ICMP, etc.)
- **Labels:** Simulated (`normal` or `anomalous`) for evaluation
- **Split:** 80% training, 20% testing
- **Format:** Stored in `protocol_data.json` as input for ML model

---

## 📊 Outputs

- Accuracy and loss metrics (training vs test)
- Line plots comparing model performance
- JSON metrics file (`metrics.json`)
- Live protocol statistics from packet capture

---

## 🛡️ Zero Trust Security Integration

This project supports **Zero Trust principles**:

- **"Never trust, always verify"**: Each network event is inspected in real time
- **Behavior-based classification**: Protocol activity is compared against learned normal behavior
- **Access decisions**: Can integrate into firewalls or NAC systems to flag/deny suspicious traffic
- **Real-time enforcement**: Enhances Zero Trust policies for dynamic network security

---

## 🖥️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOURUSERNAME/iot-network-anomaly-ml.git
cd iot-network-anomaly-ml
2. Install Dependencies
pip install -r requirements.txt

3. Capture Network Traffic
python capture_packets.py
⚠️ Admin/root privileges may be required.

4. Train and Evaluate the Model
python train_model.py

5. Visualize Accuracy & Loss
python visualize_metrics.p
