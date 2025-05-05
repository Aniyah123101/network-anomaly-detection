from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, DNSQR
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Global lists to store captured packet features and labels (protocol types)
packet_features = []
packet_labels = []

# A function to process each packet captured
def packet_handler(packet):
    # Check if the packet has an IP or ARP layer
    if packet.haslayer(IP) or packet.haslayer(ARP):
        src_port = packet.sport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0
        dst_port = packet.dport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0
        packet_size = len(packet)

        # Label packets based on the protocol
        if packet.haslayer(TCP):
            if packet[TCP].dport == 80 or packet[TCP].sport == 80:
                protocol = "HTTP"
            elif packet[TCP].dport == 443 or packet[TCP].sport == 443:
                protocol = "HTTPS"
            else:
                protocol = "TCP"
        elif packet.haslayer(UDP):
            if packet.haslayer(DNSQR):
                protocol = "DNS"
            else:
                protocol = "UDP"
        elif packet.haslayer(ICMP):
            protocol = "ICMP"
        elif packet.haslayer(ARP):
            protocol = "ARP"
        else:
            protocol = "OTHER"

        # Store the packet features and labels
        packet_features.append([src_port, dst_port, packet_size])
        packet_labels.append(protocol)


# Function to capture network traffic and train a machine learning model
def capture_and_train(interface, duration, intervals=5):
    accuracy_list = []
    loss_list = []

    # Capture packets
    print(f"Capturing traffic on interface: {interface}")

    # Capture traffic in intervals and perform model training
    for _ in range(duration // intervals):
        sniff(iface=interface, prn=packet_handler, timeout=intervals, store=0)

        # If enough packets are captured, train the model
        if len(packet_features) > 10:
            print("Training model on captured data...")

            # Convert labels to numeric format for classification
            label_map = {"TCP": 0, "UDP": 1, "ICMP": 2, "ARP": 3, "DNS": 4, "HTTP": 5, "HTTPS": 6, "OTHER": 7}
            y = [label_map[label] for label in packet_labels]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(packet_features, y, test_size=0.3, random_state=42)

            # Train a logistic regression model
            model = LogisticRegression(max_iter=2000)  # Increased max_iter to handle convergence warning
            model.fit(X_train, y_train)

            # Calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Ensure that all possible classes are considered, even if they don't appear in y_test
            all_possible_classes = list(label_map.values())  # List of all classes: [0, 1, 2, 3, 4, 5, 6, 7]

            # Adjust the predicted probabilities to include zero probability for missing classes
            y_pred_proba = model.predict_proba(X_test)
            if len(y_pred_proba[0]) != len(all_possible_classes):
                # Pad missing classes with zero probabilities
                missing_classes = set(all_possible_classes) - set(model.classes_)
                zero_probs = np.zeros((len(y_pred_proba), len(missing_classes)))
                y_pred_proba_padded = np.hstack([y_pred_proba, zero_probs])
            else:
                y_pred_proba_padded = y_pred_proba

            # Calculate log loss with the padded predicted probabilities
            loss = log_loss(y_test, y_pred_proba_padded, labels=all_possible_classes)

            print(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

            # Store accuracy and loss for plotting
            accuracy_list.append(accuracy)
            loss_list.append(loss)
        else:
            print("Not enough packets captured for model training.")
            return None, None

    return accuracy_list, loss_list


# Function to plot accuracy and loss
def plot_accuracy_loss(accuracy_list, loss_list):
    epochs = range(1, len(accuracy_list) + 1)
    plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy_list, marker='o', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_list, marker='o', color='red', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Time')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


# Main function to control the monitoring and plotting process
if __name__ == "__main__":
    # Specify the network interface to monitor (e.g., 'eth0' for Ethernet or 'en0' for macOS Wi-Fi)
    interface = input("Enter the network interface (e.g., eth0, en0): ")
    # Set the total duration for monitoring
    duration = int(input("Enter the total monitoring duration in seconds: "))

    # Capture traffic and train the model
    accuracy_list, loss_list = capture_and_train(interface, duration)

    # If model training was successful, plot the results
    if accuracy_list and loss_list:
        plot_accuracy_loss(accuracy_list, loss_list)
