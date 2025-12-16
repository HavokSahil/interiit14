import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_policy_engine_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define box styles
    box_style = dict(boxstyle="round,pad=0.5", ec="black", lw=2)
    
    # Colors
    color_catalog = '#ffcc99'  # Orange
    color_engine = '#99ccff'   # Blue
    color_rrm = '#99ff99'      # Green
    color_clients = '#ff99cc'  # Pink

    # Draw Nodes
    # SLO Catalog
    ax.text(6, 7, "SLO Catalog\n(YAML Config)", ha="center", va="center", size=12,
            bbox=dict(facecolor=color_catalog, **box_style))

    # Policy Engine
    ax.text(6, 4, "Policy Engine\n(Role Management\n& Compliance)", ha="center", va="center", size=14,
            bbox=dict(facecolor=color_engine, **box_style))

    # RRM Engine
    ax.text(10, 4, "RRM Engine\n(Optimization)", ha="center", va="center", size=12,
            bbox=dict(facecolor=color_rrm, **box_style))

    # Clients
    ax.text(2, 4, "Clients\n(Devices)", ha="center", va="center", size=12,
            bbox=dict(facecolor=color_clients, **box_style))

    # Draw Arrows
    arrow_props = dict(arrowstyle="->", lw=1.5, color="black")

    # Catalog -> Engine
    ax.annotate("", xy=(6, 4.8), xytext=(6, 6.2), arrowprops=arrow_props)
    ax.text(6.1, 5.5, "Load Roles & Rules", ha="left", va="center", size=10)

    # Clients -> Engine
    ax.annotate("", xy=(4.2, 4), xytext=(2.8, 4), arrowprops=arrow_props)
    ax.text(3.5, 4.2, "Register", ha="center", va="bottom", size=10)

    # Engine -> RRM (QoS Weights)
    ax.annotate("", xy=(8.8, 4.2), xytext=(7.8, 4.2), arrowprops=arrow_props)
    ax.text(8.3, 4.4, "QoS Weights\nConstraints", ha="center", va="bottom", size=10)

    # RRM -> Engine (Metrics)
    ax.annotate("", xy=(7.8, 3.8), xytext=(8.8, 3.8), arrowprops=arrow_props)
    ax.text(8.3, 3.4, "Metrics\n(RSSI, Retry)", ha="center", va="top", size=10)

    # Engine -> RRM (Violations) - Curved arrow
    # Using FancyArrowPatch for curve
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="red")
    # This is harder with simple annotate, stick to straight lines for simplicity or use specific patch
    
    # Internal Logic Bubble
    circle = patches.Circle((6, 2), radius=1.2, edgecolor='gray', facecolor='#f0f0f0', linestyle='--')
    ax.add_patch(circle)
    ax.text(6, 2, "Compliance Check:\nMetric vs Threshold", ha="center", va="center", size=9)
    
    # Arrow from Engine to Logic
    ax.annotate("", xy=(6, 3.2), xytext=(6, 3.2), arrowprops=dict(arrowstyle="-", lw=1)) # Just a connector visual
    ax.plot([6, 6], [3.2, 3.2], color='black') # visual anchor

    plt.title("Policy Engine Architecture & Data Flow", fontsize=16, pad=20)
    plt.tight_layout()
    
    output_path = "diagrams/policy_engine_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    import os
    if not os.path.exists("diagrams"):
        os.makedirs("diagrams")
    create_policy_engine_diagram()
