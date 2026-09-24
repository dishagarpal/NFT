import json
import torch
import torch.nn as nn


# ============================================================
# GATING NETWORK
# ============================================================

class GatingNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD GATING MODEL
# ============================================================

print("Loading gating model...")

gate = GatingNetwork()

gate.load_state_dict(
    torch.load(
        "gating_model.pth",
        map_location="cpu"
    )
)

gate.eval()

print("Gating model loaded.")


# ============================================================
# LOAD MATCHED OBJECTS
# ============================================================

with open("matched_objects.json", "r") as f:
    groups = json.load(f)


final_detections = []


# ============================================================
# PROCESS EACH OBJECT GROUP
# ============================================================

for group_number, group in enumerate(groups, start=1):

    # --------------------------------------------------------
    # Find candidate classes
    # --------------------------------------------------------

    candidate_classes = sorted(
        set(
            detection["class_name"]
            for detection in group
        )
    )

    if not candidate_classes:
        continue


    class_results = []


    # ========================================================
    # EVALUATE EACH CANDIDATE CLASS
    # ========================================================

    for candidate_class in candidate_classes:

        a_conf = 0.0
        b_conf = 0.0
        c_conf = 0.0

        a_present = 0
        b_present = 0
        c_present = 0

        best_detection = {}


        # ----------------------------------------------------
        # Collect evidence for THIS class
        # ----------------------------------------------------

        for detection in group:

            expert = detection["expert"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            if class_name != candidate_class:
                continue


            if expert == "YOLO-A":

                if confidence > a_conf:
                    a_conf = confidence
                    best_detection["YOLO-A"] = detection

                a_present = 1


            elif expert == "YOLO-B":

                if confidence > b_conf:
                    b_conf = confidence
                    best_detection["YOLO-B"] = detection

                b_present = 1


            elif expert == "YOLO-C":

                if confidence > c_conf:
                    c_conf = confidence
                    best_detection["YOLO-C"] = detection

                c_present = 1


        # ----------------------------------------------------
        # Build class-specific features
        # ----------------------------------------------------

        features = torch.tensor(
            [[
                a_conf,
                b_conf,
                c_conf,
                a_present,
                b_present,
                c_present
            ]],
            dtype=torch.float32
        )


        # ----------------------------------------------------
        # Get expert weights for THIS class
        # ----------------------------------------------------

        with torch.no_grad():

            logits = gate(features)

            weights = torch.softmax(
                logits,
                dim=1
            )[0]


        weight_a = float(weights[0])
        weight_b = float(weights[1])
        weight_c = float(weights[2])


        # ----------------------------------------------------
        # Calculate class evidence
        #
        # Use strongest gated expert rather than summing
        # multiple weaker experts for competing classes.
        # ----------------------------------------------------

        expert_evidence = [

            weight_a * a_conf,

            weight_b * b_conf,

            weight_c * c_conf

        ]

        class_score = max(expert_evidence)


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        class_results.append({

            "class_name": candidate_class,

            "score": class_score,

            "expert_weights": {

                "YOLO-A": weight_a,
                "YOLO-B": weight_b,
                "YOLO-C": weight_c

            },

            "confidences": {

                "YOLO-A": a_conf,
                "YOLO-B": b_conf,
                "YOLO-C": c_conf

            },

            "best_detection": best_detection

        })


    # ========================================================
    # SELECT FINAL CLASS
    # ========================================================

    if not class_results:
        continue


    winner = max(
        class_results,
        key=lambda x: x["score"]
    )


    final_class = winner["class_name"]

    final_confidence = winner["score"]


    # ========================================================
    # FUSE BOXES FOR WINNING CLASS
    # ========================================================

    weighted_box = [
        0.0,
        0.0,
        0.0,
        0.0
    ]

    total_weight = 0.0


    for expert, detection in winner[
        "best_detection"
    ].items():

        expert_weight = winner[
            "expert_weights"
        ][expert]


        contribution = (
            expert_weight *
            detection["confidence"]
        )


        for i in range(4):

            weighted_box[i] += (
                contribution *
                detection["box"][i]
            )


        total_weight += contribution


    if total_weight > 0:

        weighted_box = [

            value / total_weight

            for value in weighted_box

        ]


    # ========================================================
    # SAVE FINAL DETECTION
    # ========================================================

    final_detection = {

        "object": final_class,

        "confidence": final_confidence,

        "box": weighted_box,

        "expert_weights":
            winner["expert_weights"],

        "class_scores": {

            result["class_name"]:
                result["score"]

            for result in class_results

        }

    }


    final_detections.append(
        final_detection
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n================================")
    print(
        f"Object Group {group_number}"
    )
    print("================================")


    print("\nCandidate classes:")

    for result in class_results:

        print(
            f"  {result['class_name']}: "
            f"{result['score']:.3f}"
        )


    print("\nClass-specific expert weights:")

    for result in class_results:

        print(
            f"\n  {result['class_name']}:"
        )

        print(
            f"    YOLO-A: "
            f"{result['expert_weights']['YOLO-A']:.3f}"
        )

        print(
            f"    YOLO-B: "
            f"{result['expert_weights']['YOLO-B']:.3f}"
        )

        print(
            f"    YOLO-C: "
            f"{result['expert_weights']['YOLO-C']:.3f}"
        )


    print("\nClass-specific confidences:")

    for result in class_results:

        print(
            f"  {result['class_name']}: "
            f"A={result['confidences']['YOLO-A']:.3f}, "
            f"B={result['confidences']['YOLO-B']:.3f}, "
            f"C={result['confidences']['YOLO-C']:.3f}"
        )


    print(
        f"\nFINAL OBJECT: "
        f"{final_class}"
    )

    print(
        f"FINAL CONFIDENCE: "
        f"{final_confidence:.3f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "final_detections.json",
    "w"
) as f:

    json.dump(
        final_detections,
        f,
        indent=4
    )


print("\n================================")
print("MOE DETECTION COMPLETE")
print("================================")

print(
    "Saved: final_detections.json"
)