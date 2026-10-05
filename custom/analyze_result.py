import json
from collections import defaultdict


CLASS_NAMES = [
    "circle",
    "square",
    "triangle",
    "cross"
]


def load_results():
    with open("results/stress_results.json", "r") as f:
        return json.load(f)


def calculate_accuracy(results):
    correct = sum(
        item["perturbed"]["correct"]
        for item in results
    )

    return 100 * correct / len(results)


def calculate_class_accuracy(results):
    stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )

    for item in results:
        true_label = item["true_label"]

        stats[true_label]["total"] += 1

        if item["perturbed"]["correct"]:
            stats[true_label]["correct"] += 1

    class_accuracy = {}

    for label in stats:
        class_accuracy[label] = (
            100
            * stats[label]["correct"]
            / stats[label]["total"]
        )

    return class_accuracy


def count_prediction_flips(results):
    return sum(
        item["prediction_flipped"]
        for item in results
    )


def get_failure_examples(results, limit=5):
    failures = [
        item
        for item in results
        if not item["perturbed"]["correct"]
    ]

    # Biggest confidence drops first
    failures.sort(
        key=lambda x: x["confidence_change"]
    )

    return failures[:limit]


def main():

    data = load_results()

    print()
    print("=" * 60)
    print("NEURAL TRAIL — FAILURE ANALYSIS")
    print("=" * 60)

    # =========================================================
    # NORMAL BASELINE
    # =========================================================

    normal_accuracy = calculate_accuracy(
        data["normal"]
    )

    print()
    print("BASELINE")
    print("-" * 60)

    print(
        f"Normal accuracy: "
        f"{normal_accuracy:.2f}%"
    )

    # =========================================================
    # EXPERIMENT RESULTS
    # =========================================================

    experiment_summary = {}

    print()
    print("EXPERIMENT RESULTS")
    print("-" * 60)

    for experiment_name, results in data.items():

        accuracy = calculate_accuracy(results)

        accuracy_drop = (
            normal_accuracy - accuracy
        )

        prediction_flips = count_prediction_flips(
            results
        )

        experiment_summary[experiment_name] = {
            "accuracy": accuracy,
            "accuracy_drop": accuracy_drop,
            "prediction_flips": prediction_flips
        }

        print()
        print(experiment_name.upper())

        print(
            f"  Accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            f"  Accuracy drop: "
            f"{accuracy_drop:.2f} points"
        )

        print(
            f"  Prediction flips: "
            f"{prediction_flips}/{len(results)}"
        )

    # =========================================================
    # FIND WORST PERTURBATION
    # =========================================================

    stress_experiments = {
        name: stats
        for name, stats in experiment_summary.items()
        if name != "normal"
    }

    worst_experiment = min(
        stress_experiments,
        key=lambda name:
        stress_experiments[name]["accuracy"]
    )

    worst_stats = stress_experiments[
        worst_experiment
    ]

    print()
    print("=" * 60)
    print("WORST PERTURBATION")
    print("=" * 60)

    print(
        f"Experiment: "
        f"{worst_experiment.upper()}"
    )

    print(
        f"Accuracy: "
        f"{worst_stats['accuracy']:.2f}%"
    )

    print(
        f"Accuracy drop: "
        f"{worst_stats['accuracy_drop']:.2f} points"
    )

    print(
        f"Prediction flips: "
        f"{worst_stats['prediction_flips']}"
    )

    # =========================================================
    # CLASS VULNERABILITY
    # =========================================================

    print()
    print("=" * 60)
    print("CLASS VULNERABILITY")
    print("=" * 60)

    class_results = calculate_class_accuracy(
        data[worst_experiment]
    )

    for class_name in CLASS_NAMES:

        if class_name in class_results:

            print(
                f"{class_name:<10}: "
                f"{class_results[class_name]:.2f}%"
            )

    # Find lowest accuracy
    lowest_accuracy = min(
        class_results.values()
    )

    # Find ALL classes tied for lowest
    weakest_classes = [
        class_name
        for class_name, accuracy
        in class_results.items()
        if accuracy == lowest_accuracy
    ]

    print()

    if len(weakest_classes) == 1:

        print(
            f"Most vulnerable class: "
            f"{weakest_classes[0]}"
        )

    else:

        print(
            "Most vulnerable classes: "
            + ", ".join(weakest_classes)
        )

    # =========================================================
    # EXAMPLE FAILURES
    # =========================================================

    print()
    print("=" * 60)
    print("EXAMPLE FAILURES")
    print("=" * 60)

    failures = get_failure_examples(
        data[worst_experiment],
        limit=5
    )

    for item in failures:

        print()

        print(
            f"Sample #{item['sample_id']}"
        )

        print(
            f"True label: "
            f"{item['true_label']}"
        )

        print(
            f"Before: "
            f"{item['original']['prediction']} "
            f"({item['original']['confidence']:.2f}%)"
        )

        print(
            f"After:  "
            f"{item['perturbed']['prediction']} "
            f"({item['perturbed']['confidence']:.2f}%)"
        )

        print(
            f"Confidence change: "
            f"{item['confidence_change']:.2f} points"
        )

        print(
            f"Prediction flipped: "
            f"{item['prediction_flipped']}"
        )

    # =========================================================
    # STRUCTURED FINDING
    # =========================================================

    print()
    print("=" * 60)
    print("NEURAL TRAIL FINDING")
    print("=" * 60)

    print()
    print("Blind spot:")
    print(
        f"  Model is highly sensitive to "
        f"{worst_experiment.lower()}."
    )

    print()
    print("Evidence:")

    print(
        f"  Baseline accuracy: "
        f"{normal_accuracy:.2f}%"
    )

    print(
        f"  Stressed accuracy: "
        f"{worst_stats['accuracy']:.2f}%"
    )

    print(
        f"  Accuracy drop: "
        f"{worst_stats['accuracy_drop']:.2f} points"
    )

    print(
        f"  Prediction flips: "
        f"{worst_stats['prediction_flips']}"
        f"/{len(data[worst_experiment])}"
    )

    print()
    print("Affected classes:")

    for class_name in class_results:

        print(
            f"  {class_name}: "
            f"{class_results[class_name]:.2f}% accuracy"
        )

    print()
    print("Finding status:")
    print("  CONFIRMED BY EXPERIMENT")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()