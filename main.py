from src.models.cascade import IDKCascade
from src.utils.data_utils import get_data_loaders
from src.visualizations.plot_utils import plot_cascade_performance, plot_confidence_distribution

def main():

    cascade = IDKCascade(base_model_name='resnet50', confidence_threshold=0.8)

    # TODO: Change to your own dataset path
    dataset_path = 'data'
    train_loader, test_loader = get_data_loaders(dataset_path)

    metrics = cascade.evaluate(test_loader)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"IDK Rate: {metrics['idk_rate']:.3f}")
    print(f"Coverage: {metrics['coverage']:.3f}")

if __name__ == "__main__":
    main()