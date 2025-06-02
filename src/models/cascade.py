import torch
import torch.nn as nn
import torchvision.models as models

class IDKCascade:

    # Constructor
    def __init__(self, model_names=['mobilenet_v2', 'resnet50'], confidence_thresholds=[0.8, 0.7]):
        """
        Args:
            base_mode_name: name of the base model to use
            confidence_threshold: confidence threshold for the base model
        """
        if len(model_names) != len(confidence_thresholds):
            raise ValueError("Number of models must match number of confidence thresholds")

        self.confidence_thresholds = confidence_thresholds
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using {self.device.type} device")

        # Load pre-trained models
        self.models = nn.ModuleList()
        for model_name in model_names:
            if model_name == 'mobilenet_v2':
                model = models.mobilenet_v2(pretrained=True)
            elif model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            model = model.to(self.device)
            model.eval() # Set model to evaluation mode
            self.models.append(model)
        
    # Predict method
    def predict(self, x):
        """
        Make predictions with the IDK cascade

        Args: 
            x (torch.Tensor): Input tensor

        Returns: 
            tuple: (predicitons, confidence, is_idk)
        """
        
        with torch.no_grad():       # Disable gradient tracking
            x = x.to(self.device)   # Move input to device
            batch_size = x.size(0)
            
            # Initialize output tensors
            final_predictions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            final_confidences = torch.zeros(batch_size, device=self.device)
            is_idk = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            model_used = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            # Track samples that still need prediction
            remaining_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            # Try each model in the cascade
            for model_idx, (model, threshold) in enumerate(zip(self.models, self.confidence_thresholds)):
                if not remaining_mask.any():
                    break

                # Get predictions from current model
                outputs = model(x[remaining_mask])
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predictions = torch.max(probabilities, dim=1)

                # Update final predictions and confidences
                confident_mask = confidence >= threshold
                if confident_mask.any():
                    final_predictions[remaining_mask] = predictions[confident_mask]
                    final_confidences[remaining_mask] = confidence[confident_mask]
                    is_idk[remaining_mask] = False
                    model_used[remaining_mask] = model_idx

                    remaining_mask = ~confident_mask

            return final_predictions, final_confidences, is_idk, model_used
    
    # Evaluate method
    def evaluate(self, dataloader): 
        """
        Evaluate the cascade performance

        Args:
            dataloader: PyTorch DataLoader containing the test data

        Returns:
            dict: Performance metrics
        """
        correct = 0
        total = 0
        idk_count = 0
        model_usage = [0] * len(self.models)

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            predictions, confidences, is_idk, model_used = self.predict(inputs)

            # Count IDK cases
            idk_count += is_idk.sum().item()

            # Count correct predictions
            mask = ~is_idk
            if mask.any():
                correct += (predictions[mask] == labels[mask]).sum().item() # Count correct predictions
                total += mask.sum().item() # Count total predictions

            for i in range(len(self.models)):
                model_usage[i] += (model_used == i).sum().item()

            accuracy = correct / total if total > 0 else 0
            idk_rate = idk_count / len(dataloader.dataset)

            total_predictions = len(dataloader.dataset) - idk_count
            model_usage_percentages = [usage / total_predictions * 100 if total_predictions > 0 else 0 for usage in model_usage]

            return { 
                'accuracy': accuracy,
                'idk_rate': idk_rate,
                'coverage': 1 - idk_rate,
                'model_usage': model_usage_percentages
            }