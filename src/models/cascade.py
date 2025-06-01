import torch
import torch.nn as nn
import torchvision.models as models

class IDKCascade:
    def __init__(self, base_model_name='resnet50', confidence_threshold=0.8):
        """
        Args:
            base_mode_name: name of the base model to use
            confidence_threshold: confidence threshold for the base model
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # Load pre-trained models
        if base_model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif base_model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        else: 
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        self.model = self.model.to(self.device)
    
    def predict(self, x):
        """
        Make predictions with the IDK cascade

        Args: 
            x (torch.Tensor): Input tensor

        Returns: 
            tuple: (predicitons, confidence, is_idk)
        """
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

            # Apply the IDK cascade
            is_idk = confidence < self.confidence_threshold

            return prediction, confidence, is_idk
    
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

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            predictions, confidences, is_idk = self.predict(inputs)

            idk_count += is_idk.sum().item()

            mask = ~is_idk
            if mask.any():
                correct += (predictions[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

            accuracy = correct / total if total > 0 else 0
            idk_rate = idk_count / len(dataloader.dataset)

            return { 
                'accuracy': accuracy,
                'idk_rate': idk_rate,
                'coverage': 1 - idk_rate
            }