
# Bypassing Google Colab GPU Limitations

This repository provides a solution for bypassing the GPU limitations in Google Colab by utilizing multiple Google accounts. By following this guide, you can continuously train your models using Colab GPUs.

## Solution Steps

1. **Create Multiple Google Accounts**: Set up 5 to 6 Google accounts.
2. **Share Project Folder**:
    - Create a folder in Google Drive.
    - Place the `.ipynb` file of your project in this folder.
    - Share the folder with all the accounts created in the previous step.
3. **Use Sample Code**:
    - Utilize the sample code provided in this repository to save and load checkpoints for your model, including the model itself, the optimizer, the scheduler, min-loss, or any other necessary data.
    - Ensure that the path for checkpoints is in your shared folder (from the second step).
4. **Run and Switch Accounts**:
    - Run the code on one of your accounts. When the session ends or you reach the GPU usage limit, switch to another account and run the code again.
    - The model will continue training from the last saved checkpoint.

This method allows you to train your model as long as you want by simply switching accounts every approximately 5 hours.

## Sample Code

The provided code sample demonstrates how to save and load checkpoints for your model. It includes:

- Mounting Google Drive
- Defining paths for saving the best model and checkpoints
- Loading checkpoints
- Training and evaluating the model and saving the checkpoints

### Code Example

```python

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from google.colab import drive

drive.mount('/content/drive')

BEST_MODEL_PATH = '/content/drive/MyDrive/MyModel/model_best.pt'
DRIVE_CHP_PATH = '/content/drive/MyDrive/MyModel/model_last_checkpoint.pth'


class Trainer():
    def __init__(self, config, dataset, checkpoint=None):
        # Initialization
        if checkpoint is None:
            self.model = self.get_model(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), config.lr, weight_decay=config.wd)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience=3, min_lr=1e-6, threshold=0.01)
            self.epoch, self.minloss = 0, float('inf')
        else:
            self.model, self.optimizer, self.scheduler, self.epoch, self.minloss = self.load_checkpoint(checkpoint)
            print('\rCheckpoint loaded successfully'); print('-' * 50)

    # Add other necessary methods here...

    def load_checkpoint(self, checkpoint):
        minloss = checkpoint['minloss']
        epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        return model, optimizer, scheduler, epoch, minloss

    def save_checkpoint(self, model, optimizer, scheduler, epoch, minloss, save_path):
        checkpoint = {
            'minloss': minloss,
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler}
        torch.save(checkpoint, save_path)


    def train_and_evaluate(self, n_epochs):

        loss_list =  []
        for epoch in range(self.epoch, n_epochs):

            # ...
            # ...

            train_loss = self.train(self.model, self.train_dataloader, self.optimizer, epoch, self.device)
            test_loss = self.evaluate(self.model, self.test_dataloader, self.device)

            # Save the best model based on minimum loss
            if test_loss < self.minloss:
                self.minloss = test_loss
                self.save_model(self.model, BEST_MODEL_PATH, epoch)

            # ...
            # ...

            self.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch+1, self.minloss, DRIVE_CHP_PATH )

            # ...
            # ...


if __name__ == '__main__':
    # Initialize trainer
    path = DRIVE_CHP_PATH
    checkpoint = torch.load(path) if os.path.isfile(path) else None
    trainer = Trainer(config, {'train_data': train_data, 'test_data': test_data}, checkpoint)

    # Train and evaluate the model
    trainer.train_and_evaluate(50)
```

## Contact

If you encounter any issues or have questions about using this solution, feel free to reach out.
