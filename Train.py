import LinearRegressionModelClass
import torch
import PlotData

class Train():
  def __init__(self, training_data, expected_training_data, testing_data, expected_testing_data):
    torch.manual_seed(42)
    self.model = LinearRegressionModelClass.LinearRegressionModelClass()
    self.training_data = training_data
    self.expected_training_data = expected_training_data
    self.testing_data = testing_data
    self.expected_testing_data = expected_testing_data
    self.epoch = []
    self.training_loss = []
    self.testing_loss = []
    self.train_model()

    PlotData.PlotData().plot_track_training_error(self.epoch, torch.Tensor(self.training_loss).detach().numpy(), torch.Tensor(self.testing_loss).detach().numpy())

  def train_model(self):
    NUMBER_OF_EPOCHS = 2000
    loss_function = torch.nn.L1Loss()
    optimization = torch.optim.SGD(self.model.parameters(), lr=0.01)
    for epoch in range(0, NUMBER_OF_EPOCHS):
      self.model.train()
      predictions = self.model(self.training_data)
      loss = loss_function(predictions, self.expected_training_data)
      optimization.zero_grad()
      loss.backward()
      optimization.step()

      self.model.eval()
      with torch.inference_mode():
        eval_prediction = self.model(self.testing_data)
        eval_loss = loss_function(eval_prediction, self.expected_testing_data)

        if epoch % 100 == 0:
         self.epoch.append(epoch)
         self.training_loss.append(loss)
         self.testing_loss.append(eval_loss)

  def get_model(self):
    return self.model
