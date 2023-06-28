import Train
from pathlib import Path
import torch
import PlotData

class GenerateModel():
  def __init__(self):
    self.full_data = self.create_data()
    self.training_data, self.testing_data = self.segregate_data(self.full_data)
    #getting expected data
    weight = 0.7
    bias = 0.3
    self.expected_output_data = (weight * self.full_data) + bias
    self.expected_training_data, self.expected_testing_data = self.segregate_data(self.expected_output_data)

    model_obj = Train.Train(self.training_data, self.expected_training_data, self.testing_data, self.expected_testing_data)
    model = model_obj.get_model()
    print(model.state_dict())
    predictions = model(self.testing_data)
    predictions = torch.Tensor(predictions).detach().numpy()
    plot = PlotData.PlotData().plot_predictions(self.training_data, self.expected_training_data, self.testing_data, self.expected_testing_data, predictions)
    self.save_model(model)

  def create_data(self):
    start = 0
    end = 1
    step = 0.02
    return torch.arange(start, end, step, dtype=torch.float32).unsqueeze(dim=1)

  def segregate_data(self, data:torch.Tensor):
    training_test_data_split = int(len(data)*0.80) #80% we are using for training and 20% we are using for testing
    training_data = data[:training_test_data_split]
    testing_data = data[training_test_data_split:]
    return (training_data, testing_data)

  def save_model(self, model):
    model_path = Path('Models')
    model_path.mkdir(parents=True, exist_ok=True)

    model_file_name = 'LinearRegressionModelClass.pth'
    save_path = model_path/model_file_name
    torch.save(model, save_path)

  def load_model(self, file_path):
    return torch.load(file_path)