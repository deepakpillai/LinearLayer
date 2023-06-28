import matplotlib
import matplotlib.pyplot as plt

class PlotData():
  def plot_predictions(self, training_data_x=None, expected_training_data_y=None, testing_data_x=None, expected_testing_data_y=None, predictions=None):
    plt.figure(figsize=(5, 2.5))
    plt.scatter(training_data_x, expected_training_data_y, c='b', s=4, label="Training data")
    plt.scatter(testing_data_x, expected_testing_data_y, c='g', s=4, label="Testing data")
    if predictions is not None:
      plt.scatter(predictions, expected_testing_data_y, c='r', s=4, label='Predictions')
    plt.legend()
    plt.show()

  def plot_track_training_error(self, epoch, training_loss, eval_loss):
    plt.figure(figsize=(5,2.5))
    plt.plot(epoch, training_loss, c='g', label="Training loss")
    plt.plot(epoch, eval_loss, c='r', label="Testing loss")
    plt.legend()
    plt.show()
