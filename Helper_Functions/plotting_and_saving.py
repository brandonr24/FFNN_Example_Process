import matplotlib.pyplot as plt

def plot_and_save_model(x_data, y_data, file_naming):
    plt.plot(x_data.detach(), y_data.detach())
    plt.show()