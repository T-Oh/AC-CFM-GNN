import pickle
import matplotlib.pyplot as plt
import os

# Load the result dictionary from the pickle file
with open('results/results_.pkl', 'rb') as f:
    result = pickle.load(f)

# Ensure the directory exists for saving plots
os.makedirs('plots', exist_ok=True)

# Iterate over each metric in the result dictionary
for metric_name, values in result.items():
    # Check if the metric is a list of lists (multiclass metric)
    if isinstance(values[0], list):
        for class_idx in range(len(values[0])):
            # Extract the metric for the specific class
            class_values = [epoch[class_idx] for epoch in values]

            # Plot the metric
            plt.figure()
            plt.plot(class_values)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} for class {class_idx}')

            # Save the plot
            plot_filename = f'plots/{metric_name}_class_{class_idx}.png'
            plt.savefig(plot_filename)
            plt.close()
    else:
        # Plot the metric for a single list of values (non-multiclass metric)
        plt.figure()
        plt.plot(values)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name}')

        # Save the plot
        plot_filename = f'plots/{metric_name}.png'
        plt.savefig(plot_filename)
        plt.close()

print("Plots have been saved in the 'plots' directory.")
