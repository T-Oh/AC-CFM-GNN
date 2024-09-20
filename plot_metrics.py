import pickle
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Ensure the directory exists for saving plots
os.makedirs('plots', exist_ok=True)

# Get all result files matching the pattern 'results_*.pkl'
result_files = glob.glob('results/results_*.pkl')

# Dictionary to store optimal values for each metric across all files
optimal_metrics = {}

# Iterate over each results file
for result_file in result_files:
    # Load the result dictionary from the pickle file
    with open(result_file, 'rb') as f:
        result = pickle.load(f)

    # Filter the result dictionary for metric names that start with 'train_'
    train_metrics = {k: v for k, v in result.items() if k.startswith('train_')}

    # Iterate over each training metric
    for train_metric_name, train_values in train_metrics.items():
        # Derive the corresponding test metric by replacing 'train_' with 'test_'
        test_metric_name = train_metric_name.replace('train_', 'test_')
        
        # Check if the corresponding test metric exists
        if test_metric_name in result:
            test_values = result[test_metric_name]

            # Get the actual metric name without 'train_' or 'test_' prefix
            metric_name = train_metric_name.replace('train_', '')

            # Initialize optimal metric list if not already present
            if metric_name not in optimal_metrics:
                optimal_metrics[metric_name] = []

            # Check if the metric is multiclass (a list of lists)
            if isinstance(train_values[0], list):
                # Prepare a figure for plotting all class curves
                plt.figure()

                # Iterate through the classes
                optim_values = []
                for class_idx in range(len(train_values[0])):
                    # Extract the metric values for the specific class across epochs
                    train_class_values = [epoch[class_idx] for epoch in train_values]
                    test_class_values = [epoch[class_idx] for epoch in test_values]

                    # Plot the class-specific train and test values
                    plt.plot(train_class_values, label=f'Train Class {class_idx}')
                    plt.plot(test_class_values, label=f'Test Class {class_idx}', linestyle='--')

                    # Get the optimal values (min for loss, max for accuracy-like metrics)
                    if 'loss' in metric_name.lower():
                        optimal_test_value = np.min(test_class_values)
                    else:
                        optimal_test_value = np.max(test_class_values)
                    optim_values.append(optimal_test_value)

                    

                # Store the optimal value for each class
                optimal_metrics[metric_name].append([optim_values])

                # Finalize the plot
                plt.xlabel('Epoch')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} for all classes')
                plt.legend()

                # Save the plot with the filename indicating the result file
                result_filename = os.path.splitext(os.path.basename(result_file))[0]
                plot_filename = f'plots/{result_filename}_{metric_name}_all_classes.png'
                plt.savefig(plot_filename)
                plt.close()
            else:
                # Plot the metric for a single list of values (non-multiclass metric)
                plt.figure()
                plt.plot(train_values, label='Train')
                plt.plot(test_values, label='Test')
                plt.xlabel('Epoch')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name}')
                plt.legend()

                # Save the plot with the filename indicating the result file
                result_filename = os.path.splitext(os.path.basename(result_file))[0]
                plot_filename = f'plots/{result_filename}_{metric_name}.png'
                plt.savefig(plot_filename)
                plt.close()

                # Get the optimal values (min for loss, max for accuracy-like metrics)
                if 'loss' in metric_name.lower():
                    optimal_test_value = np.min(test_values)
                else:
                    optimal_test_value = np.max(test_values)

                # Store the optimal value
                optimal_metrics[metric_name].append(optimal_test_value)

# Calculate and print the mean of the optimal values for each metric
for metric_name, optimal_values in optimal_metrics.items():
    # If the metric has multiple classes, we need to reshape the optimal values
    if isinstance(optimal_values[0], list):
        optimal_values = np.array(optimal_values)

    # Print the mean optimal value per class
    mean_optimal_value = np.mean(optimal_values, axis=0)
    print(f'Mean optimal {metric_name}: {mean_optimal_value}')
