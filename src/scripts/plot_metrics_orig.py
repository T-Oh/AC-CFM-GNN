import pickle
import matplotlib.pyplot as plt
import os
import glob

# Ensure the directory exists for saving plots
os.makedirs('plots', exist_ok=True)

# Get all result files matching the pattern 'results_*.pkl'
result_files = glob.glob('results/results_*.pkl')

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

            # Check if the metric is a list of lists (multiclass metric)
            if isinstance(train_values[0], list):
                for class_idx in range(len(train_values[0])):
                    # Extract the metric for the specific class for both train and test
                    train_class_values = [epoch[class_idx] for epoch in train_values]
                    test_class_values = [epoch[class_idx] for epoch in test_values]

                    # Plot the metric for both train and test
                    plt.figure()
                    plt.plot(train_class_values, label='Train')
                    plt.plot(test_class_values, label='Test')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric_name)
                    plt.title(f'{metric_name} for class {class_idx}')
                    plt.legend()

                    # Save the plot with the filename indicating the result file
                    result_filename = os.path.splitext(os.path.basename(result_file))[0]
                    plot_filename = f'plots/{result_filename}_{metric_name}_class_{class_idx}.png'
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

print("Plots have been saved in the 'plots' directory.")
