import torch
import matplotlib.pyplot as plt
import numpy as np

FOLDER = '/home/tohlinger/LEO/Running/TAG_Zhu_NR/results/'
output = np.array(torch.load(FOLDER + 'output.pt'))
labels = np.array(torch.load(FOLDER + 'labels.pt'))
XVALUES = np.arange(2000)

for i in range(int(len(output)/2000)):
    #Plot Vreal
    fig = plt.figure(i)
    ax = plt.gca()
    ax.bar(XVALUES, output[2000*i:2000*(i+1),0], label='Output')
    ax.bar(XVALUES, labels[2000*i:2000*(i+1),0], label='Labels')
    ax.set_title('Re(V)')

    ax.legend()
    fig.savefig(FOLDER+f'outputVSlabel{i}Vreal.png', bbox_inches='tight')

    #Plot Vimag
    fig = plt.figure(i+int(len(output)/2000))
    ax = plt.gca()
    ax.bar(XVALUES, output[2000*i:2000*(i+1),1], label='Output')
    ax.bar(XVALUES, labels[2000*i:2000*(i+1),1], label='Labels')
    ax.set_title('Re(V)')

    ax.legend()
    fig.savefig(FOLDER+f'outputVSlabel{i}Vimag.png', bbox_inches='tight')
    plt.close()

    #Plot Vreal diff
    fig = plt.figure(i+(int(len(output)/2000)*2))
    ax = plt.gca()
    ax.bar(XVALUES, output[2000*i:2000*(i+1),0]-labels[2000*i:2000*(i+1),0], label='Output')
    ax.set_title('Re(V)')

    ax.legend()
    fig.savefig(FOLDER+f'outputVSlabel{i}Vreal_dif.png', bbox_inches='tight')
    plt.close()

    #Plot Vimag diff
    fig = plt.figure(i+(int(len(output)/2000)*2))
    ax = plt.gca()
    ax.bar(XVALUES, output[2000*i:2000*(i+1),1]-labels[2000*i:2000*(i+1),1], label='Output')
    ax.set_title('Imag(V)')

    ax.legend()
    fig.savefig(FOLDER+f'outputVSlabel{i}Vimag_dif.png', bbox_inches='tight')
    plt.close()




