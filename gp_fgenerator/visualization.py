
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt



#%%
def plot_contour(X, Y, Z, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(1,1,dpi=300)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_contourf_{label}_target.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_surface(X, Y, Z, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    cp = ax.plot_surface(X, Y, Z, cmap='viridis')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_contour3d_{label}_target.png')
    ax.view_init(45, -135)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_heatmap(X, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams.update({'font.size': 5})
    fig, ax = plt.subplots(figsize=(40, 40) ,dpi=300)
    ax = sns.heatmap(X, linewidth=0.1, cmap='viridis')
    path_dir = os.path.join(os.getcwd(), 'plots', f'{label}')
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_heatmap_{label}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF

#%%
def plot_barplot(data, path_dir, label=''):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(18, 3) ,dpi=300)
    plt.axhline(y=data['dist'].mean(), color='k', linestyle='--')
    sns.boxplot(data=data, x='label', y='dist', palette=sns.color_palette('tab10', 1))
    if not (os.path.isdir(path_dir)):
        os.makedirs(path_dir)
    filepath = os.path.join(path_dir, f'plot_barplot_{label}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
# END DEF