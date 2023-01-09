import matplotlib.pyplot as plt
import os
import math



def plot_images(x,y=None, indices='all', columns=12, x_size=1, y_size=1,
                cm='binary',y_padding=0.35, spines_alpha=1,
                fontsize=20, save_as='auto'):
    """
    Plot original images
    Show some images in a grid, with legends
    args:
        x             : images - Shapes must be (-1,lx,ly) (-1,lx,ly,1) or (-1,lx,ly,3)
        y             : real classes or labels or None (None)
        indices       : indices of images to show or None for all (None)
        columns       : number of columns (12)
        x_size,y_size : figure size (1), (1)
        cm            : Matplotlib color map (binary)
        y_padding     : Padding / rows (0.35)
        font_size     : Font size in px (20)
        save_as       : Filename to use if save figs is enable ('auto')
    """
    if indices=='all': indices=range(len(x))
    draw_labels = (y is not None)
    rows        = math.ceil(len(indices)/columns)
    fig=plt.figure(figsize=(columns*x_size, rows*(y_size+y_padding)))
    n=1
    for i in indices:
        axs=fig.add_subplot(rows, columns, n)
        n+=1
        # ---- Shape is (lx,ly)
        if len(x[i].shape)==2:
            xx=x[i]
        # ---- Shape is (lx,ly,n)
        if len(x[i].shape)==3:
            (lx,ly,lz)=x[i].shape
            if lz==1: 
                xx=x[i].reshape(lx,ly)
            else:
                xx=x[i]
        img=axs.imshow(xx,   cmap = cm, interpolation='lanczos')
        """
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        """
        
        for spine in ['bottom', 'left','top','right']:
            axs.spines[spine].set_visible(False)
        
        axs.set_yticks([])
        axs.set_xticks([])

        if draw_labels:
            axs.set_xlabel(y[n-2],fontsize=fontsize/2)

    # a de-commenter pour enregistrer les images generer par notre model 
    #save_fig(save_as)
    plt.show()